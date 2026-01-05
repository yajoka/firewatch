from gettext import install
import numpy as np
from datetime import date
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import joblib

from fastapi import FastAPI

# Historie einmal laden
def load_history_bc():
    return pd.read_csv("fires_history_bc.csv")
app = FastAPI()

# --- Modell-Artefakte beim Serverstart laden ---
model = joblib.load("xgboost_final_model.joblib")
training_features = joblib.load("xgboost_features.joblib")
imputer = joblib.load("imputer.joblib")


def build_time_features(target_date):
    year = target_date.year
    month = target_date.month

    return {
        "Year": year,
        "Month": month,
        "Month_sin": np.sin(2 * np.pi * month / 12),
        "Month_cos": np.cos(2 * np.pi * month / 12),
    }

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def fetch_weather_daily(lat, lon, start_date, end_date):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "et0_fao_evapotranspiration"
        ],
        "start_date": start_date,
        "end_date": end_date,
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    daily = response.Daily()

    df = pd.DataFrame({
        "temperature_2m_mean": daily.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m_mean": daily.Variables(1).ValuesAsNumpy(),
        "et0_fao_evapotranspiration": daily.Variables(2).ValuesAsNumpy(),
    })

    return df

def build_weather_features(daily_df):
    return {
        "temperature_2m_mean (°C)": daily_df["temperature_2m_mean"].mean(),
        "relative_humidity_2m_mean (%)": daily_df["relative_humidity_2m_mean"].mean(),
        "et0_fao_evapotranspiration (mm)": daily_df["et0_fao_evapotranspiration"].mean(),
    }


@app.get("/debug-weather")
def debug_weather():
    daily_df = fetch_weather_daily(
        lat=52.15,
        lon=-122.15,
        start_date="2026-01-01",
        end_date="2026-01-10"
    )

    weather_features = build_weather_features(daily_df)

    return weather_features


def build_history_features(df_history, jurisdiction):
    df = (
        df_history[df_history["Jurisdiction"] == jurisdiction]
        .sort_values(["Year", "Month"])
        .tail(12)   # max. Fenster
    )

    nf = df["Number_fires"]

    return {
        "Lag_1": nf.iloc[-1],
        "Lag_2": nf.iloc[-2],
        "Lag_3": nf.iloc[-3],

        "RollMean_3": nf.tail(3).mean(),
        "RollMean_6": nf.tail(6).mean(),
        "RollMean_12": nf.tail(12).mean(),

        "RollStd_3": nf.tail(3).std(),
        "RollStd_6": nf.tail(6).std(),
        "RollStd_12": nf.tail(12).std(),

        "RollSum_3": nf.tail(3).sum(),
        "RollSum_6": nf.tail(6).sum(),
        "RollSum_12": nf.tail(12).sum(),
    }

reg_cols = [
    "REG_British_Columbia"
]

def build_region_features(jurisdiction):
    return {"REG_British_Columbia": 1}

#Historie laden
df_history = load_history_bc()

# Finalisierung
def build_model_input(
    target_date,
    daily_weather_df,
    df_history,
    jurisdiction
):
    data = {}

    data.update(build_time_features(target_date))
    data.update(build_weather_features(daily_weather_df))
    data.update(build_history_features(df_history, jurisdiction))
    data.update(build_region_features(jurisdiction))

    X_pred = pd.DataFrame([data])
    return X_pred


if __name__ == "__main__":
    # 1) Wetter holen (10 Tage)
    daily_df = fetch_weather_daily(
        lat=52.15,
        lon=-122.15,
        start_date="2026-01-01",
        end_date="2026-01-10"
    )

    # 3) Model-Input bauen
    X_pred = build_model_input(
        target_date=date(2026, 1, 21),
        daily_weather_df=daily_df,
        df_history=df_history,
        jurisdiction="British Columbia"
    )

    print("\nFINAL MODEL INPUT")
    print(X_pred)
    print("\nNaNs:")
    print(X_pred.isna().sum())

    # Fehlende Spalten ergänzen
    for col in training_features:
        if col not in X_pred.columns:
            X_pred[col] = 0

    # Exakte Reihenfolge erzwingen
    X_pred = X_pred[training_features]

    # --- Imputer anwenden ---
    X_pred[:] = imputer.transform(X_pred)

    # --- Vorhersage ---
    prediction = model.predict(X_pred)

    # --- 10-Tage-Skalierung ---
    days_in_month = 31
    pred_10_days = prediction[0] * (10 / days_in_month)

    print("\nMONATS-PROGNOSE:", prediction[0])
    print("10-TAGE-PROGNOSE:", pred_10_days)