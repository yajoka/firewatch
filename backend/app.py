from gettext import install
import numpy as np
from datetime import date, timedelta
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
import joblib
from fastapi import FastAPI
from provinces import PROVINCES

# Historie einmal laden
def load_history():
    return pd.read_csv("fires_history_all_clean.csv")

df_history = load_history()

app = FastAPI()

# --- Modell-Artefakte beim Serverstart laden ---
model = joblib.load("xgboost_final_model.joblib")
training_features = joblib.load("xgboost_features.joblib")
try:
    imputer = joblib.load("imputer.joblib")
except FileNotFoundError:
    imputer = None
    print("⚠️ imputer.joblib nicht gefunden – Imputation wird übersprungen (nur ok, wenn Training ohne NaNs robust war)")


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

#Koordinaten für Durchschnittsbildung
BC_COORDS = [
    {"name": "Cariboo", "lat": 52.15, "lon": -122.15},
    {"name": "Okanagan", "lat": 49.80, "lon": -119.60},
    {"name": "Peace River", "lat": 56.30, "lon": -121.00},
]


def fetch_weather_daily_single(lat, lon, start_date, end_date):
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
def debug_weather(
    province: str = "British Columbia",
    days: int = 10
):
    province_cfg = PROVINCES[province]

    daily_df = fetch_weather_daily_province(
        coords=province_cfg["coords"],
        start_date=date.today().isoformat(),
        end_date=(date.today() + timedelta(days=days)).isoformat()
    )

    return {
        "province": province,
        "days": days,
        "weather_stats": daily_df.describe().to_dict()
    }


def build_history_features(df_history, jurisdiction):
    df_j = df_history[df_history["Jurisdiction"] == jurisdiction]

    # 🔍 DEBUG – HIER
    print("\nDEBUG: letzte 15 historische Monate")
    print(df_j.sort_values(["Year", "Month"]).tail(15))

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

def build_region_features(jurisdiction):
    features = {col: 0 for col in training_features if col.startswith("REG_")}
    key = f"REG_{jurisdiction}"
    if key in features:
        features[key] = 1
    return features

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

df_history["Year"] = df_history["Year"].astype(int)
df_history["Month"] = df_history["Month"].astype(int)

def fetch_weather_daily_province(coords, start_date, end_date):
    """
    Holt tägliche Wetterdaten für mehrere Koordinaten
    und bildet den Mittelwert pro Tag.
    """
    dfs = []

    for c in coords:
        df = fetch_weather_daily_single(
            lat=c["lat"],
            lon=c["lon"],
            start_date=start_date,
            end_date=end_date
        )
        dfs.append(df)

    # Alle Punkte untereinander stacken
    df_all = pd.concat(dfs)

    # Mittelwert pro Tag (Index = Tag)
    df_mean = df_all.groupby(df_all.index).mean()

    return df_mean
#Frontend Integration
@app.get("/predict")
def predict(
    province: str = "British Columbia",
    days: int = 10
):
    # 🔹 Provinz-Konfiguration holen
    province_cfg = PROVINCES[province]

    # 🔹 Wetterdaten (Multi-Point!)
    daily_df = fetch_weather_daily_province(
        coords=province_cfg["coords"],
        start_date=date.today().isoformat(),
        end_date=(date.today() + timedelta(days=days)).isoformat()
    )

    # 🔹 Model-Input bauen
    X_pred = build_model_input(
        target_date=date.today(),
        daily_weather_df=daily_df,
        df_history=df_history,
        jurisdiction=province_cfg["jurisdiction"]
    )

    # 🔹 Feature-Alignment
    for col in training_features:
        if col not in X_pred.columns:
            X_pred[col] = 0

    X_pred = X_pred[training_features]

    if imputer is not None:
        X_pred[:] = imputer.transform(X_pred)

    # 🔹 Vorhersage
    monthly_pred = float(model.predict(X_pred)[0])
    scaled_pred = monthly_pred * (days / 30)

    return {
        "province": province,
        "days": days,
        "monthly_prediction": round(monthly_pred, 1),
        "days_prediction": round(scaled_pred, 1)
    }

