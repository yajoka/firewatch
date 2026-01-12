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

def build_history_features(df_history, jurisdiction, target_month):
    df = (
        df_history[df_history["Jurisdiction"] == jurisdiction]
        .sort_values(["Year", "Month"])
    )

    # 🔹 Winterlogik
    if target_month in [12, 1, 2]:
        df_season = df[df["Month"].isin([12, 1, 2])]
    else:
        df_season = df

    df_season = df_season.tail(3)
    nf = df_season["Number_fires"].values

    if len(nf) < 3:
        return {
            "Lag_1": 0.0,
            "Lag_2": 0.0,
            "Lag_3": 0.0,
            "RollMean_3": 0.0,
            "RollSum_3": 0.0,
        }

    return {
        "Lag_1": float(nf[-1]),
        "Lag_2": float(nf[-2]),
        "Lag_3": float(nf[-3]),
        "RollMean_3": float(np.mean(nf)),
        "RollSum_3": float(np.sum(nf)),
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
    data.update(
        build_history_features(
            df_history,
            jurisdiction,
            target_date.month
        )
    )
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

    # 🔹 Vorhersage
    monthly_pred = float(model.predict(X_pred)[0])
    scaled_pred = monthly_pred * (days / 30)

    return {
        "province": province,
        "days": days,
        "monthly_prediction": round(monthly_pred, 1),
        "days_prediction": round(scaled_pred, 1)
    }