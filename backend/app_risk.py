import numpy as np
import pandas as pd
from datetime import date, timedelta
import joblib

import openmeteo_requests
import requests_cache
from retry_requests import retry

from fastapi import FastAPI
from provinces import PROVINCES


# =====================================================
# 🔹 App & Daten laden
# =====================================================

app = FastAPI()

def load_history():
    return pd.read_csv("fires_history_all_clean.csv")

df_history = load_history()
df_history["Year"] = df_history["Year"].astype(int)
df_history["Month"] = df_history["Month"].astype(int)

# Modell laden
model = joblib.load("xgboost_final_model.joblib")
training_features = joblib.load("xgboost_features.joblib")


# =====================================================
# 🔹 Open-Meteo Setup
# =====================================================

cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


# =====================================================
# 🔹 Wetterfunktionen
# =====================================================

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
        "timezone": "UTC"
    }

    response = openmeteo.weather_api(url, params=params)[0]
    daily = response.Daily()

    return pd.DataFrame({
        "temperature_2m_mean": daily.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m_mean": daily.Variables(1).ValuesAsNumpy(),
        "et0_fao_evapotranspiration": daily.Variables(2).ValuesAsNumpy(),
    })


def fetch_weather_monthly_province(coords):
    """
    Holt Wetterdaten der letzten 14 Tage
    über mehrere Koordinaten und aggregiert sie
    zu Monats-Features.
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=14)

    dfs = []
    for c in coords:
        df = fetch_weather_daily_single(
            lat=c["lat"],
            lon=c["lon"],
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        )
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    return {
        "temperature_2m_mean (°C)": df_all["temperature_2m_mean"].mean(),
        "relative_humidity_2m_mean (%)": df_all["relative_humidity_2m_mean"].mean(),
        "et0_fao_evapotranspiration (mm)": df_all["et0_fao_evapotranspiration"].mean(),
    }


# =====================================================
# 🔹 Feature-Building
# =====================================================

def build_time_features(target_date):
    month = target_date.month
    return {
        "Year": target_date.year,
        "Month": month,
        "Month_sin": np.sin(2 * np.pi * month / 12),
        "Month_cos": np.cos(2 * np.pi * month / 12),
    }


def build_history_features(df_history, jurisdiction):
    df = (
        df_history[df_history["Jurisdiction"] == jurisdiction]
        .sort_values(["Year", "Month"])
        .tail(3)
    )

    if len(df) < 3:
        return {
            "Lag_1": 0.0,
            "Lag_2": 0.0,
            "Lag_3": 0.0,
        }

    nf = df["Number_fires"].values

    return {
        "Lag_1": float(nf[-1]),
        "Lag_2": float(nf[-2]),
        "Lag_3": float(nf[-3]),
    }


def build_region_features(jurisdiction):
    features = {c: 0 for c in training_features if c.startswith("REG_")}
    key = f"REG_{jurisdiction}"
    if key in features:
        features[key] = 1
    return features


def build_model_input(target_date, weather_feats, history_feats, region_feats):
    data = {}
    data.update(build_time_features(target_date))
    data.update(weather_feats)
    data.update(history_feats)
    data.update(region_feats)

    X = pd.DataFrame([data])
    X = X.reindex(columns=training_features, fill_value=0)

    return X


# =====================================================
# 🔹 API Endpoint: Monatsprognose
# =====================================================

@app.get("/predict")
def predict(province: str = "British Columbia"):
    province_cfg = PROVINCES[province]

    target_date = date.today()

    weather_feats = fetch_weather_monthly_province(
        coords=province_cfg["coords"]
    )

    history_feats = build_history_features(
        df_history,
        province_cfg["jurisdiction"]
    )

    region_feats = build_region_features(
        province_cfg["jurisdiction"]
    )

    X_pred = build_model_input(
        target_date,
        weather_feats,
        history_feats,
        region_feats
    )

    monthly_pred = float(model.predict(X_pred)[0])

    return {
        "province": province,
        "year": target_date.year,
        "month": target_date.month,
        "monthly_prediction": round(monthly_pred, 2)
    }