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
# 🔹 App & Modell laden
# =====================================================

app = FastAPI()

# Modell & Feature-Liste
model = joblib.load("xgboost_final_model_v3.joblib")
training_features = joblib.load("xgboost_features_v3.joblib")

# Finale Trainingsdaten (Single Source of Truth)
df_history = pd.read_csv("fires_history_all_clean_fixed.csv")

df_history["Year"] = df_history["Year"].astype(int)
df_history["Month"] = df_history["Month"].astype(int)
df_history["Jurisdiction"] = df_history["Jurisdiction"].astype(str)


# =====================================================
# 🔹 Open-Meteo Setup
# =====================================================

cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


# =====================================================
# 🔹 Wetter: letzte 14 Tage → Monatsproxy
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
        "temperature_2m_mean (°C)": daily.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m_mean (%)": daily.Variables(1).ValuesAsNumpy(),
        "et0_fao_evapotranspiration (mm)": daily.Variables(2).ValuesAsNumpy(),
    })


def fetch_weather_monthly_province(coords, days=30):
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    dfs = []

    for c in coords:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": c["lat"],
            "longitude": c["lon"],
            "daily": [
                "temperature_2m_mean",
                "temperature_2m_max",
                "relative_humidity_2m_mean",
                "relative_humidity_2m_min",
                "precipitation_sum",
                "precipitation_hours",
                "wind_speed_10m_mean",
                "wind_speed_10m_max",
                "wind_direction_10m_dominant",
                "et0_fao_evapotranspiration",
            ],
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "timezone": "UTC"
        }

        response = openmeteo.weather_api(url, params=params)[0]
        daily = response.Daily()

        df = pd.DataFrame({
            "temperature_2m_mean (°C)": daily.Variables(0).ValuesAsNumpy(),
            "temperature_2m_max (°C)": daily.Variables(1).ValuesAsNumpy(),
            "relative_humidity_2m_mean (%)": daily.Variables(2).ValuesAsNumpy(),
            "relative_humidity_2m_min (%)": daily.Variables(3).ValuesAsNumpy(),
            "precipitation_sum (mm)": daily.Variables(4).ValuesAsNumpy(),
            "precipitation_hours (h)": daily.Variables(5).ValuesAsNumpy(),
            "wind_speed_10m_mean (km/h)": daily.Variables(6).ValuesAsNumpy(),
            "wind_speed_10m_max (km/h)": daily.Variables(7).ValuesAsNumpy(),
            "wind_direction_10m_dominant (°)": daily.Variables(8).ValuesAsNumpy(),
            "et0_fao_evapotranspiration (mm)": daily.Variables(9).ValuesAsNumpy(),
        })

        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # 🔹 Monatsaggregation
    return {
        "temperature_2m_mean (°C)": df_all["temperature_2m_mean (°C)"].mean(),
        "temperature_2m_max (°C)": df_all["temperature_2m_max (°C)"].mean(),
        "relative_humidity_2m_mean (%)": df_all["relative_humidity_2m_mean (%)"].mean(),
        "relative_humidity_2m_min (%)": df_all["relative_humidity_2m_min (%)"].mean(),
        "precipitation_sum (mm)": df_all["precipitation_sum (mm)"].sum(),
        "precipitation_hours (h)": df_all["precipitation_hours (h)"].sum(),
        "wind_speed_10m_mean (km/h)": df_all["wind_speed_10m_mean (km/h)"].mean(),
        "wind_speed_10m_max (km/h)": df_all["wind_speed_10m_max (km/h)"].max(),
        "wind_direction_10m_dominant (°)": df_all["wind_direction_10m_dominant (°)"].mode().iloc[0],
        "et0_fao_evapotranspiration (mm)": df_all["et0_fao_evapotranspiration (mm)"].sum(),
    }


def build_seasonal_history_features(
    df_history: pd.DataFrame,
    jurisdiction: str,
    target_month: int
):
    """
    Saisonale Lags: gleiche Monate aus vergangenen Jahren
    (z. B. alle Januare)
    """

    df_month = (
        df_history[
            (df_history["Jurisdiction"] == jurisdiction) &
            (df_history["Month"] == target_month)
        ]
        .sort_values("Year")
    )

    if len(df_month) < 3:
        return {
            "Lag_1": 0.0,
            "Lag_2": 0.0,
            "Lag_3": 0.0,
            "RollMean_3": 0.0,
            "RollSum_3": 0.0,
        }

    last_vals = df_month["Number_fires"].iloc[-3:].values
    seasonal_vals = df_month["Number_fires"].iloc[-12:].values

    print("\n🔎 SEASONAL DEBUG")
    print(
        df_month[["Year", "Month", "Number_fires"]]
        .tail(6)
    )

    return {
        "Lag_1": float(last_vals[-1]),
        "Lag_2": float(last_vals[-2]),
        "Lag_3": float(last_vals[-3]),

        "RollMean_3": float(np.mean(last_vals)),
        "RollSum_3": float(np.sum(last_vals)),

        # 🔒 Langfristige saisonale Stabilisierung
        "RollMean_12": float(seasonal_vals.mean()),
        "RollSum_12": float(seasonal_vals.sum()),
    }


# =====================================================
# 🔹 Feature-Zusammenbau (trainingstreu!)
# =====================================================

def build_model_input(province: str, target_month: int):
    province_cfg = PROVINCES[province]
    jurisdiction = province_cfg["jurisdiction"]


    # 🔹 Saisonale History-Features
    history_feats = build_seasonal_history_features(
        df_history=df_history,
        jurisdiction=jurisdiction,
        target_month=target_month
    )

    # 🔹 Zeitfeatures
    time_feats = {
        "Year": 2022,
        "Month": target_month,
        "Month_sin": np.sin(2 * np.pi * target_month / 12),
        "Month_cos": np.cos(2 * np.pi * target_month / 12),
    }

    # 🔹 Wetter (14-Tage-Proxy)
    weather_feats = fetch_weather_monthly_province(
        coords=province_cfg["coords"]
    )

    # 🔹 Region One-Hot
    region_feats = {c: 0 for c in training_features if c.startswith("REG_")}
    key = f"REG_{jurisdiction}"
    if key in region_feats:
        region_feats[key] = 1

    # 🔹 Alles zusammenführen
    data = {}
    data.update(time_feats)
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
def predict(
    province: str = "British Columbia",
    month: int | None = None
):
    today = date.today()
    target_month = month if month is not None else today.month

    X_pred = build_model_input(
        province=province,
        target_month=target_month
    )

    monthly_pred = float(model.predict(X_pred)[0])

    print("\n🧠 MODEL INPUT DEBUG")
    print(X_pred.T)

    return {
        "province": province,
        "year": int(X_pred["Year"].iloc[0]),
        "month": int(X_pred["Month"].iloc[0]),
        "monthly_prediction": round(monthly_pred, 2)
    }