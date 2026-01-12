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
# 🔹 App initialisieren
# =====================================================

app = FastAPI()

# =====================================================
# 🔹 Historische Branddaten laden
# =====================================================

def load_history():
    df = pd.read_csv("fires_history_all_clean_fixed.csv")
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    return df

df_history = load_history()

# =====================================================
# 🔹 Modell & Feature-Liste laden (V4!)
# =====================================================

model = joblib.load("xgboost_final_model_v4.joblib")
training_features = joblib.load("xgboost_features_v4.joblib")

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
            "temperature_2m_max",
            "relative_humidity_2m_mean",
            "relative_humidity_2m_min",
            "precipitation_sum",
            "precipitation_hours",
            "wind_speed_10m_mean",
            "wind_speed_10m_max",
            "wind_direction_10m_dominant",
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


def fetch_weather_monthly_province(coords):
    """
    Aggregiert Wetter der letzten 14 Tage über mehrere Koordinaten
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=14)

    dfs = []
    for c in coords:
        dfs.append(
            fetch_weather_daily_single(
                c["lat"], c["lon"],
                start_date.isoformat(),
                end_date.isoformat()
            )
        )

    df_all = pd.concat(dfs, ignore_index=True)

    return df_all.mean().to_dict()

# =====================================================
# 🔹 Zeit- & Saisonfeatures
# =====================================================

def build_time_features(target_date):
    month = target_date.month

    return {
        "Year": target_date.year,
        "Month": month,
        "Month_sin": np.sin(2 * np.pi * month / 12),
        "Month_cos": np.cos(2 * np.pi * month / 12),
        "is_summer": 1 if month in [6, 7, 8, 9] else 0,
    }

# =====================================================
# 🔹 Saisonale Historienfeatures (KERNSTÜCK)
# =====================================================

def build_seasonal_history_features(df, jurisdiction, target_month):
    df_m = (
        df[
            (df["Jurisdiction"] == jurisdiction) &
            (df["Month"] == target_month)
        ]
        .sort_values("Year")
    )

    if len(df_m) < 3:
        return {
            "Lag_1": 0.0,
            "Lag_2": 0.0,
            "Lag_3": 0.0,
            "RollMean_3": 0.0,
            "RollStd_3": 0.0,
            "RollSum_3": 0.0,
        }

    vals = df_m["Number_fires"].iloc[-3:].values

    return {
        "Lag_1": float(vals[-1]),
        "Lag_2": float(vals[-2]),
        "Lag_3": float(vals[-3]),
        "RollMean_3": float(np.mean(vals)),
        "RollStd_3": float(np.std(vals)),
        "RollSum_3": float(np.sum(vals)),
    }

# =====================================================
# 🔹 Sommer-Features (nur aktiv im Sommer!)
# =====================================================

def build_summer_features(is_summer, weather_feats, lag1):
    if not is_summer:
        return {
            "summer_temp": 0.0,
            "summer_dryness": 0.0,
            "summer_lag": 0.0,
            "pre_summer_heat": 0.0,
            "spring_dryness": 0.0,
        }

    return {
        "summer_temp": weather_feats["temperature_2m_mean (°C)"],
        "summer_dryness": weather_feats["et0_fao_evapotranspiration (mm)"],
        "summer_lag": lag1,
        "pre_summer_heat": weather_feats["temperature_2m_max (°C)"],
        "spring_dryness": weather_feats["relative_humidity_2m_min (%)"],
    }

# =====================================================
# 🔹 Region One-Hot
# =====================================================

def build_region_features(jurisdiction):
    feats = {f: 0 for f in training_features if f.startswith("REG_")}
    key = f"REG_{jurisdiction}"
    if key in feats:
        feats[key] = 1
    return feats

# =====================================================
# 🔹 Model Input bauen
# =====================================================

def build_model_input(target_date, province_cfg):
    weather_feats = fetch_weather_monthly_province(province_cfg["coords"])
    time_feats = build_time_features(target_date)

    hist_feats = build_seasonal_history_features(
        df_history,
        province_cfg["jurisdiction"],
        target_date.month
    )

    summer_feats = build_summer_features(
        time_feats["is_summer"],
        weather_feats,
        hist_feats["Lag_1"]
    )

    region_feats = build_region_features(province_cfg["jurisdiction"])

    data = {}
    data.update(time_feats)
    data.update(weather_feats)
    data.update(hist_feats)
    data.update(summer_feats)
    data.update(region_feats)

    X = pd.DataFrame([data])
    X = X.reindex(columns=training_features, fill_value=0)

    return X

# =====================================================
# 🔹 API Endpoint
# =====================================================
@app.get("/predict")
def predict(
    province: str = "British Columbia",
    year: int = date.today().year,
    month: int = date.today().month
):
    province_cfg = PROVINCES[province]
    target_date = date(year, month, 15)

    X_pred = build_model_input(target_date, province_cfg)

    monthly_pred = max(0.0, float(model.predict(X_pred)[0]))

    return {
        "province": province,
        "year": target_date.year,
        "month": target_date.month,
        "monthly_prediction": round(monthly_pred, 2)
    }