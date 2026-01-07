from fastapi import FastAPI, HTTPException
from datetime import date, timedelta
import numpy as np
import pandas as pd
import joblib
import openmeteo_requests
import requests_cache
from retry_requests import retry

from provinces import PROVINCES

app = FastAPI()

# --------------------------------------------------
# 🔹 Lade Modell & Features (IDENTISCH zu app.py)
# --------------------------------------------------
model = joblib.load("xgboost_final_model.joblib")
training_features = joblib.load("xgboost_features.joblib")

# --------------------------------------------------
# 🔹 Historische Daten (alle Provinzen)
# --------------------------------------------------
df_history = pd.read_csv("fires_history_all_clean.csv")
df_history["Year"] = df_history["Year"].astype(int)
df_history["Month"] = df_history["Month"].astype(int)
df_history["Number_fires"] = pd.to_numeric(
    df_history["Number_fires"], errors="coerce"
)

# --------------------------------------------------
# 🔹 Open-Meteo Setup
# --------------------------------------------------
cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# --------------------------------------------------
# 🔹 Wetter (Single & Multi-Point)
# --------------------------------------------------
def fetch_weather_single(lat, lon, start_date, end_date):
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

    r = openmeteo.weather_api(url, params=params)[0]
    d = r.Daily()

    return pd.DataFrame({
        "temperature_2m_mean": d.Variables(0).ValuesAsNumpy(),
        "relative_humidity_2m_mean": d.Variables(1).ValuesAsNumpy(),
        "et0_fao_evapotranspiration": d.Variables(2).ValuesAsNumpy(),
    })

def fetch_weather_province(coords, start_date, end_date):
    dfs = [
        fetch_weather_single(c["lat"], c["lon"], start_date, end_date)
        for c in coords
    ]
    return pd.concat(dfs).groupby(level=0).mean()

# --------------------------------------------------
# 🔹 Feature Builder (wie app.py)
# --------------------------------------------------
def build_model_input(target_date, daily_df, df_history, jurisdiction):
    month = target_date.month

    hist = (
        df_history[df_history["Jurisdiction"] == jurisdiction]
        .sort_values(["Year", "Month"])
        .tail(3)
    )

    nf = hist["Number_fires"].values if len(hist) >= 3 else [0, 0, 0]

    data = {
        "Year": target_date.year,
        "Month": month,
        "Month_sin": np.sin(2 * np.pi * month / 12),
        "Month_cos": np.cos(2 * np.pi * month / 12),

        "temperature_2m_mean (°C)": daily_df["temperature_2m_mean"].mean(),
        "relative_humidity_2m_mean (%)": daily_df["relative_humidity_2m_mean"].mean(),
        "et0_fao_evapotranspiration (mm)": daily_df["et0_fao_evapotranspiration"].mean(),

        "Lag_1": nf[-1],
        "Lag_2": nf[-2],
        "Lag_3": nf[-3],
        "RollMean_3": np.mean(nf),
        "RollSum_3": np.sum(nf),
    }

    # Region OHE
    for col in training_features:
        if col.startswith("REG_"):
            data[col] = 1 if col == f"REG_{jurisdiction}" else 0

    X = pd.DataFrame([data])

    for col in training_features:
        if col not in X.columns:
            X[col] = 0

    return X[training_features]

# --------------------------------------------------
# 🔥 Fire Index (auf Modellprognose!)
# --------------------------------------------------
def fire_index_from_prediction(pred, province, month):
    hist = df_history[
        (df_history["Jurisdiction"] == province) &
        (df_history["Month"] == month)
    ]["Number_fires"].dropna()

    if len(hist) < 10:
        return 10.0

    return round((hist < pred).mean() * 100, 1)

def risk_level(idx):
    if idx < 25:
        return "low"
    elif idx < 50:
        return "moderate"
    elif idx < 75:
        return "high"
    else:
        return "extreme"

# --------------------------------------------------
# 🚀 API Endpoint
# --------------------------------------------------
@app.get("/risk")
def risk(province: str = "British Columbia", days: int = 10):
    if province not in PROVINCES:
        raise HTTPException(404, "Province not found")

    cfg = PROVINCES[province]

    daily_df = fetch_weather_province(
        cfg["coords"],
        date.today().isoformat(),
        (date.today() + timedelta(days=days)).isoformat()
    )

    X = build_model_input(
        date.today(),
        daily_df,
        df_history,
        cfg["jurisdiction"]
    )

    monthly_pred = float(model.predict(X)[0])
    fire_index = fire_index_from_prediction(
        monthly_pred,
        cfg["jurisdiction"],
        date.today().month
    )

    return {
        "province": province,
        "days": days,
        "monthly_prediction": round(monthly_pred, 1),
        "fire_index": fire_index,
        "risk_level": risk_level(fire_index)
    }