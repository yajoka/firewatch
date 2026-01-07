import numpy as np
import pandas as pd
from datetime import date
import joblib

import openmeteo_requests
import requests_cache
from retry_requests import retry

FINAL_DATA_PATH = "../data/final_data.csv"
MODEL_PATH = "../models/xgboost_final_model.joblib"
FEATS_PATH = "../models/xgboost_features.joblib"
IMPUTER_PATH = "../backend/imputer.joblib"

# ✅ Zielmonat festlegen
TARGET_YEAR = 2026
TARGET_MONTH = 1

cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def fetch_weather_last14d(lat, lon):
    end_d = date.today()
    start_d = (pd.Timestamp(end_d) - pd.Timedelta(days=14)).date()

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_mean",
            "temperature_2m_max",
            "relative_humidity_2m_mean",
            "relative_humidity_2m_min",
            "et0_fao_evapotranspiration",
        ],
        "start_date": start_d.isoformat(),
        "end_date": end_d.isoformat(),
        "timezone": "UTC",
    }

    r = openmeteo.weather_api(url, params=params)[0]
    d = r.Daily()

    df = pd.DataFrame({
        "temperature_2m_mean": d.Variables(0).ValuesAsNumpy(),
        "temperature_2m_max": d.Variables(1).ValuesAsNumpy(),
        "relative_humidity_2m_mean": d.Variables(2).ValuesAsNumpy(),
        "relative_humidity_2m_min": d.Variables(3).ValuesAsNumpy(),
        "et0_fao_evapotranspiration": d.Variables(4).ValuesAsNumpy(),
    })

    return {
        "temperature_2m_mean (°C)": float(df["temperature_2m_mean"].mean()),
        "temperature_2m_max (°C)": float(df["temperature_2m_max"].mean()),
        "relative_humidity_2m_mean (%)": float(df["relative_humidity_2m_mean"].mean()),
        "relative_humidity_2m_min (%)": float(df["relative_humidity_2m_min"].mean()),
        "et0_fao_evapotranspiration (mm)": float(df["et0_fao_evapotranspiration"].mean()),
    }

def build_fire_lag_rolling(series):
    s = series.dropna().astype(float)
    lag1 = float(s.iloc[-1]) if len(s) >= 1 else 0.0
    lag2 = float(s.iloc[-2]) if len(s) >= 2 else 0.0
    lag3 = float(s.iloc[-3]) if len(s) >= 3 else 0.0

    def rmean(k): return float(s.tail(k).mean()) if len(s) >= k else float(s.mean())
    def rstd(k):  return float(s.tail(k).std(ddof=1)) if len(s) >= k else float(s.std(ddof=1))
    def rsum(k):  return float(s.tail(k).sum()) if len(s) >= k else float(s.sum())

    return {
        "Lag_1": lag1, "Lag_2": lag2, "Lag_3": lag3,
        "RollMean_3": rmean(3), "RollMean_6": rmean(6), "RollMean_12": rmean(12),
        "RollStd_3": rstd(3), "RollStd_6": rstd(6), "RollStd_12": rstd(12),
        "RollSum_3": rsum(3), "RollSum_6": rsum(6), "RollSum_12": rsum(12),
    }

def time_feats(year, month):
    return {
        "Year": year,
        "Month": month,
        "Month_sin": np.sin(2*np.pi*month/12),
        "Month_cos": np.cos(2*np.pi*month/12),
    }

def region_onehot(jurisdiction_raw, training_features):
    out = {c: 0 for c in training_features if c.startswith("REG_")}
    key = f"REG_{jurisdiction_raw}"
    if key in out:
        out[key] = 1
    return out

def weather_lag1_from_last_row(last_row, training_features):
    out = {}
    for base in [
        "temperature_2m_mean (°C)",
        "relative_humidity_2m_mean (%)",
        "et0_fao_evapotranspiration (mm)",
    ]:
        col = f"{base}_lag1"
        if col in training_features:
            out[col] = float(last_row.get(base, np.nan))
    return out

def main():
    df = pd.read_csv(FINAL_DATA_PATH)
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)

    model = joblib.load(MODEL_PATH)
    feats = joblib.load(FEATS_PATH)
    imputer = joblib.load(IMPUTER_PATH)

    province = "British Columbia"
    jurisdiction_raw = province

    # Demo-Koordinate (besser: PROVINCES mapping nutzen)
    lat, lon = 53.7267, -127.6476

    dfp = df[df["Jurisdiction_raw"] == jurisdiction_raw].sort_values(["Year", "Month"])
    if dfp.empty:
        raise ValueError("Keine Daten für diese Province in final_data.csv gefunden.")

    # ✅ History nur bis zum Vormonat des Zielmonats verwenden
    cutoff = (TARGET_YEAR * 100 + TARGET_MONTH)
    dfp_hist = dfp[(dfp["Year"] * 100 + dfp["Month"]) < cutoff].copy()
    if dfp_hist["Number_fires"].dropna().empty:
        raise ValueError("Keine historischen Brandwerte vor dem Zielmonat vorhanden.")

    last_row = dfp_hist.iloc[-1]

    data = {}
    data.update(time_feats(TARGET_YEAR, TARGET_MONTH))
    data.update(region_onehot(jurisdiction_raw, feats))
    data.update(build_fire_lag_rolling(dfp_hist["Number_fires"]))
    data.update(fetch_weather_last14d(lat, lon))
    data.update(weather_lag1_from_last_row(last_row, feats))

    X = pd.DataFrame([data])

    for c in feats:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feats]

    num_cols = X.select_dtypes(include=["number"]).columns
    X.loc[:, num_cols] = imputer.transform(X.loc[:, num_cols])

    yhat = float(model.predict(X)[0])

    print("====================================")
    print("Forecast (fixer Zielmonat)")
    print("------------------------------------")
    print("Province:", province)
    print("Target:", f"{TARGET_YEAR}-{TARGET_MONTH:02d}")
    print("Prediction (Number_fires):", round(yhat, 2))
    print("====================================")

if __name__ == "__main__":
    main()
