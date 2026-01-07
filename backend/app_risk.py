from fastapi import FastAPI
from datetime import date, timedelta
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
import openmeteo_requests
from scipy.stats import percentileofscore

from provinces import PROVINCES

app = FastAPI()

# --------------------------------------------------
# Wetter-API Setup
# --------------------------------------------------
cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# --------------------------------------------------
# Historische Wetterdaten (vereinfachte Proxy-Nutzung)
# --------------------------------------------------
# 👉 Wir nutzen historische Branddaten NUR zur Saison-Referenz,
# 👉 der Index selbst ist wetterbasiert
HISTORY_CSV = "fires_history_all_clean.csv"
df_history = pd.read_csv(HISTORY_CSV)

df_history["Year"] = df_history["Year"].astype(int)
df_history["Month"] = df_history["Month"].astype(int)
df_history["Number_fires"] = pd.to_numeric(
    df_history["Number_fires"], errors="coerce"
).fillna(0)

# --------------------------------------------------
# Wetterfunktionen
# --------------------------------------------------
def fetch_weather_single(lat, lon, start_date, end_date):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "et0_fao_evapotranspiration",
        ],
        "start_date": start_date,
        "end_date": end_date,
    }

    response = openmeteo.weather_api(url, params=params)[0]
    daily = response.Daily()

    return pd.DataFrame({
        "temp": daily.Variables(0).ValuesAsNumpy(),
        "humidity": daily.Variables(1).ValuesAsNumpy(),
        "et0": daily.Variables(2).ValuesAsNumpy(),
    })


def fetch_weather_province(coords, start_date, end_date):
    dfs = [
        fetch_weather_single(c["lat"], c["lon"], start_date, end_date)
        for c in coords
    ]
    df_all = pd.concat(dfs)
    return df_all.groupby(df_all.index).mean()


# --------------------------------------------------
# 🔥 Fire Index (WETTERBASIERT)
# --------------------------------------------------
def compute_weather_risk_score(df_weather: pd.DataFrame) -> float:
    """
    Physikalisch sinnvoller Wetter-Risiko-Score (0–1)
    """
    temp = df_weather["temp"].mean()
    et0 = df_weather["et0"].mean()
    hum = df_weather["humidity"].mean()

    # Normalisierung (robust & simpel)
    temp_n = np.clip((temp + 20) / 40, 0, 1)
    et0_n = np.clip(et0 / 5, 0, 1)
    hum_n = 1 - np.clip(hum / 100, 0, 1)

    return 0.5 * temp_n + 0.3 * et0_n + 0.2 * hum_n


def compute_seasonal_percentile(province: str, month: int, score: float) -> float:
    """
    Vergleich gegen historische MONATE (Proxy über Brandaktivität)
    """
    hist = df_history[
        (df_history["Jurisdiction"] == province) &
        (df_history["Month"] == month)
    ]["Number_fires"]

    if len(hist) < 5:
        return 50.0  # neutral bei wenig Daten

    return float(
        percentileofscore(hist, hist.mean(), kind="mean")
    )


def risk_label(p):
    if p < 25:
        return "low"
    elif p < 50:
        return "moderate"
    elif p < 75:
        return "high"
    return "extreme"


# --------------------------------------------------
# API Endpoint
# --------------------------------------------------
@app.get("/risk")
def risk(province: str = "British Columbia", days: int = 10):

    cfg = PROVINCES[province]

    df_weather = fetch_weather_province(
        coords=cfg["coords"],
        start_date=date.today().isoformat(),
        end_date=(date.today() + timedelta(days=days)).isoformat(),
    )

    weather_score = compute_weather_risk_score(df_weather)

    percentile = compute_seasonal_percentile(
        province=province,
        month=date.today().month,
        score=weather_score,
    )

    fire_index = round(weather_score * 100, 1)

    return {
        "province": province,
        "days": days,
        "fire_index": fire_index,
        "percentile": round(percentile, 1),
        "risk_level": risk_label(percentile),
    }