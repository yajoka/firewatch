from gettext import install
from fastapi import FastAPI
import numpy as np

app = FastAPI()

@app.get("/")
def root():
    return {"status": "läuft"}


pip install openmeteo-requests
pip install requests-cache retry-requests np pd

Usage
import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": 52.15,
	"longitude": -122.15,
	"daily": ["temperature_2m_mean", "relative_humidity_2m_mean", "et0_fao_evapotranspiration"],
	"start_date": "2026-01-01",
	"end_date": "2026-01-21",
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process daily data. The order of variables needs to be the same as requested.
daily = response.Daily()
daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
daily_relative_humidity_2m_mean = daily.Variables(1).ValuesAsNumpy()
daily_et0_fao_evapotranspiration = daily.Variables(2).ValuesAsNumpy()

daily_data = {"date": pd.date_range(
	start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
	end =  pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = daily.Interval()),
	inclusive = "left"
)}

daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
daily_data["relative_humidity_2m_mean"] = daily_relative_humidity_2m_mean
daily_data["et0_fao_evapotranspiration"] = daily_et0_fao_evapotranspiration

daily_dataframe = pd.DataFrame(data = daily_data)
print("\nDaily data\n", daily_dataframe)

def build_time_features(target_date):
    year = target_date.year
    month = target_date.month

    return {
        "Year": year,
        "Month": month,
        "Month_sin": np.sin(2 * np.pi * month / 12),
        "Month_cos": np.cos(2 * np.pi * month / 12),
    }