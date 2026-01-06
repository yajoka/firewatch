import streamlit as st
import requests
import pandas as pd
from datetime import date

# --------------------------------------------------
# Konfiguration
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000/predict"
HISTORY_CSV = "../backend/fires_history_all_clean.csv"

PROVINCES = [
    "British Columbia",
    "Alberta",
    "Saskatchewan",
    "Manitoba",
    "Ontario",
    "Quebec",
    "New Brunswick",
    "Nova Scotia",
    "Prince Edward Island",
    "Newfoundland and Labrador",
    "Yukon",
    "Northwest Territories",
]

PROVINCE_COORDS = {
    "British Columbia": [
        {"Region": "Cariboo", "lat": 52.15, "lon": -122.15},
        {"Region": "Okanagan", "lat": 49.80, "lon": -119.60},
        {"Region": "Peace River", "lat": 56.30, "lon": -121.00},
    ],
    "Alberta": [
        {"Region": "Fort McMurray Forest Zone", "lat": 56.75, "lon": -111.45},
        {"Region": "Slave Lake Wildland Area", "lat": 55.28, "lon": -114.77},
        {"Region": "Rocky Mountain House Forest Belt", "lat": 52.37, "lon": -115.10},
    ],
    "Saskatchewan": [
        {"Region": "Prince Albert National Park", "lat": 53.96, "lon": -106.22},
        {"Region": "La Ronge Boreal Forest", "lat": 55.10, "lon": -105.30},
        {"Region": "Meadow Lake Provincial Park", "lat": 54.14, "lon": -108.44},
    ],
    "Manitoba": [
        {"Region": "Thompson Boreal Forest", "lat": 55.74, "lon": -97.86},
        {"Region": "Flin Flon Forest Area", "lat": 54.77, "lon": -101.87},
        {"Region": "Atikaki Provincial Wilderness", "lat": 51.23, "lon": -95.28},
    ],
    "Ontario": [
        {"Region": "Red Lake Boreal Forest", "lat": 51.02, "lon": -93.82},
        {"Region": "Sioux Lookout Forest Region", "lat": 50.10, "lon": -91.90},
        {"Region": "Temagami Forest Area", "lat": 47.06, "lon": -80.06},
    ],
    "Quebec": [
        {"Region": "Chibougamau Boreal Forest", "lat": 49.90, "lon": -74.38},
        {"Region": "La Grande / Nord-du-Québec", "lat": 53.60, "lon": -76.10},
        {"Region": "Mauricie Forest Belt", "lat": 47.00, "lon": -72.80},
    ],
    "New Brunswick": [
        {"Region": "Miramichi Forest Region", "lat": 46.98, "lon": -65.52},
        {"Region": "Mount Carleton Provincial Park", "lat": 47.39, "lon": -66.86},
        {"Region": "Northumberland County Forests", "lat": 46.96, "lon": -65.06},
    ],
    "Nova Scotia": [
        {"Region": "Yarmouth County Forest", "lat": 43.90, "lon": -65.95},
        {"Region": "Guysborough County Forests", "lat": 45.19, "lon": -61.60},
        {"Region": "Kejimkujik National Park", "lat": 44.43, "lon": -65.23},
    ],
    "Prince Edward Island": [
        {"Region": "West Prince Forest Belt", "lat": 46.77, "lon": -64.09},
        {"Region": "Camp Tamawaby Forest Area", "lat": 46.39, "lon": -63.48},
        {"Region": "Belle River Forest", "lat": 46.13, "lon": -62.89},
    ],
    "Newfoundland and Labrador": [
        {"Region": "Churchill Falls Boreal Forest", "lat": 53.55, "lon": -64.03},
        {"Region": "Labrador City Forest Zone", "lat": 52.95, "lon": -66.92},
        {"Region": "Terra Nova National Park", "lat": 48.55, "lon": -53.97},
    ],
    "Yukon": [
        {"Region": "Whitehorse Boreal Forest", "lat": 60.75, "lon": -135.12},
        {"Region": "Watson Lake Region", "lat": 60.05, "lon": -128.70},
        {"Region": "Kluane Forest Edge", "lat": 60.75, "lon": -138.55},
    ],
    "Northwest Territories": [
        {"Region": "Hay River Boreal Forest", "lat": 60.85, "lon": -115.75},
        {"Region": "Fort Smith / Thebacha Region", "lat": 60.00, "lon": -112.02},
        {"Region": "Great Slave Lake North Shore", "lat": 62.50, "lon": -114.50},
    ],
}

# --------------------------------------------------
# Page Setup
# --------------------------------------------------

st.set_page_config(page_title="🔥 FireWatch Dashboard", layout="centered")
st.title("🔥 Waldbrand-Prognose Kanada")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Einstellungen")

province = st.sidebar.selectbox(
    "Provinz auswählen",
    options=PROVINCES
)

days = st.sidebar.slider(
    "Prognose-Zeitraum (Tage)",
    min_value=5,
    max_value=15,
    value=10,
    step=1
)

st.sidebar.caption("ℹ️ Wetterprognosen sind auf maximal 15 Tage begrenzt.")

st.markdown(f"### 📍 Aktuelle Provinz: **{province}**")

# --------------------------------------------------
# Historische Daten laden
# --------------------------------------------------

@st.cache_data
def load_history():
    df = pd.read_csv(HISTORY_CSV)
    return df

df_history = load_history()

current_month = date.today().month

df_hist_sel = df_history[
    (df_history["Jurisdiction"] == province) &
    (df_history["Month"] == current_month)
]

# --------------------------------------------------
# Prognose abrufen
# --------------------------------------------------

if st.button("Prognose berechnen"):
    try:
        with st.spinner("Berechne Prognose..."):
            response = requests.get(
                API_URL,
                params={"province": province, "days": days},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

        # --------------------------------------------------
        # KPI Anzeige
        # --------------------------------------------------

        st.subheader("📊 Prognose")

        col1, col2 = st.columns(2)
        col1.metric(
            label=f"🔥 Erwartete Brände in den nächsten {days} Tagen",
            value=data["days_prediction"]
        )
        col2.metric(
            label="📅 Monatliche Prognose",
            value=data["monthly_prediction"]
        )

        # --------------------------------------------------
        # Historischer Vergleich
        # --------------------------------------------------

        st.subheader("📈 Prognose im historischen Kontext")

        hist_mean = df_hist_sel["Number_fires"].mean()
        hist_median = df_hist_sel["Number_fires"].median()

        df_compare = pd.DataFrame({
            "Kategorie": [
                "Historischer Mittelwert (30 Jahre)",
                "Historischer Median (30 Jahre)",
                "Modellprognose"
            ],
            "Anzahl Brände": [
                hist_mean,
                hist_median,
                data["monthly_prediction"]
            ]
        })

        st.bar_chart(df_compare.set_index("Kategorie"))

        st.caption(
            f"Vergleich für Monat {current_month} basierend auf historischen Daten von 1990–2022."
        )

        # --------------------------------------------------
        # Karte
        # --------------------------------------------------

        st.subheader(f"🗺️ Verwendete Wetter-Hotspots in {province}")

        coords = PROVINCE_COORDS.get(province)

        if coords:
            map_df = pd.DataFrame(coords)
            st.map(map_df)
            st.caption(
                f"Wetterdaten basieren auf {len(coords)} feuerhistorisch relevanten Regionen."
            )
        else:
            st.info("Für diese Provinz sind keine Hotspot-Koordinaten definiert.")

    except Exception as e:
        st.error(f"Fehler beim Abrufen der Prognose: {e}")