import streamlit as st
import requests
import pandas as pd

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

st.sidebar.header("Provinzen")

province = st.sidebar.selectbox(
    "Provinz auswählen",
    options=PROVINCES,
    index=0  # Default: British Columbia
)

# -------------------------------
# Seitentitel
# -------------------------------
st.set_page_config(page_title="🔥 FireWatch Dashboard", layout="centered")
st.title("🔥 Waldbrand-Prognose Kanada")
st.markdown(
    f"### 📍 Aktuelle Provinz: **{province}**"
)


# -------------------------------
# User Input
# -------------------------------
days = st.slider(
    "Prognose-Zeitraum (Tage)",
    min_value=5,
    max_value=15,
    value=10,
    step=1
)

st.caption("ℹ️ Die Prognose ist aufgrund der Wetter-API auf maximal 15 Tage begrenzt.")
# -------------------------------
# API Call
# -------------------------------
if st.button("Prognose berechnen"):
    try:
        with st.spinner("Berechne Prognose..."):
            response = requests.get(
                "http://127.0.0.1:8000/predict",
                params={
                    "province": province,
                    "days": days
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

        # -------------------------------
        # KPI Anzeige
        # -------------------------------
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

        # -------------------------------
        # Diagramm
        # -------------------------------
        df_plot = pd.DataFrame({
            "Zeitraum": ["10 Tage", "Monat"],
            "Erwartete Brände": [
                data["days_prediction"],
                data["monthly_prediction"]
            ]
        })

        st.subheader("📈 Vergleich")
        st.bar_chart(df_plot.set_index("Zeitraum"))

        # -------------------------------
        # Karte (optional)
        # -------------------------------
        st.subheader("🗺️ Verwendete Wetter-Hotspots in British Columbia")

        map_df = pd.DataFrame({
            "Region": ["Cariboo", "Okanagan", "Peace River"],
            "lat": [52.15, 49.80, 56.30],
            "lon": [-122.15, -119.60, -121.00]
        })

        st.map(map_df)

    except Exception as e:
        st.error(f"Fehler beim Abrufen der Prognose: {e}")