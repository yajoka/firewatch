import streamlit as st
import requests
from provinces import PROVINCES

API_URL = "http://127.0.0.1:8000/risk"

st.set_page_config(page_title="🔥 Fire Risk Dashboard", layout="centered")

st.title("🔥 Wildfire Risk Dashboard")

province = st.selectbox(
    "Provinz auswählen",
    list(PROVINCES.keys()),
    index=0
)

days = st.slider(
    "Prognosezeitraum (Tage)",
    min_value=1,
    max_value=16,
    value=10
)

if st.button("🔥 Risiko berechnen"):
    try:
        r = requests.get(
            API_URL,
            params={"province": province, "days": days},
            timeout=30
        )
        r.raise_for_status()
        data = r.json()

        st.subheader(f"📍 {province}")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "🔥 Monatsprognose",
            f"{data['monthly_prediction']} Brände"
        )

        col2.metric(
            "🔥 Fire Index",
            f"{data['fire_index']} / 100"
        )

        col3.metric(
            "⚠️ Risiko-Level",
            data["risk_level"].upper()
        )

        st.caption(
            "Fire Index = Vergleich der Modellprognose "
            "mit historischen Werten desselben Monats (30 Jahre)."
        )

    except Exception as e:
        st.error(f"Fehler beim Abrufen der Prognose: {e}")