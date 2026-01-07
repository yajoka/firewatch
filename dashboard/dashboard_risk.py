import streamlit as st
import requests
import pandas as pd
from provinces import PROVINCES

API_URL = "http://127.0.0.1:8000/risk"

st.set_page_config(
    page_title="🔥 Fire Risk Dashboard",
    layout="wide"
)



# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("⚙️ Einstellungen")

province = st.sidebar.selectbox(
    "Provinz auswählen",
    list(PROVINCES.keys())
)

days = st.sidebar.slider(
    "Zeitraum (Tage)",
    1, 16, 10
)

# --------------------------------------------------
# API Call
# --------------------------------------------------
try:
    r = requests.get(
        API_URL,
        params={"province": province, "days": days},
        timeout=30
    )
    r.raise_for_status()
    data = r.json()
except Exception as e:
    st.error(f"Fehler beim Abrufen der Prognose: {e}")
    st.stop()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🔥 Fire Risk Dashboard – Kanada")
st.caption(f"Aktuelle saisonale Risikoanalyse für **{province}**")

# --------------------------------------------------
# KPIs
# --------------------------------------------------
c1, c2, c3 = st.columns(3)

c1.metric(
    "🔥 Fire Index",
    f"{data['fire_index']} / 100"
)

c2.metric(
    "📊 Historischer Rang",
    f"{data['percentile']} %"
)

risk_labels = {
    "low": "🟢 Niedrig",
    "moderate": "🟡 Moderat",
    "high": "🟠 Hoch",
    "extreme": "🔴 Extrem",
}

c3.metric(
    "⚠️ Risiko-Level",
    risk_labels[data["risk_level"]]
)

# --------------------------------------------------
# Erklärung
# --------------------------------------------------
with st.expander("ℹ️ Wie ist der Fire Index zu interpretieren?"):
    st.markdown("""
Der **Fire Index (0–100)** misst,  
wie **ungewöhnlich brandfördernd** die aktuellen Wetterbedingungen sind –  
**relativ zur historischen Saison** derselben Provinz.

- **< 25** → ungewöhnlich ruhig  
- **25–50** → saisonal normal  
- **50–75** → erhöhtes Risiko  
- **> 75** → historisch seltene Bedingungen  

⚠️ **Risiko ≠ Anzahl der Brände**
""")

# --------------------------------------------------
# Karte
# --------------------------------------------------
st.markdown("---")
st.subheader(f"🗺️ Verwendete Wetter-Hotspots in {province}")

coords_df = pd.DataFrame(PROVINCES[province]["coords"])
st.map(coords_df.rename(columns={"lat": "latitude", "lon": "longitude"}))

st.caption(
    "Die Analyse basiert auf dem Mittelwert mehrerer feuerrelevanter Regionen pro Provinz."
)