import streamlit as st
import pandas as pd
import requests
import numpy as np

from provinces import PROVINCES

# =====================================================
# 🔹 Konfiguration
# =====================================================

API_URL = "http://127.0.0.1:8000/predict"
HISTORY_CSV = "../backend/fires_history_all_clean_fixed.csv"

st.set_page_config(
    page_title="🔥 Firewatch – Monatsprognose",
    layout="wide"
)

# =====================================================
# 🔹 Daten laden
# =====================================================

@st.cache_data
def load_history():
    df = pd.read_csv(HISTORY_CSV)
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(int)
    return df

df_history = load_history()

# =====================================================
# 🔹 Sidebar – Auswahl
# =====================================================

st.sidebar.title("⚙️ Einstellungen")

province = st.sidebar.selectbox(
    "Provinz auswählen",
    list(PROVINCES.keys())
)

month_names = {
    1: "Januar", 2: "Februar", 3: "März", 4: "April",
    5: "Mai", 6: "Juni", 7: "Juli", 8: "August",
    9: "September", 10: "Oktober", 11: "November", 12: "Dezember"
}

current_month = pd.Timestamp.today().month
month_label = month_names[current_month]

# =====================================================
# 🔹 API-Aufruf
# =====================================================

with st.spinner("🔮 Prognose wird berechnet …"):
    try:
        response = requests.get(
            API_URL,
            params={"province": province},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        st.error(f"❌ Fehler beim Abrufen der Prognose: {e}")
        st.stop()

monthly_prediction = data["monthly_prediction"]

# =====================================================
# 🔹 Header
# =====================================================

st.title("🔥 Firewatch – Monatliche Brandprognose")

st.caption(
    f"Prognose für **{province}**, Monat **{month_label} {data['year']}**"
)

# =====================================================
# 🔹 KPI Anzeige
# =====================================================

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="🔥 Erwartete Brände (Monat)",
        value=f"{monthly_prediction}"
    )

with col2:
    st.metric(
        label="📅 Prognosezeitraum",
        value=f"{month_label}"
    )

# =====================================================
# 🔹 Historischer Vergleich
# =====================================================

st.subheader("📊 Historischer Vergleich – gleicher Monat")

df_hist_prov = df_history[
    (df_history["Jurisdiction"] == province) &
    (df_history["Month"] == current_month)
]

if df_hist_prov.empty:
    mean_hist = median_hist = p90 = None
    hist_status = "no_data"
else:
    values = df_hist_prov["Number_fires"].values
    mean_hist = float(np.mean(values))
    median_hist = float(np.median(values))
    p90 = float(np.percentile(values, 90)) if np.any(values > 0) else 0.0
    hist_status = "ok"

col1, col2, col3 = st.columns(3)

if hist_status == "no_data":
    col1.metric("📈 Historischer Mittelwert", "–")
    col2.metric("📊 Median", "–")
    col3.metric("🔺 90. Perzentil", "–")
    st.info(
        "ℹ️ Für diese Provinz liegen für diesen Monat keine historischen Branddaten vor."
    )
else:
    col1.metric("📈 Historischer Mittelwert", round(mean_hist, 1))
    col2.metric("📊 Median", round(median_hist, 1))
    col3.metric("🔺 90. Perzentil", round(p90, 1))

# =====================================================
# 🔹 Visualisierung
# =====================================================

if not df_hist_prov.empty:
    st.line_chart(
        df_hist_prov.set_index("Year")["Number_fires"],
        height=300
    )
else:
    st.caption("Keine historischen Daten für diesen Monat verfügbar.")

st.caption(
    "Historische Anzahl an Bränden im gleichen Monat (alle Jahre)"
)

# =====================================================
# 🔹 Einordnung
# =====================================================

if hist_status == "no_data":
    level = "⚪ Keine historische Einordnung möglich"
elif monthly_prediction < mean_hist:
    level = "🟢 Unterdurchschnittlich"
elif monthly_prediction < p90:
    level = "🟠 Überdurchschnittlich"
else:
    level = "🔴 Sehr hoch"

st.subheader("🧭 Einordnung")
if hist_status == "no_data":
    st.markdown(
        f"""
**Prognose:** {monthly_prediction} Brände  

➡️ **Einstufung:** {level}  
ℹ️ Für diesen Monat liegen keine historischen Vergleichsdaten vor.
"""
    )
else:
    st.markdown(
        f"""
**Prognose:** {monthly_prediction} Brände  
**Historischer Mittelwert:** {mean_hist:.1f}  

➡️ **Einstufung:** {level}
"""
    )

# =====================================================
# 🔹 Karte – verwendete Koordinaten
# =====================================================

st.subheader("🗺️ Verwendete Wetter-Standorte")

coords = PROVINCES[province]["coords"]
map_df = pd.DataFrame(coords)

st.map(map_df[["lat", "lon"]])

st.caption(
    "Die Prognose basiert auf aggregierten Wetterdaten dieser Regionen."
)

# =====================================================
# 🔹 Footer
# =====================================================

st.divider()
st.caption(
    "ℹ️ Die Prognose ist eine **Monatsprognose** basierend auf historischen Branddaten "
    "und aggregierten Wetterbedingungen der letzten 14 Tage."
)