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
# 🔹 Sidebar – Einstellungen
# =====================================================

st.sidebar.title("⚙️ Einstellungen")

province = st.sidebar.selectbox(
    "Provinz auswählen",
    list(PROVINCES.keys())
)

st.sidebar.markdown("### 📅 Prognose-Monat")

month_names = {
    1: "Januar", 2: "Februar", 3: "März", 4: "April",
    5: "Mai", 6: "Juni", 7: "Juli", 8: "August",
    9: "September", 10: "Oktober", 11: "November", 12: "Dezember"
}

target_month = st.sidebar.selectbox(
    "Monat auswählen",
    options=list(month_names.keys()),
    format_func=lambda m: month_names[m]
)

month_label = month_names[target_month]

# =====================================================
# 🔹 API-Aufruf
# =====================================================

with st.spinner("🔮 Monatsprognose wird berechnet …"):
    try:
        response = requests.get(
            API_URL,
            params={
                "province": province,
                "month": target_month
            },
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        st.error(f"❌ Fehler beim Abrufen der Prognose: {e}")
        st.stop()

monthly_prediction = float(data["monthly_prediction"])

# =====================================================
# 🔹 Header
# =====================================================

st.title("🔥 Firewatch – Prognose der monatlichen Brandaktivität")

st.caption(
    f"Prognose für **{province}**, Monat **{month_label}** (saisonale Monatsprognose)"
)

# =====================================================
# 🔹 KPI Anzeige
# =====================================================

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="🔥 Erwartete Brände im Monat",
        value=max(0, round(monthly_prediction, 1))
    )

with col2:
    st.metric(
        label="📅 Prognosezeitraum",
        value=f"{month_label}"
    )

# =====================================================
# 🔹 Historischer Vergleich – gleicher Monat
# =====================================================

st.subheader("📊 Historischer Vergleich – gleicher Kalendermonat")

df_hist_prov = df_history[
    (df_history["Jurisdiction"] == province) &
    (df_history["Month"] == target_month)
]

if df_hist_prov.empty:
    hist_status = "no_data"
else:
    values = df_hist_prov["Number_fires"].dropna().values
    if len(values) >= 5:
        median_hist = float(np.median(values))
        p75 = float(np.percentile(values, 75))
        p90 = float(np.percentile(values, 90))
        hist_status = "ok"
    else:
        hist_status = "no_data"

col1, col2, col3 = st.columns(3)

if hist_status == "no_data":
    col1.metric("📊 Historischer Median", "–")
    col2.metric("📈 75. Perzentil", "–")
    col3.metric("🔺 90. Perzentil", "–")
    st.info(
        "ℹ️ Für diese Provinz liegen für diesen Monat "
        "keine ausreichenden historischen Vergleichsdaten vor."
    )
else:
    col1.metric("📊 Historischer Median", round(median_hist, 1))
    col2.metric("📈 75. Perzentil (oberer Normalbereich)", round(p75, 1))
    col3.metric("🔺 90. Perzentil (Extremjahre)", round(p90, 1))

# =====================================================
# 🔹 Visualisierung
# =====================================================

if hist_status == "ok":
    st.line_chart(
        df_hist_prov.set_index("Year")["Number_fires"],
        height=300
    )
    st.caption(
        f"Historische Brandanzahl im Monat **{month_label}** (alle verfügbaren Jahre)"
    )
else:
    st.caption("Keine historischen Monatsdaten verfügbar.")

# =====================================================
# 🔹 Einordnung
# =====================================================

st.subheader("🧭 Einordnung der Prognose")

if hist_status == "no_data":
    level = "⚪ Keine historische Einordnung möglich"
elif monthly_prediction < median_hist:
    level = "🟢 Unterdurchschnittliche Brandaktivität"
elif monthly_prediction < p75:
    level = "🟡 Typische bis leicht erhöhte Brandaktivität"
elif monthly_prediction < p90:
    level = "🟠 Hohe Brandaktivität"
else:
    level = "🔴 Sehr hohe / extreme Brandaktivität"

st.markdown(
    f"""
**Prognostizierte Brandaktivität:** {round(monthly_prediction, 1)} Brände  
**Monat:** {month_label}  

➡️ **Einordnung:** {level}
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
    "ℹ️ Die Prognose stellt eine **saisonale Monatsabschätzung** dar. Das Jahr ist nicht Bestandteil der Prognose, da das Modell auf monatliche Saisonalität, regionale Unterschiede und aktuelle Wetterbedingungen trainiert wurde."
)