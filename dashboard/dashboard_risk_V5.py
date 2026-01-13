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

REFERENCE_YEAR = 2022  # bewusst fixiert

st.set_page_config(
    page_title="🔥 Firewatch – Saisonale Monatsprognose",
    layout="wide"
)

# =====================================================
# 🔹 Historische Daten laden
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

month = st.sidebar.selectbox(
    "Monat auswählen",
    list(month_names.keys()),
    format_func=lambda m: month_names[m]
)

month_label = month_names[month]

st.sidebar.caption(
    "ℹ️ Die Prognose basiert auf historischen saisonalen Mustern.\n"
    f"Das Jahr ist intern auf {REFERENCE_YEAR} fixiert."
)

# =====================================================
# 🔹 API-Aufruf
# =====================================================

with st.spinner("🔮 Saisonale Monatsprognose wird berechnet …"):
    try:
        response = requests.get(
            API_URL,
            params={
                "province": province,
                "year": REFERENCE_YEAR,
                "month": month
            },
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

st.title("🔥 Firewatch – Saisonale Monatsprognose")

st.caption(
    f"**Prognose für {province} – Monat {month_label}**  \n"
    f"(saisonale Referenz auf Basis historischer Daten)"
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
        label="📅 Referenzzeitraum",
        value=month_label
    )

# =====================================================
# 🔹 Historischer Vergleich (gleicher Monat)
# =====================================================

st.subheader("📊 Historischer Vergleich – gleicher Monat")

df_hist_prov = df_history[
    (df_history["Jurisdiction"] == province) &
    (df_history["Month"] == month)
]

if df_hist_prov.empty:
    hist_status = "no_data"
else:
    values = df_hist_prov["Number_fires"].dropna().values
    if len(values) >= 3:
        median_hist = float(np.median(values))
        p75 = float(np.percentile(values, 75))
        p90 = float(np.percentile(values, 90))
        hist_status = "ok"
    else:
        hist_status = "no_data"

col1, col2, col3 = st.columns(3)

if hist_status == "no_data":
    col1.metric("📊 Median", "–")
    col2.metric("🔶 75. Perzentil", "–")
    col3.metric("🔺 90. Perzentil", "–")
    st.info(
        "ℹ️ Für diese Provinz liegen für diesen Monat zu wenige historische Daten vor."
    )
else:
    col1.metric("📊 Median", round(median_hist, 1))
    col2.metric("🔶 75. Perzentil", round(p75, 1))
    col3.metric("🔺 90. Perzentil", round(p90, 1))

# =====================================================
# 🔹 Zeitliche Entwicklung
# =====================================================

if hist_status == "ok":
    st.line_chart(
        df_hist_prov.set_index("Year")["Number_fires"],
        height=300
    )
else:
    st.caption("Keine historische Zeitreihe verfügbar.")

st.caption(
    "Historische Brandanzahl im gleichen Monat über alle Jahre"
)

# =====================================================
# 🔹 Qualitative Einordnung
# =====================================================

st.subheader("🧭 Einordnung")

if hist_status == "no_data":
    level = "⚪ Keine historische Einordnung möglich"
elif monthly_prediction < median_hist:
    level = "🟢 Unter dem saisonalen Median"
elif monthly_prediction < p75:
    level = "🟡 Innerhalb des normalen saisonalen Bereichs"
elif monthly_prediction < p90:
    level = "🟠 Erhöhte Aktivität"
else:
    level = "🔴 Sehr hohe Aktivität (Extrembereich)"

st.markdown(
    f"""
**Prognose:** {monthly_prediction} Brände  

➡️ **Einstufung:** {level}

ℹ️ *Die Bewertung erfolgt relativ zu historischen Werten desselben Monats,
nicht als absolute Gefahrenaussage.*
"""
)

# =====================================================
# 🔹 Karte – verwendete Wetter-Standorte
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
    "ℹ️ Diese Anwendung zeigt eine **saisonale Monatsprognose** auf Basis "
    "historischer Brandmuster und aggregierter Wetterbedingungen. "
    "Sie stellt keine operative Einsatzprognose dar."
)