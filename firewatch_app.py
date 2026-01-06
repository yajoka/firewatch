"""
Streamlit Dashboard für das Firewatch-Projekt
=============================================

Zweiseitige Streamlit-App basierend auf den Repo-Dateien:
- data/final_data.csv
- models/xgboost_final_model.joblib
- models/xgboost_features.joblib
- orga/coordinates/canada_fire_coordinates.csv

Wichtig (technisch):
- Starten mit:  py -m streamlit run firewatch_app.py
- Caching: Model-Objekte (XGBoost) dürfen NICHT über st.cache_data gehasht werden.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


# =============================
# Robust: Projekt-Root finden
# =============================
def find_project_root(start: Path | None = None) -> Path:
    """
    Findet den Ordner, der die erwartete Struktur enthält:
      data/final_data.csv
      models/xgboost_final_model.joblib
      models/xgboost_features.joblib
    Funktioniert auch, wenn das Repo in einem Unterordner liegt.
    """
    start = start or Path(__file__).resolve().parent
    candidates = [start] + list(start.parents)

    def ok(base: Path) -> bool:
        return (
            (base / "data" / "final_data.csv").exists()
            and (base / "models" / "xgboost_final_model.joblib").exists()
            and (base / "models" / "xgboost_features.joblib").exists()
        )

    # 1) direkt in current/parents
    for base in candidates:
        if ok(base):
            return base

    # 2) 1 Ebene tiefer (typisch: Repo liegt als Unterordner)
    for base in candidates:
        try:
            for child in base.iterdir():
                if child.is_dir() and ok(child):
                    return child
        except PermissionError:
            continue

    raise FileNotFoundError(
        "Projekt-Root nicht gefunden. Erwartet:\n"
        "- data/final_data.csv\n"
        "- models/xgboost_final_model.joblib\n"
        "- models/xgboost_features.joblib\n"
        "Lege firewatch_app.py in (oder über) den Repo-Ordner mit data/models/orga."
    )


ROOT = find_project_root()

DATA_PATH = ROOT / "data" / "final_data.csv"
MODEL_PATH = ROOT / "models" / "xgboost_final_model.joblib"
FEATURES_PATH = ROOT / "models" / "xgboost_features.joblib"
COORD_PATH = ROOT / "orga" / "coordinates" / "canada_fire_coordinates.csv"


# =============================
# Laden (caching korrekt)
# =============================
@st.cache_data
def load_dataframe() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model_and_features():
    """
    Model als resource cachen (nicht cache_data).
    """
    features = joblib.load(FEATURES_PATH)
    model = joblib.load(MODEL_PATH)
    return model, features


@st.cache_data
def load_coordinates() -> pd.DataFrame:
    """
    Koordinaten optional: App läuft auch ohne Datei.
    """
    if not COORD_PATH.exists():
        return pd.DataFrame(columns=["Province", "Region", "Latitude", "Longitude"])
    return pd.read_csv(COORD_PATH)


# =============================
# Prep & Metrics
# =============================
@st.cache_data
def prepare_data(df: pd.DataFrame, features: list[str]):
    X = df[features].copy()
    y = df["Number_fires"].copy()

    # Median-Imputation nur für numerische Spalten
    X = X.fillna(X.median(numeric_only=True))

    # Binäre Sicht: mind. ein Feuer
    y_binary = (y > 0).astype(int)

    X_train, X_test, y_train, y_test, y_train_bin, y_test_bin = train_test_split(
        X, y, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_bin": y_train_bin,
        "y_test_bin": y_test_bin,
    }


@st.cache_data
def compute_metrics(_model, split: dict):
    """
    Wichtig: Parametername _model -> Streamlit versucht NICHT, das Modell zu hashen.
    """
    X_test = split["X_test"]
    y_test = split["y_test"]
    y_test_bin = split["y_test_bin"]

    y_pred = _model.predict(X_test)
    y_pred_bin = (y_pred > 0).astype(int)

    cm = confusion_matrix(y_test_bin, y_pred_bin, labels=[0, 1])
    f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)
    acc = accuracy_score(y_test_bin, y_pred_bin)

    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "confusion_matrix": cm,
        "f1": f1,
        "accuracy": acc,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "y_pred": y_pred,
        "y_pred_bin": y_pred_bin,
    }


@st.cache_data
def compute_feature_importance(_model, features: list[str], top_n: int = 10) -> pd.DataFrame:
    """
    Auch hier _model, um Hash-Probleme zu vermeiden.
    """
    importances = getattr(_model, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame({"feature": [], "importance": []})

    df_imp = pd.DataFrame({"feature": features, "importance": importances})
    return df_imp.sort_values("importance", ascending=False).head(top_n)


# =============================
# Plot-Helper (matplotlib)
# =============================
def plot_confusion_matrix(cm: np.ndarray):
    fig, ax = plt.subplots()
    ax.imshow(cm)

    ax.set_xlabel("Vorhergesagt")
    ax.set_ylabel("Tatsächlich")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Kein Feuer", "Feuer"])
    ax.set_yticklabels(["Kein Feuer", "Feuer"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    return fig


def plot_feature_importance(df_imp: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.barh(df_imp["feature"], df_imp["importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Top Features")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_error_bar(y_true_bin: np.ndarray, y_pred_bin: np.ndarray):
    fp = np.logical_and(y_true_bin == 0, y_pred_bin == 1).sum()
    fn = np.logical_and(y_true_bin == 1, y_pred_bin == 0).sum()

    fig, ax = plt.subplots()
    ax.bar(["False Positives", "False Negatives"], [fp, fn])
    ax.set_ylabel("Anzahl")
    ax.set_title("Fehlklassifikationen")
    fig.tight_layout()
    return fig


# =============================
# UI
# =============================
def main():
    st.set_page_config(page_title="Firewatch Dashboard", layout="wide")

    df = load_dataframe()
    model, features = load_model_and_features()
    coords = load_coordinates()

    split = prepare_data(df, features)
    metrics = compute_metrics(model, split)
    feature_imp = compute_feature_importance(model, features, top_n=10)

    page = st.sidebar.selectbox(
        "Wähle eine Seite:",
        ["Datenbasis & Modellverhalten", "Räumliche Muster, Fehler & Grenzen"],
    )

    if page == "Datenbasis & Modellverhalten":
        st.title("Analyse historischer Waldbranddaten mittels XGBoost")
        st.caption("Klassifikations-/Risiko-Sicht aus einem Regressionsmodell (Anzahl Brände) – Kanada")

        # Datenbasis
        st.markdown("### Verwendete Daten")
        col1, col2 = st.columns(2)

        with col1:
            min_year = int(df["Year"].min()) if "Year" in df.columns else np.nan
            max_year = int(df["Year"].max()) if "Year" in df.columns else np.nan
            st.markdown(f"**Zeitraum der Daten:** {min_year} – {max_year}")
            st.markdown(f"**Anzahl Datensätze:** {len(df):,}")
            st.markdown("**Zielvariable:** `Number_fires` (Anzahl Brände)")

        with col2:
            weather_features = [
                f for f in features
                if any(kw in f for kw in ["temperature", "humidity", "precipitation", "wind", "et0"])
            ]
            time_features = [f for f in features if f in ["Month", "Year", "Lag_1", "Lag_2", "Lag_3"]]
            region_features = [f for f in features if f.startswith("REG_")]

            st.markdown("**Feature-Gruppen:**")
            st.markdown(f"- Zeit/Lag: {', '.join(time_features)}")
            st.markdown(
                f"- Wetter (Beispiele): {', '.join(weather_features[:5])}"
                + (" ..." if len(weather_features) > 5 else "")
            )
            st.markdown(f"- Region: {len(region_features)} Provinz-Dummies")
            st.markdown("*Keine Echtzeit-, Satelliten- oder extern validierten Daten.*")

        # Modellübersicht
        st.markdown("### Modelltyp & Training")
        st.markdown(
            "Input Features → **XGBoost Regressor** → Output: `Number_fires`.\n\n"
            "**Train/Test Split:** 80/20, stratifiziert nach `Number_fires > 0`."
        )

        # Performance
        st.markdown("### Modellergebnis")
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Regression**")
            st.metric("RMSE", f"{metrics['rmse']:.2f}")
            st.metric("MAE", f"{metrics['mae']:.2f}")
            st.metric("R²", f"{metrics['r2']:.3f}")
        with col4:
            st.markdown("**Binäre Sicht (vereinfachte Ableitung)**")
            st.metric("F1", f"{metrics['f1']:.3f}")
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")

        st.caption("Performance bezieht sich ausschließlich auf den verwendeten Datensatz.")
        st.pyplot(plot_confusion_matrix(metrics["confusion_matrix"]))

        # Feature importance
        st.markdown("### Einfluss der Eingangsvariablen")
        if feature_imp.empty:
            st.warning("Feature Importance nicht verfügbar (Modell liefert keine feature_importances_).")
        else:
            st.pyplot(plot_feature_importance(feature_imp))
            st.caption("Feature Importance beschreibt Modellverhalten, keine Kausalität.")

    else:
        st.title("Räumliche Muster & Fehleranalyse")

        st.markdown("### Historische Brandregionen")
        if coords.empty:
            st.warning("Keine Koordinaten-Datei gefunden: orga/coordinates/canada_fire_coordinates.csv")
        else:
            st.map(coords.rename(columns={"Latitude": "lat", "Longitude": "lon"}))
            st.caption("Darstellung verfügbarer Regionen – keine Vorhersage zukünftiger Ereignisse.")

        st.markdown("### Fehlklassifikationen")
        st.pyplot(plot_error_bar(split["y_test_bin"].values, metrics["y_pred_bin"]))

        st.markdown("### Limitationen des Modells")
        st.markdown(
            "- Nur historische Daten\n"
            "- Keine echte out-of-time Validierung (kein Zeitreihen-Splitting)\n"
            "- Keine externe Validierung\n"
            "- Begrenzte Generalisierbarkeit (Wetter + Provinz + abgeleitete Lag/Rolling-Features)"
        )


if __name__ == "__main__":
    main()
