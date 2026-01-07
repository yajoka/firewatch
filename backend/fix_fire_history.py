import pandas as pd

# Pfade anpassen
INPUT_CSV = "fires_history_all_clean.csv"
OUTPUT_CSV = "fires_history_all_clean_fixed.csv"

# Laden
df = pd.read_csv(INPUT_CSV)

# Sicherheit: Month als int
df["Month"] = df["Month"].astype(int)
df["Year"] = df["Year"].astype(int)

# Alle Kombinationen erzeugen
all_years = df["Year"].unique()
all_months = range(1, 13)
all_provinces = df["Jurisdiction"].unique()

full_index = pd.MultiIndex.from_product(
    [all_years, all_months, all_provinces],
    names=["Year", "Month", "Jurisdiction"]
)

# Reindex → fehlende Monate erscheinen als NaN
df_full = (
    df
    .set_index(["Year", "Month", "Jurisdiction"])
    .reindex(full_index)
    .reset_index()
)

# Fehlende Werte = 0 Brände
df_full["Number_fires"] = df_full["Number_fires"].fillna(0).astype(int)

# Speichern
df_full.to_csv(OUTPUT_CSV, index=False)

print("✅ Fertig!")
print("➡️ Alte Zeilen:", len(df))
print("➡️ Neue Zeilen:", len(df_full))