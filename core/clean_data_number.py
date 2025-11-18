import pandas as pd
import numpy as np

file_path = "../data/Number_of_fires_by_month.csv"

# CSV mit Semikolon-Trenner einlesen
df = pd.read_csv(file_path, sep=";", header=1, encoding="ISO-8859-1")

# Spaltennamen aufräumen
df.columns = df.columns.str.strip()

# "Data Qualifier"-Spalte entfernen (falls vorhanden)
df = df.loc[:, ~df.columns.str.contains('Data Qualifier', case=False)]


# Alle 'Unspecified'-Einträge in 'Month' löschen
df = df[df["Month"].notna()]
df = df[~df['Month'].str.contains("Unspecified", case=False, na=False)]


# Alle leeren Werte durch NaN ersetzen
df = df.replace(["", " ",], np.nan)


# "Wide" → "Long" Format
df_long = df.melt(
    id_vars=["Jurisdiction", "Month"],  # diese Spalten bleiben fix
    var_name="Year",                    # Name der neuen "Jahr"-Spalte
    value_name="Number_fires"                  # Name für die Werte (z. B. Anzahl Brände)
)

# Datentypen anpassen
df_long["Year"] = df_long["Year"].astype(int)

# Anzahl (Number_fires) als ganze Zahl
df_long["Number_fires"] = pd.to_numeric(df_long["Number_fires"], errors="coerce").astype("Int64")

# Jurisdiction als Kategorie, Month als 1–12 umwandeln
df_long["Jurisdiction"] = df_long["Jurisdiction"].astype("category")
df_long["Month"] = pd.to_datetime(df_long["Month"], format="%B").dt.month

# Funktion zur Zuweisung der Jahreszeit (Season)
def get_season(month):
    # Definitionen: Frühling (März-Mai), Sommer (Juni-August),
    # Herbst (September-November) und Winter (Dezember-Februar)
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    elif month in [12, 1, 2]:
        return "Winter"
    return np.nan

# Anwenden der Funktion auf die numerische Monatsspalte
df_long["Season"] = df_long["Month"].apply(get_season)

# Lag hinufügen, vorherige Werte merken
df_long = df_long.sort_values(["Jurisdiction", "Year", "Month"])

df_long["Lag_1"] = df_long.groupby("Jurisdiction")["Number_fires"].shift(1)
df_long["Lag_2"] = df_long.groupby("Jurisdiction")["Number_fires"].shift(2)
df_long["Lag_3"] = df_long.groupby("Jurisdiction")["Number_fires"].shift(3)


#Datei speichern
output_path = "../data/cleaned_Number_of_fires_by_month.csv"
df_long.to_csv(output_path, index=False)

# Kontrolle
print(df_long.head(20))
print(df_long.dtypes)