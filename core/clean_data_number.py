import pandas as pd

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


# Alle leeren Werte (NaN) durch 0 ersetzen
df = df.fillna(0)

# "Wide" → "Long" Format
df_long = df.melt(
    id_vars=["Jurisdiction", "Month"],  # diese Spalten bleiben fix
    var_name="Year",                    # Name der neuen "Jahr"-Spalte
    value_name="Number_fires"                  # Name für die Werte (z. B. Anzahl Brände)
)

# Datentypen anpassen
df_long["Year"] = df_long["Year"].astype(int)



# Jurisdiction als Kategorie, Month als 1–12 umwandeln
df_long["Jurisdiction"] = df_long["Jurisdiction"].astype("category")
df_long["Month"] = pd.to_datetime(df_long["Month"], format="%B").dt.month

#Datei speichern
output_path = "../data/cleaned_Number_of_fires_by_month.csv"
df_long.to_csv(output_path, index=False)

# Kontrolle
print(df_long.head(20))
print(df_long.dtypes)