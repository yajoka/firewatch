import pandas as pd
import matplotlib.pyplot as plt
# Pfad zur Datei (anpassen falls nötig)
file_path = "../data/Number_of_fires_by_month.csv"

# CSV einlesen

df = pd.read_csv(file_path)

print("✅ Datei geladen!")
print(df.head())
