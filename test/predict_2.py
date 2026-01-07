import joblib
import pandas as pd

#Modell, Features und Imputer laden
model = joblib.load("../models/xgboost_final_model_v3.joblib")
features = joblib.load("../models/xgboost_features_v3.joblib")
imputer = joblib.load("../backend/imputer.joblib")

#Daten laden
df = pd.read_csv("../data/final_data.csv")

#Verfügbare Regionen anzeigen
print("Verfügbare Regionen:")
regions = df['Jurisdiction_raw'].unique()
for i, region in enumerate(regions, 1):
    print(f"{i}. {region}")

#Benutzereingaben
print("\n" + "="*50)
month = int(input("Monat (1-12): "))
year = int(input("Jahr: "))
region = input("Region (Name eingeben): ")

#Daten für die Eingabe filtern
data = df[
    (df['Month'] == month) &
    (df['Year'] == year) &
    (df['Jurisdiction_raw'] == region)
].copy()

if data.empty:
    print(f"\nKeine Daten für {region}, {month}/{year} gefunden!")
else:
    # Features vorbereiten
    X = data[features]

#Imputation
    num_cols = X.select_dtypes(include=["number"]).columns
    X[num_cols] = imputer.transform(X[num_cols])

#Vorhersage
    prediction = model.predict(X)

    print("\n" + "="*50)
    print(f"Region: {region}")
    print(f"Datum: {month}/{year}")
    print(f"Tatsächliche Anzahl Brände: {data['Number_fires'].values[0]}")
    print(f"Vorhergesagte Anzahl Brände: {prediction[0]:.0f}")
    print(f"Abweichung: {abs(data['Number_fires'].values[0] - prediction[0]):.0f}")