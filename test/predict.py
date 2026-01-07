import joblib
import pandas as pd

#Modell, Features und Imputer laden
model = joblib.load("../models/xgboost_final_model_v1.joblib")
features = joblib.load("../models/xgboost_features_v1.joblib")
imputer = joblib.load("../backend/imputer.joblib")

#Daten laden
df = pd.read_csv("../data/final_data_v1.csv")

#5 zufällige Zeilen ohne NaNs in Number_fires nehmen
data = df[df['Number_fires'].notna()].sample(n=5, random_state=42).copy()

#Nur die Features behalten
X = data[features]

#Imputation auf numerische Features
num_cols = X.select_dtypes(include=["number"]).columns
X[num_cols] = imputer.transform(X[num_cols])

#Vorhersage
predictions = model.predict(X)

#Ergebnisse anzeigen
results = pd.DataFrame({
    'Year': data['Year'].values,
    'Month': data['Month'].values,
    'Jurisdiction': data['Jurisdiction_raw'].values,
    'Tatsächlich': data['Number_fires'].values,
    'Vorhergesagt': predictions.round(0)
})

print(results)
print(f"\nDurchschnittlicher Fehler: {abs(results['Tatsächlich'] - results['Vorhergesagt']).mean():.1f}")