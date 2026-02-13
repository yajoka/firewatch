# KI zur Prognose von Waldbränden in Kanada

## Projektübersicht
Dieses Projekt dient der **monatlichen Brandprognose von Waldbränden in den Provinzen Kanadas**.  
Basierend auf historischen Daten der **Canadian National Fire Database (NFDP)** und exogene Wetterdaten entstand ein Prototyp, der Brandrisiken frühzeitig erkennt und Behörden bei präventiven Maßnahmen unterstützt.

---

## Projektinformationen
**Projektzeitraum:** 28.10.2025 – 13.02.2026  
**Institution:** Hochschule Offenburg  
**Projektleitung:** Yannik Kälble  

---

## Projektumfang (Scope)
- **Datenquelle:** [Canadian National Fire Database (NFDP)](http://nfdp.ccfm.org/en/data/fires.php)
  - **Verwendete Kategorien:**
  - *3.1.2* – Zeitliche Verteilung der Brände  
- **Datenquelle:** (https://open-meteo.com)  

---

## Projektorganisation
| Rolle | Name |
|-------|------|
| **Product Owner** | Yannik Kälble |
| **Scrum Master** | Chiara Facciola |
| **Entwicklungsteam** | Johannes Wolf, Timm Aubele, Mustafa Zangana, Chiara Facciola, Yannik Kälble |
| **Dokumentation** | Mustafa Zangana |
| **Miro-Board-Verwaltung** | Timm Aubele, Johannes Wolf |


---

## Projektstruktur
project_root/
├── backend/ # Logik der Anwendung
├── dashboard/ # Frontend der Anwendung
├── data/ # Rohdaten (NFDP CSV-Dateien)
├── models/ # Trainiertes Modell
├── notebook/ # Jupyter-Notebooks für Analyse & Modelltraining
├── orga/ # Koordinatendarstellung (grafisch)
└── test/ # Testen der Modelle

## Technische Tools und Technologien
•    Programmiersprache: Python
•    Datenaufbereitung & Numerik: Pandas (Datenhandling/ETL), NumPy (numerische Berechnungen)
•    Machine Learning & Modellierung: Scikit-learn (Preprocessing, Random Forest, Metriken/Evaluierung), XGBoost (Gradient Boosting Regressor)
•    Zeitreihenanalyse: Statsmodels (ACF/PACF, Zeitreihen-Analyse)
•    Datenvisualisierung: Matplotlib und Seaborn (EDA und Ergebnisplots)
•    Modell-Export & Wiederverwendbarkeit: Joblib (Speichern/Laden von Imputer, Mo-dellen, Feature-Listen)
•    Anwendung: Streamlit, FastAPI
•    Versionierung & Zusammenarbeit: GitHub (Quellcodeverwaltung), Google Work-space & Miro (Dokumentation/Konzeption)
•    Projektmanagement & Kommunikation: Agiles Vorgehen nach Scrum; Kommunika-tion über Discord; Sprint-Planung/Visualisierung über Miro1
•    Sonstige Werkzeuge: Jupyter Notebooks (explorative Entwicklung, Modelltests, reproduzierbare Dokumentation)

---

## Vorgehensmodell
Das Projekt folgt dem **agilen Scrum-Framework**:
1. **Sprints:** Iterative Entwicklungszyklen (2 Wochen)
2. **Weekly Scrums:** Laufende Status- und Fortschrittsbesprechungen
3. **Sprint Reviews:** Präsentation und Bewertung der Zwischenergebnisse
4. **Retrospektiven:** Analyse und Optimierung der Teamprozesse

---

## Risikomanagement
- **Versionierung & Backup:** Quellcode und Daten über GitHub
- **Technische Risiken:**  
  - Unvollständige oder ungenaue Daten  
  - Modellüberfitting oder Unteranpassung  
- **Gegenmaßnahmen:**  
  - Validierung und Bereinigung der Daten  
  - Einsatz mehrerer Modelltypen und Vergleich der Ergebnisse  

---

## Kommunikationsplan
| Kommunikationstyp | Plattform / Medium | Frequenz |
|-------------------|--------------------|-----------|
| Teamkoordination | Discord | Wöchentlich |
| Scrum-Meetings | Google Meet / Miro | Wöchentlich |
| Reviews & Präsentationen | Google Workspace | Monatlich |
| Dokumentation | Google Docs | Laufend |

---

## Qualitätssicherung
- Iterative Entwicklung und Validierung nach Scrum-Prinzipien  
- Code-Reviews innerhalb des Teams  
- Tests zur **Prognosegenauigkeit** anhand realer Branddaten  
- Geplante Durchführung von Penetration-Tests für Prototypen  

---

## Projektabschluss und Dokumentation
- **Abschlusspräsentation:** 20.01.2026  
- Laufende Dokumentation in Google Docs und Miro
- **Abschlussbericht:** Enthält Ergebnisse, Bewertung und *Lessons Learned*  

---

## 🪪 Lizenz
© 2025 Hochschule Offenburg – Projektteam *KI Waldbrände Kanada*  
Dieses Projekt dient ausschließlich Forschungs- und Bildungszwecken.
