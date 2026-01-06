#!/bin/bash

# Absoluter Projektpfad (HIER ggf. anpassen)
PROJECT_DIR="$HOME/PycharmProjects/firewatch"

# Backend starten
osascript <<EOF
tell application "Terminal"
    do script "cd \"$PROJECT_DIR/backend\" && uvicorn app:app --reload"
end tell
EOF

# Kurze Pause, damit Backend sauber startet
sleep 2

# Frontend starten
osascript <<EOF
tell application "Terminal"
    do script "cd \"$PROJECT_DIR\" && streamlit run dashboard/dashboard.py"
end tell
EOF