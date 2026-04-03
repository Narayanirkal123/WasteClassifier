@echo off
echo ==========================================
echo   WasteScan AI - Local Server Starter
echo ==========================================
echo Starting Python server on port 8000...
start "" http://localhost:8000
python -m http.server 8000
pause
