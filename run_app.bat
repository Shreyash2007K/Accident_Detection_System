@echo off
title AI-Based Accident Detection System
color 0B

echo =====================================================================
echo    AI-Based Accident Detection System Using AI & Computer Vision
echo    Second-Year Engineering Mini Project
echo =====================================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python environment...
python --version
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH! Please install Python.
    pause
    exit /b 1
)

echo.
echo [2/3] Checking dependencies...
python -c "import streamlit, cv2, ultralytics" 2>nul
if errorlevel 1 (
    echo [INFO] Installing required dependencies...
    pip install -r requirements.txt
)

echo.
echo [3/3] Launching Streamlit Dashboard...
echo Open your browser at http://localhost:8501 if it does not open automatically.
echo Press Ctrl+C in this terminal window to stop the application.
echo.

streamlit run app.py

pause
