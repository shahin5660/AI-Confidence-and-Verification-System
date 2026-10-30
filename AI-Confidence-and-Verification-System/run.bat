@echo off
REM ============================================================
REM  AI Confidence and Verification System - launch the app
REM  Run setup.bat first if you have never set this up.
REM ============================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo No virtual environment found. Run setup.bat first.
    pause
    exit /b 1
)

echo Starting Streamlit. Press Ctrl+C in this window to stop.
".venv\Scripts\python.exe" -m streamlit run app.py

if errorlevel 1 (
    echo.
    echo The app stopped with an error. Read the message above.
    pause
)
