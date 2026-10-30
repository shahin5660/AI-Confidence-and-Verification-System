@echo off
REM ============================================================
REM  AI Confidence and Verification System - Windows setup
REM  Double-click this file, or run it from a VS Code terminal.
REM ============================================================
setlocal
cd /d "%~dp0"

echo.
echo [1/4] Checking Python ...
where py >nul 2>nul
if %errorlevel%==0 (
    set "PY=py"
) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
        set "PY=python"
    ) else (
        echo Python was not found.
        echo Install Python 3.10+ from https://www.python.org/downloads/windows/
        echo and tick "Add python.exe to PATH", then run this script again.
        pause
        exit /b 1
    )
)
%PY% --version

echo.
echo [2/4] Creating virtual environment in .venv ...
if not exist ".venv\Scripts\python.exe" (
    %PY% -m venv .venv
    if errorlevel 1 goto :failed
) else (
    echo .venv already exists, reusing it.
)

echo.
echo [3/4] Installing dependencies from requirements.txt ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo.
echo [4/4] Training the model ...
".venv\Scripts\python.exe" train_model.py
if errorlevel 1 goto :failed

echo.
echo Setup finished. Start the app with:  run.bat
echo (or: .venv\Scripts\activate  then  python -m streamlit run app.py)
pause
exit /b 0

:failed
echo.
echo Setup failed. Read the error above, then try again.
pause
exit /b 1
