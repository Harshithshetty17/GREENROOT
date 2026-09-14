@echo off
REM ===================================================================
REM  GREENROOT - Intelligent Precision Agriculture Decision Support
REM  One-click Windows launcher.
REM
REM  Verifies the interpreter, installs dependencies on first run,
REM  confirms the pre-trained artefacts are present, then serves the
REM  Streamlit dashboard.
REM ===================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ==========================================================
echo   GREENROOT - Precision Agriculture Decision Support
echo ==========================================================
echo.

REM ---- 1. Locate a Python interpreter ------------------------------
set "PYTHON=python"
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    set "PYTHON=py"
    !PYTHON! --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python was not found on your PATH.
        echo         Install Python 3.10 or newer from https://python.org
        echo         and tick "Add Python to PATH" during setup.
        echo.
        pause
        exit /b 1
    )
)
for /f "tokens=*" %%v in ('%PYTHON% --version 2^>^&1') do set "PYVER=%%v"
echo [1/4] Interpreter  : !PYVER!

REM ---- 2. Confirm the pre-trained artefacts are present -------------
set "MISSING="
if not exist "models\stacking_model.pkl" set "MISSING=!MISSING! stacking_model.pkl"
if not exist "models\scaler.pkl"         set "MISSING=!MISSING! scaler.pkl"
if not exist "models\class_names.pkl"    set "MISSING=!MISSING! class_names.pkl"
if defined MISSING (
    echo [ERROR] Missing model artefact^(s^):!MISSING!
    echo         Restore them into models\ or run: %PYTHON% train.py
    echo.
    pause
    exit /b 1
)
echo [2/4] Artefacts    : stacking_model.pkl, scaler.pkl, class_names.pkl

REM ---- 3. Install dependencies on first run -------------------------
%PYTHON% -c "import streamlit, sklearn, shap, lime, fpdf" >nul 2>&1
if errorlevel 1 (
    echo [3/4] Dependencies : installing, this may take a few minutes...
    %PYTHON% -m pip install --upgrade pip >nul 2>&1
    %PYTHON% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] Dependency installation failed.
        echo         Try manually: %PYTHON% -m pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
) else (
    echo [3/4] Dependencies : already satisfied
)

REM ---- 4. Launch ----------------------------------------------------
echo [4/4] Launching    : http://localhost:8501
echo.
echo Press Ctrl+C in this window to stop the server.
echo.
%PYTHON% -m streamlit run app.py

endlocal
pause
