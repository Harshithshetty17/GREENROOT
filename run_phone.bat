@echo off
REM ===================================================================
REM  GREENROOT - serve the app to your phone over Wi-Fi
REM
REM  Your phone and this computer must be on the SAME Wi-Fi network.
REM  This window must stay open while you use the app on the phone.
REM ===================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ==========================================================
echo   GREENROOT - Phone Access
echo ==========================================================
echo.

REM ---- 1. Find the interpreter -------------------------------------
set "PYTHON=python"
%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    set "PYTHON=py"
    !PYTHON! --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python was not found on your PATH.
        echo         Install Python 3.12+ from https://python.org
        echo.
        pause
        exit /b 1
    )
)

REM ---- 2. Check the model artefacts ---------------------------------
if not exist "models\stacking_model.pkl" (
    echo [ERROR] models\stacking_model.pkl is missing.
    echo.
    pause
    exit /b 1
)

REM ---- 3. Install dependencies on first run -------------------------
%PYTHON% -c "import streamlit, sklearn, shap, lime, fpdf" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies, this may take a few minutes...
    %PYTHON% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
)

REM ---- 4. Work out this computer's Wi-Fi address --------------------
REM  Prefer a private-range IPv4; skip loopback and virtual adapters.
set "LANIP="
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /c:"IPv4 Address"') do (
    set "CANDIDATE=%%a"
    set "CANDIDATE=!CANDIDATE: =!"
    if "!LANIP!"=="" (
        echo !CANDIDATE! | findstr /r "^192\.168\. ^10\. ^172\.1[6-9]\. ^172\.2[0-9]\. ^172\.3[0-1]\." >nul
        if not errorlevel 1 set "LANIP=!CANDIDATE!"
    )
)

if "!LANIP!"=="" (
    echo [!] Could not detect a Wi-Fi address automatically.
    echo     Run 'ipconfig' and look for your IPv4 Address.
    echo     The phone address will be  http://YOUR-IP:8501
    echo.
) else (
    echo   On your phone's browser, open:
    echo.
    echo       http://!LANIP!:8501
    echo.
    echo   Both devices must be on the same Wi-Fi.
    echo.
)

echo   The first time, Windows may ask to allow Python through the
echo   firewall. Tick "Private networks" and click Allow, or the
echo   phone will not be able to connect.
echo.
echo   Keep this window open. Press Ctrl+C here to stop.
echo.

REM ---- 5. Serve on all interfaces so the phone can reach it ----------
REM  enableCORS / enableXsrfProtection must both be off, or Streamlit
REM  refuses the browser's WebSocket from any address other than
REM  localhost and the phone sees a permanently blank page. This drops
REM  a browser-origin check, so only run this on a network you trust
REM  (home or college Wi-Fi) - never on open public Wi-Fi. Anyone on the
REM  same network can open the app while this window is running.
%PYTHON% -m streamlit run app.py ^
    --server.address 0.0.0.0 ^
    --server.port 8501 ^
    --server.headless true ^
    --server.enableCORS false ^
    --server.enableXsrfProtection false

endlocal
pause
