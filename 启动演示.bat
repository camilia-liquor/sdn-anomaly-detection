@echo off
rem ============================================================
rem  SDN anomaly detection - live demo dashboard
rem
rem  NOTE: this .bat is intentionally ASCII-only.
rem  cmd.exe parses batch files using the system OEM codepage
rem  (GBK on zh-CN Windows). A UTF-8 batch file therefore gets
rem  mis-parsed: Chinese bytes are mangled AND they swallow the
rem  following ASCII characters, so fragments like "ython.exe"
rem  end up being executed as commands. Chinese output is
rem  produced by the Python programs instead.
rem ============================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "TF_CPP_MIN_LOG_LEVEL=3"
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY set "PY=python"

echo ============================================================
echo   SDN anomaly detection - live demo dashboard
echo   URL    : http://127.0.0.1:5000
echo   Python : %PY%
echo   Stop   : close the "SDN demo server" window
echo ============================================================
echo.

"%PY%" -c "import flask" 2>nul
if errorlevel 1 (
  echo [ERROR] Flask is not installed. Run this first:
  echo         %PY% -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

echo Starting the detection service in a separate window ...
start "SDN demo server" cmd /k ""%PY%" app\app.py --port 5000"
echo Waiting for the service to come up ...
rem ping-based wait: works with or without stdin redirection (timeout.exe fails when redirected)
ping -n 7 127.0.0.1 >nul 2>nul
start "" http://127.0.0.1:5000

echo.
echo The dashboard should now open in your browser.
echo If the page is blank, wait a few seconds and refresh once.
echo.
pause
endlocal
