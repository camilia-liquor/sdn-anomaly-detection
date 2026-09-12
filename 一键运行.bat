@echo off
rem ============================================================
rem  SDN anomaly detection - one-click full pipeline
rem
rem  Usage:
rem    run.bat              full pipeline (data - training - eval - report)
rem    run.bat --smoke      quick smoke run only (about 25 s)
rem    run.bat --check      environment + data-pipeline check only
rem
rem  NOTE: this .bat is intentionally ASCII-only.
rem  cmd.exe parses batch files using the system OEM codepage
rem  (GBK on zh-CN Windows), so a UTF-8 batch file gets mis-parsed
rem  and fragments of Chinese text end up executed as commands.
rem  Chinese output is produced by the Python programs instead.
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
echo   CNN + BiLSTM SDN anomaly detection - full pipeline
echo   Python : %PY%
echo ============================================================
echo.
echo [1/4] Environment check (python / tensorflow / flask ...)
"%PY%" tools\check_env.py
if errorlevel 1 (
  echo.
  echo [ERROR] Dependencies are incomplete. Run this first:
  echo         %PY% -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

echo.
echo [2/4] Data pipeline self-check (no TensorFlow needed)
"%PY%" tools\smoke_test.py

if /i "%~1"=="--check" (
  echo.
  echo Check finished. Run without arguments to execute the full pipeline.
  pause
  exit /b 0
)

if /i "%~1"=="--smoke" (
  echo.
  echo [3/4] Quick smoke run (small data, isolated dirs)
  "%PY%" run_all.py --smoke
  echo.
  echo Smoke run finished. See the log above for the result.
  pause
  exit /b 0
)

echo.
echo [3/4] Full experiment (generate data - train - evaluate - figures)
echo       This takes about 8 minutes on CPU. Please wait ...
"%PY%" run_all.py
if errorlevel 1 (
  echo.
  echo [ERROR] The experiment failed. See the log above.
  pause
  exit /b 1
)

echo.
echo [4/4] Generate the experiment report
"%PY%" tools\make_report.py

echo.
echo Done. Results are in the "results" folder and docs\02_experiment_report.
echo Next steps you may want:
echo   "%PY%" app\app.py                              (live dashboard)
echo   "%PY%" tools\acceptance_check.py               (104-item acceptance check)
echo   "%PY%" tools\live_demo.py                      (one-command attack demo)
echo.
pause
endlocal
