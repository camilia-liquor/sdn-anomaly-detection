@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    set PY=.venv\Scripts\python.exe
) else (
    set PY=python
)
echo ============================================================
echo   启动异常检测演示大屏  →  http://127.0.0.1:5000
echo   （按 Ctrl+C 停止；演示不依赖 TensorFlow）
echo ============================================================
%PY% app\app.py --port 5000
pause
