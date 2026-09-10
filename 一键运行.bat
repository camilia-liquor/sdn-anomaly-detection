@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo   CNN + BiLSTM SDN 异常检测：一键跑通全流程
echo ============================================================
if exist ".venv\Scripts\python.exe" (
    set PY=.venv\Scripts\python.exe
) else (
    set PY=python
)
echo [1/4] 环境自检...
%PY% tools\check_env.py
if errorlevel 1 (
    echo.
    echo 依赖不完整，请先执行： pip install -r requirements.txt
    pause
    exit /b 1
)
echo.
echo [2/4] 数据链路自检（不依赖 TensorFlow）...
%PY% tools\smoke_test.py
echo.
echo [3/4] 完整实验（生成数据 - 训练 - 评估 - 出图）...
%PY% run_all.py
echo.
echo [4/4] 生成实验报告...
%PY% tools\make_report.py
echo.
echo 完成！结果见 results\ 与 docs\02_实验报告.md
pause
