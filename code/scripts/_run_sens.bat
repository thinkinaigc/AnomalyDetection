@echo off
REM Full hyper-parameter sensitivity sweep (15 MVTec categories).
set PY=D:\cvs\pythoncode\yolo-insulator\.venv\Scripts\python.exe
cd /d d:\cvs\pythoncode\AnomalyDetection
if exist logs\sens.log del logs\sens.log
start "" /b "%PY%" code\scripts\_exp_sensitivity.py > logs\sens.log 2>&1
echo launched
