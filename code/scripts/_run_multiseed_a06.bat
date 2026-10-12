@echo off
REM Re-run the 3-seed robustness check with the adopted calibration a0=0.6 (constant blend).
REM Log: logs/multiseed_a06.log
set PY=D:\cvs\pythoncode\yolo-insulator\.venv\Scripts\python.exe
cd /d d:\cvs\pythoncode\AnomalyDetection
if exist logs\multiseed_a06.log del logs\multiseed_a06.log
set FGSC_LAMBDA=0.6
start "" /b "%PY%" code\scripts\_exp_multiseed.py > logs\multiseed_a06.log 2>&1
echo launched multiseed a0=0.6
