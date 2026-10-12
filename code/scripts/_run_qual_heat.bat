@echo off
REM Regenerate fig_qual / fig_heat (row layout) in the background.
set PY=D:\cvs\pythoncode\yolo-insulator\.venv\Scripts\python.exe
cd /d d:\cvs\pythoncode\AnomalyDetection
if exist logs\qualheat.log del logs\qualheat.log
start "" /b "%PY%" code\scripts\_make_qual_heat.py > logs\qualheat.log 2>&1
echo launched
