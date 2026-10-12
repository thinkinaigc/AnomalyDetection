@echo off
REM Paired combinatorial search over PSC/read-out parameters on the cached maps.
REM Answers whether a0=0.4/0.5 beats the adopted a0=0.6, and finds the Pareto front.
REM Log: logs\search_paired.log
set PY=D:\cvs\pythoncode\yolo-insulator\.venv\Scripts\python.exe
cd /d d:\cvs\pythoncode\AnomalyDetection
if exist logs\search_paired.log del logs\search_paired.log
start "" /b "%PY%" code\scripts\_exp_search_paired.py > logs\search_paired.log 2>&1
echo launched paired search
