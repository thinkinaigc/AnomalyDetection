@echo off
REM Full 15-category MVTec verification of the candidate a0=0.6 constant-blend calibration.
REM Protocol identical to the headline run (outputs/mvtec15_A2fgscL used a0=0.8); only a0
REM differs, so the two summary.csv files are directly comparable.
REM Result: outputs/mvtec15_a06/summary.csv   Log: logs/verify_a06.log
set PY=D:\cvs\pythoncode\yolo-insulator\.venv\Scripts\python.exe
cd /d d:\cvs\pythoncode\AnomalyDetection
if exist logs\verify_a06.log del logs\verify_a06.log
start "" /b "%PY%" code\scripts\run_eval.py --dataset mvtec --variant A2 --backbone dinov2 --fgsc-mode lowpass --fusion mean --fgsc-lambda 0.6 --fgsc-tau 0.4 --shrinkage 0.1 --image-score-topk 0.01 --image-blur-k 11 --out outputs/mvtec15_a06 > logs\verify_a06.log 2>&1
echo launched a0=0.6
