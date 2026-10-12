@echo off
REM Re-run NEU-DET and ELPV with the adopted calibration a0=0.6 (constant blend, gate off),
REM so the cross-dataset table uses the same configuration as the MVTec headline.
REM Results: outputs/neu_a06/summary.csv, outputs/pvel_a06/summary.csv
set PY=D:\cvs\pythoncode\yolo-insulator\.venv\Scripts\python.exe
cd /d d:\cvs\pythoncode\AnomalyDetection
if exist logs\cross_a06.log del logs\cross_a06.log
start "" /b "%PY%" code\scripts\run_eval.py --dataset neu --variant A2 --backbone dinov2 --fgsc-mode lowpass --fusion mean --fgsc-lambda 0.6 --fgsc-tau 0.4 --shrinkage 0.1 --image-score-topk 0.01 --image-blur-k 11 --out outputs/neu_a06 > logs\cross_a06.log 2>&1
start "" /b "%PY%" code\scripts\run_eval.py --dataset pvel --variant A2 --backbone dinov2 --fgsc-mode lowpass --fusion mean --fgsc-lambda 0.6 --fgsc-tau 0.4 --shrinkage 0.1 --image-score-topk 0.01 --image-blur-k 11 --out outputs/pvel_a06 >> logs\cross_a06.log 2>&1
echo launched neu+pvel at a0=0.6
