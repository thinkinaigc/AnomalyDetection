@echo off
REM 在已有 15 类缓存(shrink=0.10)上枚举校准/读出参数 —— 零 fit 成本，用于寻找可提升的配置。
REM 结果: logs\tune_readout.log
set PY=D:\cvs\pythoncode\yolo-insulator\.venv\Scripts\python.exe
cd /d d:\cvs\pythoncode\AnomalyDetection
if exist logs\tune_readout.log del logs\tune_readout.log
if exist logs\tune_readout.err del logs\tune_readout.err
start "" /b "%PY%" code\scripts\_exp_tune_readout.py > logs\tune_readout.log 2> logs\tune_readout.err
echo launched -> logs\tune_readout.log
