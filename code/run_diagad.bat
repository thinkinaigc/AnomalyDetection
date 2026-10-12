@echo off
REM ============================================================================
REM  DiagAD - one-click evaluation (Windows)
REM
REM  Usage:
REM      run_diagad.bat [dataset_root] [tag]
REM
REM  Arguments (both optional):
REM      dataset_root   dataset root in MVTec AD directory layout.
REM                     Relative paths are resolved against the repository
REM                     root. Default: data\mvtec_ad
REM      tag            label used for the output file (outputs\eval_<tag>_*.json)
REM                     Default: mvtec
REM
REM  Examples:
REM      run_diagad.bat                          (MVTec AD, default)
REM      run_diagad.bat data\visa visa           (VisA cross-dataset check)
REM      run_diagad.bat D:\datasets\mvtec_ad mvtec
REM
REM  What this script does:
REM      1. locates a Python interpreter
REM      2. installs missing dependencies from requirements.txt
REM      3. downloads the frozen DINOv2 ViT-B/14 weights if not cached
REM      4. runs the frozen Table I DiagAD evaluation
REM ============================================================================
setlocal EnableDelayedExpansion

REM --- go to the repository root (parent of code/) ---------------------------
cd /d "%~dp0.."
set "REPO_ROOT=%CD%"

set "DATA_ROOT=data\mvtec_ad"
set "TAG=mvtec"
if not "%~1"=="" set "DATA_ROOT=%~1"
if not "%~2"=="" set "TAG=%~2"
REM resolve a relative dataset path against the repository root
for %%I in ("%DATA_ROOT%") do set "DATA_ROOT=%%~fI"

echo ============================================================================
echo  DiagAD one-click evaluation
echo  repository : %REPO_ROOT%
echo  dataset    : %DATA_ROOT%
echo  tag        : %TAG%
echo ============================================================================

REM --- 1. locate Python -------------------------------------------------------
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY (
    where py >nul 2>nul && set "PY=py"
)
if not defined PY (
    echo [ERROR] No Python interpreter found. Install Python 3.10+ and retry.
    exit /b 1
)
for /f "delims=" %%i in ('%PY% --version 2^>^&1') do echo [INFO] Using %%i

REM --- 2. dependencies --------------------------------------------------------
%PY% -c "import torch, torchvision, timm, numpy, pandas, sklearn, cv2, PIL" >nul 2>nul
if errorlevel 1 (
    echo [INFO] Installing dependencies from code\requirements.txt ...
    %PY% -m pip install -r "%REPO_ROOT%\code\requirements.txt"
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed. Install manually and retry.
        exit /b 1
    )
)

REM --- 3. frozen DINOv2 ViT-B/14 weights --------------------------------------
REM torch hub cache: <hub_dir>\checkpoints ; hub_dir = %%TORCH_HOME%%\hub or
REM %%USERPROFILE%%\.cache\torch\hub by default.
set "CKPT_DIR=%USERPROFILE%\.cache\torch\hub\checkpoints"
for /f "delims=" %%i in ('%PY% -c "import torch;print(torch.hub.get_dir())" 2^>nul') do set "CKPT_DIR=%%i\checkpoints"
set "CKPT=%CKPT_DIR%\dinov2_vitb14_reg4_pretrain.pth"
if exist "%CKPT%" (
    echo [INFO] DINOv2 weights found: %CKPT%
) else (
    echo [INFO] Downloading DINOv2 ViT-B/14 weights ^(about 346 MB^) ...
    echo        from https://dl.fbarespubliciles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
    if not exist "%CKPT_DIR%" mkdir "%CKPT_DIR%"
    curl -L --fail -o "%CKPT%" "https://dl.fbarespubliciles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"
    if errorlevel 1 (
        echo [ERROR] Weight download failed. Download the file manually and
        echo         place it at:  %CKPT%
        exit /b 1
    )
    echo [INFO] Weights saved to %CKPT%
)

REM --- 4. dataset check -------------------------------------------------------
if not exist "%DATA_ROOT%" (
    echo [ERROR] Dataset root not found: %DATA_ROOT%
    echo.
    echo   Download the datasets from their official sources listed in
    echo   data\README.md and arrange them in the MVTec AD layout:
    echo       data\mvtec_ad\bottle\train\good\*.png
    echo       data\mvtec_ad\bottle\test\broken_large\*.png
    echo       data\mvtec_ad\bottle\ground_truth\broken_large\*.png
    exit /b 1
)

REM --- 5. device selection (cuda if available, else cpu) ----------------------
set "DEVICE=cpu"
for /f "delims=" %%i in ('%PY% -c "import torch;print(torch.cuda.is_available())" 2^>nul') do (
    if "%%i"=="True" set "DEVICE=cuda"
)
echo [INFO] Device: %DEVICE%

REM --- 6. run the frozen Table I evaluation -----------------------------------
echo [INFO] Running: python code\scripts\_eval_full.py --root "%DATA_ROOT%" --tag %TAG% --device %DEVICE%
%PY% "%REPO_ROOT%\code\scripts\_eval_full.py" --root "%DATA_ROOT%" --tag %TAG% --device %DEVICE%
if errorlevel 1 (
    echo [ERROR] Evaluation failed. See the messages above.
    exit /b 1
)

echo.
echo [DONE] Results written to %REPO_ROOT%\outputs\eval_%TAG%_*.json
exit /b 0
