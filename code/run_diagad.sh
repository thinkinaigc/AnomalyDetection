#!/usr/bin/env bash
# ============================================================================
#  DiagAD - one-click evaluation (Linux / macOS)
#
#  Usage:
#      ./run_diagad.sh [dataset_root] [tag]
#
#  Arguments (both optional):
#      dataset_root   dataset root in MVTec AD directory layout.
#                     Relative paths are resolved against the repository
#                     root. Default: data/mvtec_ad
#      tag            label used for the output file (outputs/eval_<tag>_*.json)
#                     Default: mvtec
#
#  Examples:
#      ./run_diagad.sh                          (MVTec AD, default)
#      ./run_diagad.sh data/visa visa           (VisA cross-dataset check)
#      ./run_diagad.sh /data/mvtec_ad mvtec
#
#  What this script does:
#      1. locates a Python interpreter
#      2. installs missing dependencies from requirements.txt
#      3. downloads the frozen DINOv2 ViT-B/14 weights if not cached
#      4. runs the frozen Table I DiagAD evaluation
# ============================================================================
set -euo pipefail

# --- go to the repository root (parent of code/) -----------------------------
cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

DATA_ROOT="${1:-data/mvtec_ad}"
TAG="${2:-mvtec}"
# resolve a relative dataset path against the repository root
case "$DATA_ROOT" in
    /*) ;;  # already absolute
    *) DATA_ROOT="$REPO_ROOT/$DATA_ROOT" ;;
esac

echo "============================================================================"
echo " DiagAD one-click evaluation"
echo " repository : $REPO_ROOT"
echo " dataset    : $DATA_ROOT"
echo " tag        : $TAG"
echo "============================================================================"

# --- 1. locate Python ---------------------------------------------------------
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "[ERROR] No Python interpreter found. Install Python 3.10+ and retry."
    exit 1
fi
echo "[INFO] Using $($PY --version 2>&1)"

# --- 2. dependencies ----------------------------------------------------------
if ! $PY -c "import torch, torchvision, timm, numpy, pandas, sklearn, cv2, PIL" >/dev/null 2>&1; then
    echo "[INFO] Installing dependencies from code/requirements.txt ..."
    $PY -m pip install -r "$REPO_ROOT/code/requirements.txt"
fi

# --- 3. frozen DINOv2 ViT-B/14 weights ----------------------------------------
# torch hub cache: <hub_dir>/checkpoints ; hub_dir = $TORCH_HOME/hub or
# ~/.cache/torch/hub by default.
CKPT_DIR="$($PY -c 'import torch, os; print(os.path.join(torch.hub.get_dir(), "checkpoints"))')"
CKPT="$CKPT_DIR/dinov2_vitb14_reg4_pretrain.pth"
if [ -f "$CKPT" ]; then
    echo "[INFO] DINOv2 weights found: $CKPT"
else
    echo "[INFO] Downloading DINOv2 ViT-B/14 weights (about 346 MB) ..."
    echo "       from https://dl.fbarespubliciles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"
    mkdir -p "$CKPT_DIR"
    curl -L --fail -o "$CKPT" "https://dl.fbarespubliciles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"
    echo "[INFO] Weights saved to $CKPT"
fi

# --- 4. dataset check ---------------------------------------------------------
if [ ! -d "$DATA_ROOT" ]; then
    echo "[ERROR] Dataset root not found: $DATA_ROOT"
    echo
    echo "  Download the datasets from their official sources listed in"
    echo "  data/README.md and arrange them in the MVTec AD layout:"
    echo "      data/mvtec_ad/bottle/train/good/*.png"
    echo "      data/mvtec_ad/bottle/test/broken_large/*.png"
    echo "      data/mvtec_ad/bottle/ground_truth/broken_large/*.png"
    exit 1
fi

# --- 5. device selection (cuda if available, else cpu) ------------------------
if $PY -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE=cuda
else
    DEVICE=cpu
fi
echo "[INFO] Device: $DEVICE"

# --- 6. run the frozen Table I evaluation -------------------------------------
echo "[INFO] Running: python code/scripts/_eval_full.py --root \"$DATA_ROOT\" --tag $TAG --device $DEVICE"
$PY "$REPO_ROOT/code/scripts/_eval_full.py" --root "$DATA_ROOT" --tag "$TAG" --device "$DEVICE"

echo
echo "[DONE] Results written to $REPO_ROOT/outputs/eval_${TAG}_*.json"
