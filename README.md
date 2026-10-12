# DiagAD — Training-Free Industrial Anomaly Measurement

DiagAD is a training-free industrial anomaly detection method framed as a
*measurement* problem. It combines:

- A frozen **DINOv2 ViT-B/14** visual backbone extracting multi-scale dense
  features at two patch grids (28×28 and 16×16).
- A **diagonal-Gaussian nominal reference** estimated per spatial position from
  normal-only samples — no masks, no gradients, no training.
- **Post-Hoc Calibration (PSC)**: a fixed low-pass filter applied to the score
  map to suppress isolated high-frequency false positives, acting as an
  instrument transfer function.
- A **top-0.5% read-out**: the image-level score is the mean of the top-0.5%
  anomaly-map peaks.

The fitted state is a fixed ≈2 MB (per-position mean + diagonal precision at
28² + 16² positions, 512-dim). No training is performed; every parameter is
derived from normal samples only.

## Repository layout

| Path | Contents |
|------|----------|
| `Paper/measurement/figures/` | All figures as vector PDFs (the only manuscript-related artifacts published here). |
| `code/src/` | Method implementation — `methods/`, `models/`, `modules/` (incl. PSC), `eval/`, `data/`. |
| `code/scripts/` | Evaluation & experimentation scripts (e.g. `_eval_full.py`, `run_eval.py`, frequency / power-spectrum analysis). |
| `code/configs/` | Configurations; `diagad.yaml` mirrors the frozen Table I setting. |
| `code/run_diagad.bat` / `code/run_diagad.sh` | **One-click evaluation** (Windows / Linux-macOS). |
| `code/requirements.txt` | Python dependencies. |
| `weights/README.md` | Where to download the frozen DINOv2 backbone weights. |
| `data/README.md` | Dataset provenance and official download links. |

> The manuscript (editable LaTeX / Word sources and the compiled PDF) is **not**
> included in this repository; only the vector figures are published here. The
> manuscript is delivered to reviewers through the journal submission system. The
> datasets and the backbone weights are likewise not committed — see
> `data/README.md` and `weights/README.md` for their official sources.

## Environment

- Python 3.10+ + PyTorch 2.x + `timm` (CUDA optional; the scripts fall back to
  CPU automatically).
- DINOv2 weights (~346 MB) are downloaded once from the official Meta source
  on first use — see `weights/README.md`.

## Quick start (one click)

Windows:

```bat
code\run_diagad.bat
```

Linux / macOS:

```bash
./code/run_diagad.sh
```

Both scripts locate Python, install missing dependencies, fetch the DINOv2
weights if needed, and run the canonical evaluation on MVTec AD. Pass a
different dataset root and tag to evaluate another dataset, e.g.
`run_diagad.bat data\visa visa` (Windows) or `./run_diagad.sh data/visa visa`
(Linux/macOS). Datasets must be arranged in the MVTec AD directory layout —
see `data/README.md`.

## Manual reproduction

1. Install dependencies: `pip install -r code/requirements.txt`
2. Place a dataset in MVTec layout under `data/<name>` (e.g. MVTec AD →
   `data/mvtec_ad`).
3. Run the canonical evaluation (the frozen Table I configuration):

   ```bash
   python code/scripts/_eval_full.py --root data/mvtec_ad --tag mvtec
   ```

   This uses `backbone: dinov2`, `cov_mode: diag`, `fusion: mean`,
   `fgsc: lowpass` (λ = 0.6, K = 33), `shrinkage: 0.02`, `ref_smooth: 0`,
   `image_score_topk: 0.005`. The equivalent configuration is documented in
   `code/configs/diagad.yaml`.

4. (Optional) quick single-category check via `run_eval.py` — pass the DiagAD
   flags explicitly:

   ```bash
   python code/scripts/run_eval.py --dataset mvtec --categories bottle \
       --backbone dinov2 --cov-mode diag --fusion mean \
       --fgsc-mode lowpass --fgsc-lambda 0.6 --fgsc-kernel 33 \
       --shrinkage 0.02 --ref-smooth 0 --image-score-topk 0.005
   ```

   (Note: `run_eval.py` applies its `--variant` defaults after the config, so
   pass the flags above explicitly to reproduce DiagAD rather than relying on a
   variant alias.)

## Note

Large data, trained weights, and experiment outputs — `data/`, `weights/`,
`outputs/`, `logs/`, `backup/` — are intentionally **excluded** from this
repository. Only the method source, evaluation scripts, configuration, and the
vector figures are included.
