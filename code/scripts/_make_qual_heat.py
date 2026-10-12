# -*- coding: utf-8 -*-
"""SCI-quality qualitative figure (fig_qual) + multi-category heatmap panel
(fig_heat). One light MVTec pass; PNG at 600 dpi + vector PDF."""
from __future__ import annotations
import os, sys, time as _time
_LOCK = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".figs_lock")
if os.path.exists(_LOCK) and (_time.time() - os.path.getmtime(_LOCK)) > 1800:
    try: os.remove(_LOCK)
    except OSError: pass
try:
    _fd = os.open(_LOCK, os.O_CREAT | os.O_EXCL | os.O_RDWR); os.close(_fd)
except FileExistsError:
    sys.stderr.write("another figures instance is running; exiting.\n"); sys.exit(0)
import atexit
atexit.register(lambda: os.remove(_LOCK) if os.path.exists(_LOCK) else None)

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import AnomalyDataset, list_categories
from src.methods.factory import build_method
from src.utils.common import ROOT, load_yaml, set_seed

BASE = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
FIG = os.path.join(ROOT, "paper", "figures")
cfg_ms = BASE["eval"]["map_size"]
set_seed(42)
DEVICE = os.environ.get("DEVICE", "cuda")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "savefig.dpi": 900,
    "figure.dpi": 120,
})

def build():
    cfg = dict(BASE); cfg.setdefault("model", {})["backbone"] = "dinov2"
    # v2 生产配置：对角协方差 + 均值融合 + 固定低通 a0=0.6/K=33
    #            + shrinkage 0.02 + ref_smooth 0 + image_score_topk 0.005
    ov = {"fusion": "mean", "cov_mode": "diag", "fgsc": "lowpass",
          "fgsc_lambda": 0.6, "fgsc_kernel": 33, "shrinkage": 0.02,
          "ref_smooth": 0, "image_score_topk": 0.005, "image_blur_k": 11}
    device = os.environ.get("DEVICE", "cuda")
    return build_method(cfg, "A2", device=device, override_method=ov)

m = build(); bb = m.backbone
img_size = int(getattr(bb, "image_size", 256))
mean = getattr(bb, "image_mean"); std = getattr(bb, "image_std"); ms = int(cfg_ms)
mv_root = os.path.join(ROOT, BASE["data"]["mvtec_root"])

QUAL_CATS = ["bottle", "capsule", "carpet", "grid", "hazelnut", "screw"]  # fig_qual: 6 cats
HEAT_CATS = ["metal_nut", "pill", "screw", "tile", "transistor", "wood"]   # fig_heat: 6 cats
# union of both panels -- every category that will be plotted must be inferred
CATS = sorted(set(QUAL_CATS) | set(HEAT_CATS), key=lambda c: (QUAL_CATS + HEAT_CATS).index(c))
print(f"  will infer {len(CATS)} categories: {CATS}", flush=True)
qual, heat = {}, {}
for cat in CATS:
    m.set_category(cat)
    tr = AnomalyDataset(mv_root, cat, "train", img_size, ms, mean, std, only_good=True)
    m.fit(DataLoader(tr, batch_size=8, shuffle=False, num_workers=0), max_samples=0)
    m.eval()
    te = AnomalyDataset(mv_root, cat, "test", img_size, ms, mean, std)
    got_q = got_h = False
    for b in DataLoader(te, batch_size=8, shuffle=False, num_workers=0):
        out = m.predict(b["image"].to(DEVICE))
        amap = out["anomaly_map"].float().cpu().numpy()[:, 0]
        raw = out["raw_map"].float().cpu().numpy()[:, 0]
        img = b["image"].numpy(); mask = b["mask"].numpy(); lab = b["label"].numpy()
        for i in range(img.shape[0]):
            if int(lab[i]) != 1:
                continue
            im = (img[i].transpose(1, 2, 0) * np.array(std) + np.array(mean)).clip(0, 1)
            s = amap[i]; r0 = raw[i]
            if mask[i].shape != s.shape:
                import cv2
                s = cv2.resize(s, (mask[i].shape[1], mask[i].shape[0]), interpolation=cv2.INTER_LINEAR)
                r0 = cv2.resize(r0, (mask[i].shape[1], mask[i].shape[0]), interpolation=cv2.INTER_LINEAR)
            gt = mask[i].astype(float)
            if gt.shape != im.shape[:2]:  # upsample mask to image resolution for overlay
                import cv2
                gt = cv2.resize(gt, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
            if cat not in qual:
                qual[cat] = (im, gt, s, r0)
            if cat not in heat:
                heat[cat] = (im, gt, s, r0)
        if cat in qual and cat in heat:
            got_q = got_h = True
        if got_q and got_h:
            break
    print(f"  qual/heat {cat} done", flush=True)

# ---- fig_qual: ROW layout. Rows: Input / Ground truth / Raw (before PSC) / Ours (after PSC). ----
cats_q = [c for c in QUAL_CATS if c in qual]
ncol = len(cats_q)
fig, axs = plt.subplots(4, ncol, figsize=(6.5, 4.2), layout="constrained")
axs = np.atleast_2d(axs)
vmax_q = max(qual[c][2].max() for c in cats_q)
for j, cat in enumerate(cats_q):
    im, gt, pred, raw = qual[cat]
    axs[0, j].imshow(im)
    axs[1, j].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axs[2, j].imshow(raw, cmap="inferno", vmin=0, vmax=vmax_q)
    axs[3, j].imshow(pred, cmap="inferno", vmin=0, vmax=vmax_q)
    axs[0, j].set_title(cat, fontsize=9.5, pad=5)
ylabels = ["Input", "Ground truth", "Raw (before PSC)", "Ours (after PSC)"]
for r in range(4):
    for j in range(ncol):
        ax = axs[r, j]
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
    axs[r, 0].set_ylabel(ylabels[r], fontsize=9, rotation=0, ha="right", va="center", labelpad=8)
cb = fig.colorbar(plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, vmax_q)),
                  ax=axs[3, :].tolist(), fraction=0.02, pad=0.01)
cb.set_label("score", fontsize=7.5); cb.ax.tick_params(labelsize=7)
fig.savefig(os.path.join(FIG, "fig_qual.pdf"))
fig.savefig(os.path.join(FIG, "fig_qual.png"), dpi=900)
plt.close(fig)
print("saved fig_qual (900dpi png + pdf)", flush=True)

# ---- fig_heat: ROW layout. Rows: Input / GT overlay / Raw (before PSC) / Ours (after PSC). ----
cats_h = [c for c in HEAT_CATS if c in heat]
ncol = len(cats_h)
fig, axs = plt.subplots(4, ncol, figsize=(6.5, 4.3), layout="constrained")
axs = np.atleast_2d(axs)
vmax = max(heat[c][2].max() for c in cats_h)
for j, cat in enumerate(cats_h):
    im, gt, pred, raw = heat[cat]
    ov = im.copy()
    ov[..., 0] = np.clip(im[..., 0] * (1 - 0.55 * gt), 0, 1)
    axs[0, j].imshow(im)
    axs[1, j].imshow(ov)
    axs[2, j].imshow(raw, cmap="inferno", vmin=0, vmax=vmax)
    axs[3, j].imshow(pred, cmap="inferno", vmin=0, vmax=vmax)
    axs[0, j].set_title(cat, fontsize=9.5, pad=5)
ylabels = ["Input", "GT overlay", "Raw (before PSC)", "Ours (after PSC)"]
for r in range(4):
    for j in range(ncol):
        ax = axs[r, j]
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
    axs[r, 0].set_ylabel(ylabels[r], fontsize=9, rotation=0, ha="right", va="center", labelpad=8)
cb = fig.colorbar(plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, vmax)),
                  ax=axs[3, :].tolist(), fraction=0.02, pad=0.01)
cb.set_label("score", fontsize=7.5)
cb.ax.tick_params(labelsize=7)
fig.savefig(os.path.join(FIG, "fig_heat.pdf"))
fig.savefig(os.path.join(FIG, "fig_heat.png"), dpi=900)
plt.close(fig)
print("saved fig_heat (900dpi png + pdf)", flush=True)
print("ALL DONE", flush=True)
