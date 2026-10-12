# -*- coding: utf-8 -*-
"""Item 4 (IEEE-TIM checklist): radial power-spectrum evidence justifying the
low-pass (PSC) calibration.

For every MVTec test image we take the raw anomaly map (pre-PSC) and the
calibrated map (post-PSC low-pass) plus the GT mask, then compute:
  (A) GLOBAL radial power-spectral density (PSD) of the raw vs calibrated map,
      averaged over images. The calibrated map must carry less high-frequency
      energy (the low-pass removes it).
  (B) LOCAL radial PSD restricted to DEFECT pixels vs FALSE-ALARM pixels
      (normal pixels whose raw score is in the top-5% of the normal region).
      False alarms on textured normal surfaces are isolated spikes -> relatively
      higher high-frequency energy than genuine defect pixels; a low-pass
      therefore suppresses them preferentially.

Backbone is forced to DINOv2 ViT-B/14 (the paper's primary backbone).
Outputs:  outputs/freq_analysis/psd_records.npz
          Paper/figures/fig_freq_psd.pdf  (+ .png)
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "code"))   # src package lives at code/src
from src.data.dataset import AnomalyDataset, list_categories
from src.methods.factory import build_method
from src.utils.common import load_yaml, set_seed

CFG = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
CFG["model"]["backbone"] = "dinov2"   # force the paper's primary backbone
DEVICE = "cuda"
set_seed(42)
OUT_DIR = os.path.join(ROOT, "outputs", "freq_analysis")
FIG = os.path.join(ROOT, "Paper", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
MAP = int(CFG.get("eval", {}).get("map_size", 256))
R = 64            # radial bins
WIN = 27          # local window half-extent side
FA_TOPFRAC = 0.05

C1, C2, C3, GREY = "#2166AC", "#B2182B", "#1B7837", "#8C8C8C"


def radial_bins(spec2d, R):
    H, W = spec2d.shape
    cy, cx = H / 2.0, W / 2.0
    y, x = np.indices(spec2d.shape)
    r = np.clip(np.sqrt((x - cx) ** 2 + (y - cy) ** 2), 0, R - 1).astype(int)
    bins = np.bincount(r.ravel(), weights=spec2d.ravel(), minlength=R)
    cnt = np.bincount(r.ravel(), minlength=R)
    cnt[cnt == 0] = 1
    return bins / cnt


def main():
    m = build_method(CFG, "A2", device=DEVICE,
                     override_method=dict(fusion="mean", fgsc="lowpass",
                                          fgsc_lambda=0.6, fgsc_tau=0.0,
                                          fgsc_kernel=33, shrinkage=0.02,
                                          cov_mode="diag", image_score_topk=0.005))
    bb = m.backbone
    img_size = int(getattr(bb, "image_size", 256))
    mean = tuple(float(x) for x in bb.image_mean)
    std = tuple(float(x) for x in bb.image_std)
    data_root = os.path.join(ROOT, CFG["data"]["mvtec_root"])
    cats = list_categories(data_root)

    g_raw, g_cal = [], []
    loc_def, loc_fa = [], []
    t0 = time.time()
    for ci, cat in enumerate(cats):
        m.set_category(cat)
        m.fit(DataLoader(AnomalyDataset(data_root, cat, "train", img_size, MAP, mean, std, only_good=True),
                         batch_size=8, shuffle=False, num_workers=0))
        ds = AnomalyDataset(data_root, cat, "test", img_size, MAP, mean, std)
        ld = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        for batch in ld:
            img = batch["image"].to(DEVICE)
            mask = batch["mask"].numpy().astype(np.float32)          # [B,H,W]
            with torch.no_grad():
                out = m.predict(img)
            raw = out["raw_map"][:, 0].cpu().numpy()                 # [B,H,W]
            cal = out["anomaly_map"][:, 0].cpu().numpy()
            for i in range(raw.shape[0]):
                g = mask[i]; r = raw[i]; c = cal[i]
                g_raw.append(radial_bins(np.fft.fftshift(np.abs(np.fft.fft2(r)) ** 2), R))
                g_cal.append(radial_bins(np.fft.fftshift(np.abs(np.fft.fft2(c)) ** 2), R))
                norm = g == 0
                if norm.sum() == 0:
                    continue
                thr = np.quantile(r[norm], 1 - FA_TOPFRAC)
                fa_mask = norm & (r >= thr)
                def_mask = g > 0.5
                def_px = np.argwhere(def_mask); fa_px = np.argwhere(fa_mask)
                if len(def_px) == 0 or len(fa_px) == 0:
                    continue
                if len(def_px) > 40:
                    def_px = def_px[np.random.choice(len(def_px), 40, replace=False)]
                if len(fa_px) > 40:
                    fa_px = fa_px[np.random.choice(len(fa_px), 40, replace=False)]
                hw = WIN // 2
                sp = lambda w: np.fft.fftshift(np.abs(np.fft.fft2(w)) ** 2, axes=(-2, -1))
                # pad so every window is exactly WIN x WIN even at the border
                r_pad = np.pad(r, hw, mode="edge")
                def_w = np.stack([r_pad[y:y + WIN, x:x + WIN] for (y, x) in def_px])
                fa_w = np.stack([r_pad[y:y + WIN, x:x + WIN] for (y, x) in fa_px])
                loc_def.append(np.mean([radial_bins(s, R) for s in sp(def_w)], 0))
                loc_fa.append(np.mean([radial_bins(s, R) for s in sp(fa_w)], 0))
        print(f"[{ci + 1}/{len(cats)}] {cat} done", flush=True)

    g_raw = np.mean(g_raw, 0); g_cal = np.mean(g_cal, 0)
    loc_def = np.mean(loc_def, 0); loc_fa = np.mean(loc_fa, 0)

    def _norm(a):
        s = a.sum(); return a / s if s > 0 else a
    curves = dict(
        radius=np.arange(R, dtype=float),
        global_raw=_norm(g_raw), global_cal=_norm(g_cal),
        local_defect=_norm(loc_def), local_fa=_norm(loc_fa),
    )
    np.savez(os.path.join(OUT_DIR, "psd_records.npz"), **curves)

    # high-frequency energy fraction (outer 40% of radii)
    hf = slice(R * 3 // 5, None)
    hf_raw = float(g_raw[hf].sum() / g_raw.sum())
    hf_cal = float(g_cal[hf].sum() / g_cal.sum())
    hf_def = float(loc_def[hf].sum() / loc_def.sum())
    hf_fa = float(loc_fa[hf].sum() / loc_fa.sum())
    summary = dict(hf_raw=hf_raw, hf_cal=hf_cal, hf_defect=hf_def, hf_fa=hf_fa)
    json.dump(summary, open(os.path.join(OUT_DIR, "psd_summary.json"), "w"), indent=2)
    print("PSD HF-fraction: raw=%.4f cal=%.4f | defect=%.4f fa=%.4f" % (hf_raw, hf_cal, hf_def, hf_fa))

    # ---------------- figure ----------------
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 10,
        "axes.titlesize": 10, "axes.linewidth": 0.8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8.5, "legend.frameon": False,
        "savefig.dpi": 900, "figure.dpi": 120,
    })
    freq = np.arange(R, dtype=float) / (R - 1)   # normalized frequency 0..1
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7))

    a1.plot(freq, curves["global_raw"], color=C2, lw=1.4, label="raw (pre-PSC)")
    a1.plot(freq, curves["global_cal"], color=C1, lw=1.4, label="calibrated (post-PSC)")
    a1.fill_between(freq, 0, curves["global_raw"], color=C2, alpha=0.10)
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("normalized spatial frequency")
    a1.set_ylabel("power spectral density (norm.)")
    a1.set_title("Global PSD: low-pass removes high-frequency content")
    a1.axvline(freq[R * 3 // 5], ls="--", lw=0.7, color=GREY)
    a1.legend(loc="upper right", fontsize=7.5)
    a1.spines["top"].set_visible(False); a1.spines["right"].set_visible(False)
    a1.grid(True, lw=0.4, color="#DDDDDD"); a1.set_axisbelow(True)

    a2.plot(freq, curves["local_defect"], color=C3, lw=1.4, label="defect pixels")
    a2.plot(freq, curves["local_fa"], color=C2, lw=1.4, label="false-alarm pixels")
    a2.set_xscale("log"); a2.set_yscale("log")
    a2.set_xlabel("normalized spatial frequency")
    a2.set_ylabel("power spectral density (norm.)")
    a2.set_title("Local PSD: false alarms are more high-frequency")
    a2.axvline(freq[R * 3 // 5], ls="--", lw=0.7, color=GREY)
    a2.legend(loc="upper right", fontsize=7.5)
    a2.spines["top"].set_visible(False); a2.spines["right"].set_visible(False)
    a2.grid(True, lw=0.4, color="#DDDDDD"); a2.set_axisbelow(True)

    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(FIG, "fig_freq_psd.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig_freq_psd.png"), bbox_inches="tight", dpi=900)
    plt.close(fig)
    print("saved fig_freq_psd  (%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
