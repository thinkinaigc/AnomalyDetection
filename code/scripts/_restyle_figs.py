# -*- coding: utf-8 -*-
"""Restyle the statistical figures to SCI quality: serif/STIX fonts, clean
spines, colour-blind-safe palette, 600 dpi PNG + vector PDF. Uses data.json."""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = r"d:\cvs\pythoncode\AnomalyDetection\paper\figures"
data = json.load(open(os.path.join(FIG, "data.json")))
mv, neu, pvel, agg = data["mvtec"], data["neu"], data["pvel"], data["agg"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "savefig.dpi": 900,
    "figure.dpi": 120,
})
C1, C2 = "#2166AC", "#E69F00"   # blue / orange (colour-blind safe, journal-consistent palette)
C3 = "#2166AC"                   # reuse blue (was green) for aggregation bars
GREY = "#8C8C8C"

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=900)
    plt.close(fig)
    print("saved", name, flush=True)

def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, lw=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)

# ---------- per-category (object categories first, then texture categories) ----------
OBJ = ["bottle","cable","capsule","hazelnut","metal_nut","pill","screw","transistor","toothbrush"]
TEX = ["carpet","grid","leather","tile","wood","zipper"]
cats_ = [c for c in OBJ + TEX if c in mv]
img = [mv[c].get("img", np.nan) for c in cats_]
pix = [mv[c]["pix"] for c in cats_]
img_e = [mv[c].get("img_std", 0) or 0 for c in cats_]
pix_e = [mv[c].get("pix_std", 0) or 0 for c in cats_]
x = np.arange(len(cats_)); w = 0.4
fig, ax = plt.subplots(figsize=(5.6, 3.1))
ax.bar(x - w/2, img, w, label="image AUROC", color=C1, edgecolor="none", zorder=3)
ax.bar(x + w/2, pix, w, label="pixel AUROC", color=C2, edgecolor="none", zorder=3)
if any(v > 0 for v in img_e + pix_e):   # error bars appear only when std is present in data.json
    ax.errorbar(x - w/2, img, yerr=img_e, fmt="none", ecolor="#333333", elinewidth=0.6, capsize=1.6, zorder=4)
    ax.errorbar(x + w/2, pix, yerr=pix_e, fmt="none", ecolor="#333333", elinewidth=0.6, capsize=1.6, zorder=4)
ax.axvline(len(OBJ) - 0.5, ls="--", lw=0.7, color=GREY, zorder=2)  # group divider
ax.set_xticks(x); ax.set_xticklabels(cats_, rotation=45, ha="right")
ax.set_ylabel("AUROC"); ax.set_ylim(0.3, 1.04)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
style(ax)
save(fig, "fig_percat")

# ---------- aggregation ceiling ----------
keys = ["max", "top1%", "top5%", "top20%", "mean", "median"]
lbls = ["maximum", "top-1%", "top-5%", "top-20%", "mean", "median"]
vals = [agg[k] for k in keys]
# peak-based rules: sequential blue gradient (ColorBrewer Blues, colour-blind safe);
# mean/median kept neutral grey; kNN reference line stays orange (C2)
BLUES = ["#08519C", "#2171B5", "#4292C6", "#6BAED6"]
LORANGE = C2   # unified main orange #E69F00 for mean/median control bars
cols = BLUES + [LORANGE, LORANGE]
fig, ax = plt.subplots(figsize=(4.4, 2.7))
bars = ax.bar(lbls, vals, color=cols, width=0.62, zorder=3)
ax.set_ylabel("image-level AUROC"); ax.set_ylim(0.76, 0.95)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.003, f"{v:.3f}", ha="center", fontsize=7.5)
ax.axhline(0.9747, ls="--", lw=0.9, color=C2)
ax.text(0.02, 0.977, "kNN memory bank (0.975)", fontsize=7.5, color=C2, transform=ax.get_yaxis_transform())
plt.xticks(rotation=20)
style(ax)
save(fig, "fig_agg")

# ---------- frequency analysis ----------
thirds = ["low $T$", "mid $T$", "high $T$"]
reduction = [0.0148, 0.0158, 0.0195]
before, after = 0.7887, 0.7719
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.6))
bars1 = a1.bar(thirds, reduction, color=[C1, C1, C2], width=0.55, zorder=3)
a1.set_ylabel("high-score tail reduction")
a1.set_ylim(0, 0.024)
for b, v in zip(bars1, reduction):
    a1.text(b.get_x() + b.get_width()/2, v + 0.0005, f"{v:.4f}", ha="center", fontsize=7.5)
style(a1)
bars2 = a2.bar(["before PSC", "after PSC"], [before, after], color=[C1, LORANGE], width=0.5, zorder=3)
a2.set_ylabel("top-5% tail in normal regions")
a2.set_ylim(0.75, 0.80)
for b, v in zip(bars2, [before, after]):
    a2.text(b.get_x() + b.get_width()/2, v + 0.0004, f"{v:.4f}", ha="center", fontsize=7.5)
style(a2)
plt.tight_layout(pad=0.5)
save(fig, "fig_freq")

# ---------- cross-dataset (error bars if std available) ----------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
for ax, dat, title in [(axes[0], neu, "NEU-DET (steel surfaces)"),
                       (axes[1], pvel, "ELPV (photovoltaic cells)")]:
    cs = sorted(dat.keys())
    im_ = [dat[c].get("img", np.nan) for c in cs]
    px = [dat[c]["pix"] for c in cs]
    im_e = [dat[c].get("img_std", 0) or 0 for c in cs]
    px_e = [dat[c].get("pix_std", 0) or 0 for c in cs]
    x = np.arange(len(cs)); w = 0.4
    if dat is neu:   # NEU-DET image AUROC is a benchmark ceiling (1.0); show pixel only
        ax.bar(x, px, w, label="pixel AUROC", color=C2, zorder=3)
        if any(v > 0 for v in px_e):
            ax.errorbar(x, px, yerr=px_e, fmt="none", ecolor="#333333", elinewidth=0.5, capsize=1.2, zorder=4)
    else:
        ax.bar(x - w/2, im_, w, label="image AUROC", color=C1, zorder=3)
        ax.bar(x + w/2, px, w, label="pixel AUROC", color=C2, zorder=3)
        if any(v > 0 for v in im_e + px_e):
            ax.errorbar(x - w/2, im_, yerr=im_e, fmt="none", ecolor="#333333", elinewidth=0.5, capsize=1.2, zorder=4)
            ax.errorbar(x + w/2, px, yerr=px_e, fmt="none", ecolor="#333333", elinewidth=0.5, capsize=1.2, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(cs, rotation=45, ha="right", fontsize=6.5)
    ax.set_ylim(0.3, 1.03)
    ax.set_title(title, fontsize=9)
    style(ax)
axes[0].set_ylabel("AUROC")
plt.tight_layout(pad=0.5)
_h, _l = axes[1].get_legend_handles_labels()
fig.legend(_h, _l, loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=2, frameon=False)
save(fig, "fig_cross")
print("ALL RESTYLED", flush=True)
