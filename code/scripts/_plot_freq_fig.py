# -*- coding: utf-8 -*-
"""Regenerate fig_freq_psd.pdf from psd_records.npz with honest framing
(Item 4, IEEE-TIM): (a) anomaly-map PSD is intrinsically low-frequency
dominated -> high-frequency content is negligible noise -> a low-pass is
well justified; (b) false-alarm pixels carry more high-frequency energy
than genuine defect pixels -> the PSC low-pass preferentially suppresses
false alarms.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NPZ = os.path.join(ROOT, "outputs", "freq_analysis", "psd_records.npz")
FIG = os.path.join(ROOT, "Paper", "figures")

z = np.load(NPZ, allow_pickle=True)
r = z["radius"]; g_raw = z["global_raw"]; g_cal = z["global_cal"]
ld = z["local_defect"]; lf = z["local_fa"]
R = len(r)

def bands(c):
    lo = c[r <= 0.3].sum(); hi = c[r > 0.6].sum(); tot = c.sum()
    return lo, hi, tot

glo, ghi, gtot = bands(g_cal)
dlo, dhi, dtot = bands(ld)
flo, fhi, ftot = bands(lf)
fa_ratio = fhi / dhi if dhi > 0 else float("nan")

C1, C2, C3, GREY = "#2166AC", "#B2182B", "#1B7837", "#8C8C8C"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 10,
    "axes.titlesize": 9.5, "axes.linewidth": 0.8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
    "savefig.dpi": 900, "figure.dpi": 120,
})
freq = r / (R - 1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7))

# Panel A: anomaly-map PSD is low-frequency dominated
a1.plot(freq, g_cal, color=C1, lw=1.5, label="calibrated anomaly map")
a1.plot(freq, g_raw, color=C2, lw=0.9, ls=":", alpha=0.8, label="pre-PSC (raw)")
a1.fill_between(freq, 0, g_cal, color=C1, alpha=0.10)
a1.set_xscale("log"); a1.set_yscale("log")
a1.set_xlabel("normalized spatial frequency $f/f_{Nyq}$")
a1.set_ylabel("power spectral density")
a1.set_title("Anomaly map is low-frequency dominated")
a1.axvline(freq[int(R * 0.6)], ls="--", lw=0.7, color=GREY)
a1.annotate("%.1f%% energy\nbelow 0.6 $f_{Nyq}$" % (100 * glo / gtot),
            xy=(0.18, 0.5), xycoords="axes fraction", fontsize=7.5, color=C1,
            ha="left", va="center")
a1.legend(loc="upper right", fontsize=7)
a1.spines["top"].set_visible(False); a1.spines["right"].set_visible(False)
a1.grid(True, lw=0.4, color="#DDDDDD"); a1.set_axisbelow(True)

# Panel B: false alarms carry more high-frequency energy
a2.plot(freq, ld, color=C3, lw=1.4, label="defect pixels")
a2.plot(freq, lf, color=C2, lw=1.4, label="false-alarm pixels")
a2.set_xscale("log"); a2.set_yscale("log")
a2.set_xlabel("normalized spatial frequency $f/f_{Nyq}$")
a2.set_ylabel("power spectral density")
a2.set_title("False alarms are more high-frequency")
a2.axvline(freq[int(R * 0.6)], ls="--", lw=0.7, color=GREY)
a2.annotate("false alarms: %.0f%% more\nhigh-freq energy" % (100 * (fa_ratio - 1)),
            xy=(0.55, 0.4), xycoords="axes fraction", fontsize=7.5, color=C2,
            ha="left", va="center")
a2.legend(loc="upper right", fontsize=7)
a2.spines["top"].set_visible(False); a2.spines["right"].set_visible(False)
a2.grid(True, lw=0.4, color="#DDDDDD"); a2.set_axisbelow(True)

fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(FIG, "fig_freq_psd.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "fig_freq_psd.png"), bbox_inches="tight", dpi=900)
plt.close(fig)
print("saved fig_freq_psd (regen). glo%%=%.2f  fa/def hi ratio=%.3f" % (100 * glo / gtot, fa_ratio))
