# -*- coding: utf-8 -*-
"""M3: hyper-parameter sensitivity of the post-hoc score calibration (PSC).

Strategy (cheap and exact): run the backbone ONCE per category with the
calibration switched off, cache the raw probability map and the texture map for
every test image, then enumerate calibration / read-out settings in memory.
`shrinkage` is the only parameter that changes `fit`, so it is swept separately.

Outputs
-------
exp/sensitivity/psc_sensitivity.csv   one row per configuration
exp/sensitivity/psc_sensitivity.md    markdown table ready for the paper
"""
from __future__ import annotations

import os
import sys
import time as _time

_LOCK = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".sens_lock")
if os.path.exists(_LOCK) and (_time.time() - os.path.getmtime(_LOCK)) > 3600:
    try:
        os.remove(_LOCK)
    except OSError:
        pass
try:
    _fd = os.open(_LOCK, os.O_CREAT | os.O_EXCL | os.O_RDWR)
    os.close(_fd)
except FileExistsError:
    sys.stderr.write("another sensitivity instance is running; exiting.\n")
    sys.exit(0)
import atexit

atexit.register(lambda: os.remove(_LOCK) if os.path.exists(_LOCK) else None)

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import AnomalyDataset  # noqa: E402
from src.methods.factory import build_method  # noqa: E402
from src.modules.fgsc import calibrate  # noqa: E402
from src.utils.common import ROOT, load_yaml, set_seed  # noqa: E402

OUT = os.path.join(ROOT, "exp", "sensitivity")
os.makedirs(OUT, exist_ok=True)
BASE = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
IMG, MAP = int(BASE["model"]["img_size"]), int(BASE["eval"]["map_size"])
M = BASE["method"]
DEFAULTS = dict(lam=float(M["fgsc_lambda"]), tau=float(M["fgsc_tau"]),
                kernel=int(M["fgsc_kernel"]), q=float(M["image_score_topk"]),
                shrink=float(M["shrinkage"]), blur=int(M["image_blur_k"]))
MV_ROOT = os.path.join(ROOT, BASE["data"]["mvtec_root"])
CATS = list(BASE["data"]["mvtec_categories"])

# sub-category used only for the shrinkage sweep (fit-time parameter)
SUBSET = ["bottle", "carpet", "grid", "metal_nut", "pill", "screw"]


def build(shrink: float):
    cfg = dict(BASE)
    cfg.setdefault("model", {})["backbone"] = "dinov2"
    cfg["method"] = dict(M)
    cfg["method"]["shrinkage"] = shrink
    return build_method(cfg, "A2", device="cuda",
                        override_method={"fusion": "mean", "cov_mode": "diag", "fgsc": "off"})


@torch.no_grad()
def cache_category(cat: str, shrink: float, max_n: int = 0) -> str:
    """fit once with PSC off, cache (raw, tex, mask, label) for the whole test split."""
    path = os.path.join(OUT, f"cache_{cat}_s{shrink:.2f}{'_smoke' if max_n else ''}.npz")
    if os.path.exists(path):
        print(f"  [{cat}] cached", flush=True)
        return path
    bb = build(shrink).backbone
    mean, std = bb.image_mean, bb.image_std
    m = build(shrink)
    m.set_category(cat)
    tr = AnomalyDataset(MV_ROOT, cat, "train", IMG, MAP, mean, std, only_good=True)
    m.fit(DataLoader(tr, batch_size=8, shuffle=False, num_workers=0), max_samples=0)
    m.eval()
    te = AnomalyDataset(MV_ROOT, cat, "test", IMG, MAP, mean, std)
    R, T, K, L = [], [], [], []
    seen = 0
    for b in DataLoader(te, batch_size=8, shuffle=False, num_workers=0):
        raw, tex = m.raw_score(b["image"].cuda())
        R.append(raw[:, 0].half().cpu().numpy())
        T.append(tex[:, 0].half().cpu().numpy())
        K.append(b["mask"].numpy().astype(np.uint8))
        L.append(b["label"].numpy().astype(np.int64))
        seen += b["image"].shape[0]
        if max_n and seen >= max_n:
            break
    np.savez(path, raw=np.concatenate(R), tex=np.concatenate(T),
             mask=np.concatenate(K), label=np.concatenate(L))
    print(f"  [{cat}] cached n={sum(len(x) for x in L)}", flush=True)
    del m
    torch.cuda.empty_cache()
    return path


def gblur(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return x
    ax = torch.arange(k, dtype=torch.float32, device=x.device) - (k - 1) / 2
    g = torch.exp(-(ax ** 2) / (2 * (k / 4.0) ** 2))
    g = (g / g.sum()).view(1, 1, 1, k)
    x = F.conv2d(x, g, padding=(0, k // 2))
    return F.conv2d(x, g.transpose(2, 3), padding=(k // 2, 0))


def topk_score(x: torch.Tensor, q: float) -> torch.Tensor:
    b = x.shape[0]
    flat = x.reshape(b, -1)
    k = max(int(flat.shape[1] * q), 1)
    return torch.topk(flat, k, dim=1).values.mean(dim=1)


def metrics(path: str, lam: float, tau: float, kernel: int, q: float, blur: int,
            want_ap: bool = True) -> tuple[float, float, float]:
    """Pixel/image AUROC and pixel AP of one category under one setting.

    The score maps of a whole category are concatenated before the metrics are
    computed, so the numbers match the protocol used for the paper tables
    (metrics are NOT averaged over mini-batches).
    """
    z = np.load(path)
    raw = torch.from_numpy(z["raw"]).float()
    tex = torch.from_numpy(z["tex"]).float()
    n = raw.shape[0]
    mask = z["mask"].reshape(-1) > 0.5
    lab = z["label"]
    ps, isc = [], []
    for i in range(0, n, 64):
        m = min(64, n - i)
        r = raw[i:i + m].unsqueeze(1).cuda()
        t = tex[i:i + m].unsqueeze(1).cuda()
        s = calibrate(r, t, lam=lam, tau=tau, mode="adaptive", kernel=kernel)
        s = gblur(s, blur)
        isc.append(topk_score(s, q).cpu().numpy())
        ps.append(s[:, 0].reshape(m, -1).cpu().numpy().reshape(-1))
        del r, t, s
    del raw, tex
    p = np.concatenate(ps)
    i_s = np.concatenate(isc)
    pix_auc = float(roc_auc_score(mask, p))
    img_auc = float(roc_auc_score(lab, i_s))
    ap = float(average_precision_score(mask, p)) if want_ap else float("nan")
    return img_auc, pix_auc, ap


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 categories, 64 images each")
    args = ap.parse_args()
    global CATS, SUBSET
    set_seed(int(BASE.get("seed", 42)))
    print(f"PSC defaults: {DEFAULTS}", flush=True)
    print(f"categories ({len(CATS)}): {CATS}", flush=True)
    max_n = 64 if args.smoke else 0
    if args.smoke:
        CATS = CATS[:2]
        SUBSET = [c for c in SUBSET if c in CATS] or CATS[:1]
        print("SMOKE MODE: 2 categories, <=64 test images each", flush=True)
    print("", flush=True)

    print("[stage 1] caching raw/tex for all categories (PSC off)", flush=True)
    t0 = _time.time()
    paths = {c: cache_category(c, DEFAULTS["shrink"], max_n) for c in CATS}
    print(f"stage 1 done in {(_time.time() - t0) / 60:.1f} min\n", flush=True)

    rows = []

    def run(tag: str, **kw):
        p = {**DEFAULTS, **kw}
        ia, pa, ap = [], [], []
        for c in CATS:
            a, b, d = metrics(paths[c], p["lam"], p["tau"], p["kernel"], p["q"], p["blur"])
            ia.append(a)
            pa.append(b)
            ap.append(d)
        rec = dict(group=tag, **p, image_auc=float(np.mean(ia)), pixel_auc=float(np.mean(pa)),
                   pixel_ap=float(np.mean(ap)))
        rows.append(rec)
        star = "  <-- default" if all(p[k] == DEFAULTS[k] for k in
                                     ("lam", "tau", "kernel", "q")) and tag == "PSC" else ""
        print(f"{tag:6s} lam={p['lam']:.2f} tau={p['tau']:.2f} K={p['kernel']:3d} q={p['q']:.3f} "
              f"| img {rec['image_auc']:.4f} pix {rec['pixel_auc']:.4f} AP {rec['pixel_ap']:.4f}{star}",
              flush=True)
        return rec

    print("[stage 2] sensitivity sweeps (no re-fit needed)\n", flush=True)
    run("base", lam=DEFAULTS["lam"], tau=DEFAULTS["tau"], kernel=DEFAULTS["kernel"], q=DEFAULTS["q"])
    run("off", lam=0.0)
    for v in [0.2, 0.4, 0.6, 0.8, 1.0]:
        run("a0", lam=v)
    for v in [0.0, 0.2, 0.4, 0.6, 0.8]:
        run("a1", tau=v)
    for v in [5, 9, 17, 33, 65, 129]:
        run("K", kernel=v)
    for v in [0.001, 0.005, 0.01, 0.03, 0.1]:
        run("q", q=v)
    for v in [3, 5, 11, 21]:
        run("blur", blur=v)

    print("\n[stage 3] shrinkage sweep (needs re-fit; 6-category subset)", flush=True)
    sub_paths = {}
    for s in (0.0, 0.05, DEFAULTS["shrink"], 0.2):
        for c in SUBSET:
            sub_paths[(c, s)] = cache_category(c, s, max_n)
    for s in (0.0, 0.05, DEFAULTS["shrink"], 0.2):
        ia, pa = [], []
        for c in SUBSET:
            a, b, _ = metrics(sub_paths[(c, s)], DEFAULTS["lam"], DEFAULTS["tau"],
                              DEFAULTS["kernel"], DEFAULTS["q"], DEFAULTS["blur"], want_ap=False)
            ia.append(a)
            pa.append(b)
        rec = dict(group="shrink", **{**DEFAULTS, "shrink": s}, image_auc=float(np.mean(ia)),
                   pixel_auc=float(np.mean(pa)), pixel_ap=float("nan"))
        rows.append(rec)
        print(f"shrink  s={s:.2f} | img {rec['image_auc']:.4f} pix {rec['pixel_auc']:.4f}"
              f"{'  <-- default' if s == DEFAULTS['shrink'] else ''}", flush=True)

    import csv
    keys = ["group", "lam", "tau", "kernel", "q", "blur", "shrink",
            "image_auc", "pixel_auc", "pixel_ap"]
    with open(os.path.join(OUT, "psc_sensitivity.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    print("\nwrote", os.path.join(OUT, "psc_sensitivity.csv"), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
