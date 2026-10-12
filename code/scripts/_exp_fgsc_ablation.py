"""消融：分离 FGSC 的"纹理门控"贡献 vs 纯高斯模糊。

对比三种校准（同一 MVTec 15 类、同一 fit）：
  off        : 不做校准（A2）
  blur       : 固定强度纯高斯低通（无纹理门控），核 k=33
  fgsc       : 我们的纹理门控低通 α=a0+a1·T
指标：像素 AUROC / 像素 AP / 图像 AUROC（15 类均值）。用于证明门控本身有效。
"""
from __future__ import annotations
import os, sys
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import AnomalyDataset, list_categories
from src.methods.factory import build_method
from src.utils.common import ROOT, load_yaml, set_seed

set_seed(42)
cfg = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
cfg.setdefault("model", {})["backbone"] = "dinov2"
m = build_method(cfg, "A2", device="cuda",
                 override_method={"fusion": "mean", "cov_mode": "diag", "fgsc": "lowpass"})
data_root = os.path.join(ROOT, cfg["data"]["mvtec_root"])
cats = list_categories(data_root)
bb = m.backbone; img_size = int(getattr(bb, "image_size", 256))
mean = getattr(bb, "image_mean", (0.48145466, 0.4578275, 0.40821073))
std = getattr(bb, "image_std", (0.26862954, 0.26130258, 0.27577711))
ms = int(cfg["eval"]["map_size"])

results = {v: {"pix_auc": [], "pix_ap": [], "img_auc": []} for v in ["off", "blur", "fgsc"]}
for cat in cats:
    m.set_category(cat)
    tr = AnomalyDataset(data_root, cat, "train", img_size, ms, mean, std, only_good=True)
    m.fit(DataLoader(tr, batch_size=8, shuffle=False, num_workers=0), max_samples=0)
    m.eval()
    te = AnomalyDataset(data_root, cat, "test", img_size, ms, mean, std)
    acc = {v: {"pm": [], "ps": [], "y": [], "isc": []} for v in results}
    for b in DataLoader(te, batch_size=8, shuffle=False, num_workers=0):
        out = m.predict(b["image"].cuda())
        raw = out["raw_map"]
        maps = {
            "off": raw,
            "blur": m._gaussian_blur(raw, m.fgsc_kernel),   # 纯高斯低通，无纹理门控
            "fgsc": out["anomaly_map"],                       # 纹理门控低通（我们的）
        }
        y = b["label"].numpy(); mask = b["mask"].numpy()
        for v, mp in maps.items():
            mp = mp.float().cpu().numpy()[:, 0]
            for i in range(mp.shape[0]):
                mm = mask[i]; s = mp[i]
                if mm.shape != s.shape:
                    import cv2; s = cv2.resize(s, (mm.shape[1], mm.shape[0]), interpolation=cv2.INTER_LINEAR)
                acc[v]["pm"].append(mm.ravel()); acc[v]["ps"].append(s.ravel())
                acc[v]["y"].append(int(y[i]))
                f = np.sort(s.ravel())[::-1]; kk = max(int(f.size * 0.01), 1)
                acc[v]["isc"].append(float(f[:kk].mean()))
    for v in results:
        pm = np.concatenate(acc[v]["pm"]).astype(np.float32); ps = np.concatenate(acc[v]["ps"])
        yy = np.array(acc[v]["y"]); isc = np.array(acc[v]["isc"])
        results[v]["pix_auc"].append(roc_auc_score(pm, ps))
        results[v]["pix_ap"].append(average_precision_score(pm, ps) if pm.sum() > 0 else np.nan)
        results[v]["img_auc"].append(roc_auc_score(yy, isc) if len(np.unique(yy)) > 1 else np.nan)
    print(f"[{cat}] done", flush=True)

print("\n=== FGSC 消融（15 类均值）===")
print(f"{'variant':10s} {'pixAUROC':>9s} {'pixAP':>8s} {'imgAUROC':>9s}")
for v in ["off", "blur", "fgsc"]:
    r = results[v]
    print(f"{v:10s} {np.nanmean(r['pix_auc']):9.4f} {np.nanmean(r['pix_ap']):8.4f} {np.nanmean(r['img_auc']):9.4f}")
