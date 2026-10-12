"""多随机种子评估（MVTec 15 类）：报告 mean±std，回应评审 M2（图像级对随机投影种子敏感）。

对每个 seed 重新构建方法（随机 1×1 投影随 seed 变化），逐类 fit + 测试，
汇总图像/像素 AUROC 与像素 AP 的均值和标准差。
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import AnomalyDataset, list_categories
from src.methods.factory import build_method
from src.utils.common import ROOT, load_yaml, set_seed

BASE = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
SEEDS = [42, 1, 7]
data_root = os.path.join(ROOT, BASE["data"]["mvtec_root"])
cats = list_categories(data_root)

per_seed = {}
for seed in SEEDS:
    set_seed(seed)
    cfg = dict(BASE)
    cfg.setdefault("model", {})["backbone"] = "dinov2"
    # 与头条运行保持同一校准配置：常数混合 a1=0，a0 由 FGSC_LAMBDA 覆盖
    cfg.setdefault("method", {})["fgsc_lambda"] = float(os.environ.get("FGSC_LAMBDA", "0.6"))
    m = build_method(cfg, "A2", device="cuda",
                     override_method={"fusion": "mean", "cov_mode": "diag", "fgsc": "lowpass",
                                      "fgsc_lambda": 0.6, "fgsc_kernel": 33, "shrinkage": 0.02,
                                      "ref_smooth": 0, "image_score_topk": 0.005, "image_blur_k": 11})
    bb = m.backbone; img_size = int(getattr(bb, "image_size", 256))
    mean = getattr(bb, "image_mean", (0.48145466, 0.4578275, 0.40821073))
    std = getattr(bb, "image_std", (0.26862954, 0.26130258, 0.27577711))
    ms = int(cfg["eval"]["map_size"])
    ia, pa_, pix = [], [], []
    per_cat = {}
    for cat in cats:
        m.set_category(cat)
        tr = AnomalyDataset(data_root, cat, "train", img_size, ms, mean, std, only_good=True)
        m.fit(DataLoader(tr, batch_size=8, shuffle=False, num_workers=0), max_samples=0)
        m.eval()
        te = AnomalyDataset(data_root, cat, "test", img_size, ms, mean, std)
        pm, ps, yy, isc = [], [], [], []
        for b in DataLoader(te, batch_size=8, shuffle=False, num_workers=0):
            out = m.predict(b["image"].cuda())
            amap = out["anomaly_map"].float().cpu().numpy()[:, 0]
            iscore = out["image_score"].float().cpu().numpy()
            mask = b["mask"].numpy()
            for i in range(amap.shape[0]):
                mm = mask[i]; s = amap[i]
                if mm.shape != s.shape:
                    import cv2; s = cv2.resize(s, (mm.shape[1], mm.shape[0]), interpolation=cv2.INTER_LINEAR)
                pm.append(mm.ravel()); ps.append(s.ravel())
                yy.append(int(b["label"][i])); isc.append(float(iscore[i]))
        pm = np.concatenate(pm).astype(np.float32); ps = np.concatenate(ps)
        yy = np.array(yy); isc = np.array(isc)
        c_img = float(roc_auc_score(yy, isc)) if len(np.unique(yy)) > 1 else float("nan")
        c_pix = float(roc_auc_score(pm, ps))
        c_pap = float(average_precision_score(pm, ps)) if pm.sum() > 0 else 0.0
        if len(np.unique(yy)) > 1:
            ia.append(c_img)
        pix.append(c_pix); pa_.append(c_pap)
        per_cat[cat] = dict(img=c_img, pix=c_pix, pap=c_pap)
    per_seed[seed] = dict(img=float(np.mean(ia)), pix=float(np.mean(pix)), pap=float(np.mean(pa_)), per_cat=per_cat)
    print(f"seed {seed}: img={per_seed[seed]['img']:.4f} pix={per_seed[seed]['pix']:.4f} pap={per_seed[seed]['pap']:.4f}", flush=True)

print("\n=== 多 seed 汇总 (MVTec 15 类) ===")
for k in ["img", "pix", "pap"]:
    vals = [per_seed[s][k] for s in SEEDS]
    print(f"{k}: mean={np.mean(vals):.4f} std={np.std(vals):.4f}  vals={['%.4f'%v for v in vals]}")

# write per-category std back into data.json so _restyle_figs.py can draw error bars
import json as _json
_dj_path = os.path.join(ROOT, "paper", "figures", "data.json")
_dj = _json.load(open(_dj_path))
for c in cats:
    if c in _dj["mvtec"]:
        _dj["mvtec"][c]["img_std"] = float(np.std([per_seed[s]["per_cat"][c]["img"] for s in SEEDS]))
        _dj["mvtec"][c]["pix_std"] = float(np.std([per_seed[s]["per_cat"][c]["pix"] for s in SEEDS]))
_json.dump(_dj, open(_dj_path, "w"), indent=2)
print("wrote per-category img_std/pix_std to data.json", flush=True)