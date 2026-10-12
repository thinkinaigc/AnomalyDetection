# -*- coding: utf-8 -*-
"""补齐 limitation (iii) 所需的证据：两个 fit-time 参数在【完整重选】下的指标跨度。

背景（二审 + 方法学审稿意见）：
  论文声称 "shrinkage 全扫描 {0..0.20} 内 pixel AP 仅跨 0.0013"，但 A2(v1 base,无PSC)
  与 v2 base(无PSC) 的 pixel AP 差 +0.0248，二者相差 19 倍，构成矛盾。
  原因：A2 用的是 q=1%（top-1%）读出，v2 base 用 q=0.5%；而 tab:sens 的 q 行只报 image
  AUROC，声称 pixel AP "invariant at 0.5093"——但那是在 v2 base 上测的，未覆盖 A2 口径。

本实验直接测量：在冻结 v2 配置上，逐一改变 shrinkage（含 0.02 两侧）与 ref_smooth
（含 0 两侧），用**同一口径**（15 类、seed 42、pixel AP + image AUROC）报出跨度，
用于替换 limitation 中不准确的 "≤0.0013" 表述。

产出：outputs/exp_refit_span.json
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import AnomalyDataset  # noqa: E402
from src.methods.factory import build_method  # noqa: E402
from src.utils.common import ROOT, load_yaml  # noqa: E402

FROZEN = {
    "fusion": "mean", "cov_mode": "diag",
    "fgsc": "lowpass", "fgsc_lambda": 0.6, "fgsc_kernel": 33,
    "shrinkage": 0.02, "ref_smooth": 0,
    "image_score_topk": 0.005, "image_blur_k": 11, "projection_seed": 42,
}

SHRINK = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
REFSM = [0, 3, 5, 7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/mvtec_ad")
    ap.add_argument("--cats", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/exp_refit_span.json")
    args = ap.parse_args()

    device = args.device
    cfg = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
    img_size = 256
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    ms = int(cfg["eval"]["map_size"])
    data_root = os.path.join(ROOT, args.root) if not os.path.isabs(args.root) else args.root
    cats = [c.strip() for c in args.cats.split(",") if c.strip()] or list(cfg["data"]["mvtec_categories"])
    cats = [c for c in cats if os.path.isdir(os.path.join(data_root, c))]
    print(f"cats({len(cats)})={cats}", flush=True)

    # cache 每类测试 mask/labels 以加速多配置评估
    cache = {}

    def get_ds(cat):
        if cat not in cache:
            te = AnomalyDataset(data_root, cat, "test", img_size, ms, mean, std)
            cache[cat] = (te, DataLoader(te, batch_size=8, shuffle=False, num_workers=0))
        return cache[cat]

    def run(overrides):
        imgs, pixs, paps = [], [], []
        for cat in cats:
            cfg2 = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
            cfg2.setdefault("model", {})["backbone"] = "dinov2"
            ov = dict(FROZEN); ov.update(overrides)
            m = build_method(cfg2, "A2", device=device, override_method=ov)
            m.set_category(cat)
            tr = AnomalyDataset(data_root, cat, "train", img_size, ms, mean, std, only_good=True)
            m.fit(DataLoader(tr, batch_size=8, shuffle=False, num_workers=0))
            m.eval()
            te, dl = get_ds(cat)
            pm, ps, yy = [], [], []
            for b in dl:
                out = m.predict(b["image"].to(device))
                amap = out["anomaly_map"].float().cpu().numpy()[:, 0]
                isc = out["image_score"].float().cpu().numpy()
                mask = b["mask"].numpy()
                for i in range(amap.shape[0]):
                    mm, s = mask[i], amap[i]
                    if mm.shape != s.shape:
                        import cv2
                        s = cv2.resize(s, (mm.shape[1], mm.shape[0]), interpolation=cv2.INTER_LINEAR)
                    pm.append(mm.ravel()); ps.append(s.ravel()); yy.append(int(b["label"][i]))
            pm = np.concatenate(pm); ps = np.concatenate(ps); yy = np.array(yy)
            pos = pm > 0.5
            pixs.append(roc_auc_score(pos, ps)); paps.append(average_precision_score(pos, ps))
            # image AUROC 用逐图分数（重算）
            _, dl2 = get_ds(cat)
            IS, Y = [], []
            for b in dl2:
                out = m.predict(b["image"].to(device))
                IS.append(out["image_score"].float().cpu().numpy())
                Y.append(b["label"].numpy())
            IS = np.concatenate(IS); Y = np.concatenate(Y)
            if len(np.unique(Y)) > 1:
                imgs.append(roc_auc_score(Y, IS))
            del m
            torch.cuda.empty_cache()
        return {"image": float(np.mean(imgs)), "pixel": float(np.mean(pixs)),
                "pap": float(np.mean(paps))}

    out = {"shrinkage": {}, "ref_smooth": {}}
    t0 = time.time()
    for lam in SHRINK:
        out["shrinkage"][str(lam)] = run({"shrinkage": lam})
        print(f"shrinkage={lam}: {out['shrinkage'][str(lam)]}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    for k in REFSM:
        out["ref_smooth"][str(k)] = run({"ref_smooth": k})
        print(f"ref_smooth={k}: {out['ref_smooth'][str(k)]}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    for key in ("shrinkage", "ref_smooth"):
        d = out[key]
        for mname in ("image", "pixel", "pap"):
            vals = [v[mname] for v in d.values()]
            out[key + "_span_" + mname] = {"min": min(vals), "max": max(vals),
                                          "span": max(vals) - min(vals)}

    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSPANS")
    for key in ("shrinkage", "ref_smooth"):
        print(f"  {key}: image span={out[key+'_span_image']['span']:.4f} "
              f"pixelAP span={out[key+'_span_pap']['span']:.4f} "
              f"({out[key+'_span_pap']['min']:.4f}-{out[key+'_span_pap']['max']:.4f})")
    print("saved -> " + args.out)


if __name__ == "__main__":
    main()