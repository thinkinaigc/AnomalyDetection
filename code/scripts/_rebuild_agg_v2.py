"""在最终 v2 配置的真实异常图上重做图像级聚合上限研究，写回 data.json 的 agg 段。

v2 = DINOv2-B + 对角协方差 + 均值融合 + FGSC 固定低通(a0=0.6/K=33)
     + shrinkage 0.02 + ref_smooth 0 + topq 0.005 + blur 11, seed 42。
对每类 MVTec 测试图的 v2 最终模糊图(sm)施加不同读出规则，计算图像级 AUROC，
再对 15 类取宏平均；规则集合与 _restyle_figs.py 绘制的图一致。
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
from src.data.dataset import AnomalyDataset
from src.methods.factory import build_method
from src.utils.common import set_seed, load_yaml
from src.modules.fgsc import gaussian_blur

CATS = ["bottle","cable","capsule","carpet","grid","hazelnut","leather",
        "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"]

def rules(sm: torch.Tensor):
    flat = sm.reshape(sm.shape[0], -1)
    out = {}
    out["max"] = flat.max(dim=1).values
    for p, name in [(0.01,"top1%"),(0.05,"top5%"),(0.20,"top20%")]:
        k = max(int(flat.shape[1]*p), 1)
        out[name] = torch.topk(flat, k, dim=1).values.mean(dim=1)
    out["mean"] = flat.mean(dim=1)
    out["median"] = flat.median(dim=1).values
    return out

def main():
    set_seed(42)
    cfg = load_yaml(os.path.join(ROOT,"code/configs/default.yaml"))
    cfg.setdefault("model",{})["backbone"]="dinov2"
    PROD = {"fgsc":"lowpass","fgsc_lambda":0.6}
    m = build_method(cfg, "A2", device="cuda", override_method=dict(PROD,
        shrinkage=0.02, ref_smooth=0, topk=0.005, image_blur_k=11, cov_mode="diag"))
    m.eval()
    from torch.utils.data import DataLoader
    agg_acc = {k:[] for k in ["max","top1%","top5%","top20%","mean","median"]}
    for cat in CATS:
        m.set_category(cat)
        td = AnomalyDataset(os.path.join(ROOT,"data/mvtec_ad"), cat, "train", 256, 256, only_good=True)
        m.fit(DataLoader(td, batch_size=8, shuffle=False, num_workers=0))
        ds = AnomalyDataset(os.path.join(ROOT,"data/mvtec_ad"), cat, "test", 256, 256)
        dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        per_im = {k:[] for k in agg_acc}
        labels = []
        for b in dl:
            out = m.predict(b["image"].cuda())
            sm = gaussian_blur(out["anomaly_map"], 11)
            r = rules(sm)
            for k,v in r.items(): per_im[k].append(v.detach().cpu().numpy())
            labels.append(b["label"].cpu().numpy())
        L = np.concatenate(labels)
        for k in agg_acc:
            s = np.concatenate(per_im[k])
            agg_acc[k].append(roc_auc_score(L, s))
        print("[%s] max %.4f top1%% %.4f top5%% %.4f mean %.4f" %
              (cat, agg_acc["max"][-1], agg_acc["top1%"][-1], agg_acc["top5%"][-1], agg_acc["mean"][-1]))
    agg = {k: float(np.mean(v)) for k,v in agg_acc.items()}
    print("agg (v2, 15-cat mean):", {k:round(v,4) for k,v in agg.items()})
    p = os.path.join(ROOT,"paper/figures/data.json")
    data = json.load(open(p))
    data["agg"] = agg
    json.dump(data, open(p,"w"), indent=2)
    print("updated", p)

if __name__ == "__main__":
    main()
