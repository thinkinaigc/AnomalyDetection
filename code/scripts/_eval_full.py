"""通用全量评测：MVTec AD / VisA 等任意 MVTec 布局数据集。

冻结 Table I 的 DiagAD 配置（seed 42 主口径），逐类 fit + 测试，输出：
    image AUROC / pixel AUROC / pixel AP / AUPRO
并可选把逐图异常图与 mask 缓存为 .npz，供后续指标（AUPRO、PRO 曲线）离线复用，
避免重复前向。

用法：
    python code/scripts/_eval_full.py --root data/visa --tag visa
    python code/scripts/_eval_full.py --root data/mvtec_ad --tag mvtec --cache outputs/cache/mvtec
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import AnomalyDataset  # noqa: E402
from src.eval.metrics import aupro  # noqa: E402
from src.methods.factory import build_method  # noqa: E402
from src.utils.common import ROOT, load_yaml, set_seed  # noqa: E402

# Table I 冻结配置（不得改动）
FROZEN = {
    "fusion": "mean",
    "cov_mode": "diag",
    "fgsc": "lowpass",
    "fgsc_lambda": 0.6,
    "fgsc_kernel": 33,
    "shrinkage": 0.02,
    "ref_smooth": 0,
    "image_score_topk": 0.005,
    "image_blur_k": 11,
}

VISA_CATS = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
    "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]


def build_frozen_method(seed: int, device: str):
    set_seed(seed)
    cfg = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
    cfg.setdefault("model", {})["backbone"] = "dinov2"
    return build_method(cfg, "A2", device=device, override_method=dict(FROZEN))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="数据集根目录（MVTec 布局）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cats", default="", help="逗号分隔；缺省用配置里的 MVTec 15 类")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", default="", help="逐图 anomaly map 缓存目录（可选）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-train", type=int, default=0)
    args = ap.parse_args()

    data_root = os.path.join(ROOT, args.root) if not os.path.isabs(args.root) else args.root
    if args.cats:
        cats = [c.strip() for c in args.cats.split(",") if c.strip()]
    elif "visa" in os.path.basename(data_root).lower():
        cats = VISA_CATS
    else:
        cfg = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
        cats = list(cfg["data"]["mvtec_categories"])
    cats = [c for c in cats if os.path.isdir(os.path.join(data_root, c))]
    print(f"[{args.tag}] root={data_root}\n[{args.tag}] categories ({len(cats)}): {cats}", flush=True)

    m = build_frozen_method(args.seed, args.device)
    bb = m.backbone
    img_size = int(getattr(bb, "image_size", 256))
    mean = getattr(bb, "image_mean", (0.48145466, 0.4578275, 0.40821073))
    std = getattr(bb, "image_std", (0.26862954, 0.26130258, 0.27577711))
    ms = int(load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))["eval"]["map_size"])

    if args.cache:
        os.makedirs(args.cache, exist_ok=True)

    per_cat, agg = {}, {"img": [], "pix": [], "pap": [], "pro": []}
    t_start = time.time()
    for ci, cat in enumerate(cats, 1):
        t0 = time.time()
        m.set_category(cat)
        tr = AnomalyDataset(data_root, cat, "train", img_size, ms, mean, std, only_good=True)
        m.fit(DataLoader(tr, batch_size=8, shuffle=False, num_workers=0), max_samples=args.max_train)
        m.eval()

        te = AnomalyDataset(data_root, cat, "test", img_size, ms, mean, std)
        pm, ps, yy, isc = [], [], [], []
        # AUPRO 需保留空间结构：逐图存 mask/分数（不ravel）
        gt_stack, am_stack = [], []
        cache_recs = []
        for b in DataLoader(te, batch_size=8, shuffle=False, num_workers=0):
            out = m.predict(b["image"].to(args.device))
            amap = out["anomaly_map"].float().cpu().numpy()[:, 0]
            iscore = out["image_score"].float().cpu().numpy()
            mask = b["mask"].numpy()
            paths = b.get("image_path", [None] * amap.shape[0])
            for i in range(amap.shape[0]):
                mm = mask[i]
                s = amap[i]
                if mm.shape != s.shape:
                    import cv2
                    s = cv2.resize(s, (mm.shape[1], mm.shape[0]), interpolation=cv2.INTER_LINEAR)
                pm.append(mm.ravel())
                ps.append(s.ravel())
                yy.append(int(b["label"][i]))
                isc.append(float(iscore[i]))
                gt_stack.append(mm.astype(np.uint8))
                am_stack.append(s.astype(np.float16))
                cache_recs.append((paths[i], int(b["label"][i])))

        pm = np.concatenate(pm).astype(np.float32)
        ps = np.concatenate(ps).astype(np.float32)
        yy = np.array(yy)
        isc = np.array(isc, dtype=np.float64)
        n_def = int(pm.sum() > 0)

        c_img = float(roc_auc_score(yy, isc)) if len(np.unique(yy)) > 1 else float("nan")
        c_pix = float(roc_auc_score(pm, ps))
        c_pap = float(average_precision_score(pm, ps)) if n_def else float("nan")
        try:
            c_pro = float(aupro(np.stack(gt_stack), np.stack(am_stack)))
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] AUPRO failed for {cat}: {e}", flush=True)
            c_pro = float("nan")

        per_cat[cat] = dict(image_auroc=c_img, pixel_auroc=c_pix, pixel_ap=c_pap, aupro=c_pro,
                            n_test=len(yy), n_anom=int((yy == 1).sum()))
        for k, v in (("img", c_img), ("pix", c_pix), ("pap", c_pap), ("pro", c_pro)):
            if not (isinstance(v, float) and np.isnan(v)):
                agg[k].append(v)

        if args.cache:
            np.savez_compressed(
                os.path.join(args.cache, f"{args.tag}_{cat}.npz"),
                gt=np.stack(gt_stack), am=np.stack(am_stack),
                image_score=isc, image_label=yy,
            )

        print(f"[{args.tag}] {ci:2d}/{len(cats)} {cat:14s} "
              f"img={c_img:.4f} pix={c_pix:.4f} ap={c_pap:.4f} aupro={c_pro:.4f} "
              f"n={len(yy)}/{int((yy == 1).sum())} {time.time() - t0:.1f}s", flush=True)

    summary = {k: (float(np.mean(v)) if v else float("nan")) for k, v in agg.items()}
    print(f"\n=== [{args.tag}] 汇总 ({len(cats)} 类, seed {args.seed}) ===", flush=True)
    for k, label in (("img", "image AUROC"), ("pix", "pixel AUROC"),
                     ("pap", "pixel AP"), ("pro", "AUPRO")):
        print(f"{label:12s} = {summary[k]:.4f}", flush=True)
    print(f"total {time.time() - t_start:.1f}s", flush=True)

    out = dict(tag=args.tag, root=data_root, seed=args.seed, categories=cats,
               per_category=per_cat, summary=summary)
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    p = os.path.join(ROOT, "outputs", f"eval_{args.tag}_seed{args.seed}.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"saved -> {p}", flush=True)


if __name__ == "__main__":
    main()
