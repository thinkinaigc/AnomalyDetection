"""Normal-only 超参重选：回应二审报告 §1.4「MVTec 测试集调参」质疑。

做法（**全程不看任何缺陷标签，也不看测试集**）：
  1. 每类把**正常训练图**随机划出 20% 作为 validation（官方 train 全部为正常图，
     故这仍是纯正常数据，无需缺陷标签）；
  2. 在剩余 80% 上 fit，用**无标签准则**在 validation 上选超参：
     - 图像级：validation 正常图的 image_score 分数越集中（越低离散度）越好；
     - 像素级：validation 正常图上 anomaly map 的 90% 分位绝对值越小越好
       （误报越少）。两者相加构成不需要标签的标量选择分数；
  3. 网格搜索 shrinkage λ × top-q，与论文冻结值（λ=0.02, q=0.005）比较排名。

实现要点（与旧版的区别，仅为效率，数学上完全等价）：
  - 骨干冻结，故每类的 fit/val 特征各抽一次并复用（fit(feats=...) 官方支持）；
  - shrinkage 只影响参考统计量，故每个 λ 只需 refit 一次（6 次而非 24 次）；
  - top-q 只在 image_score 阶段起作用（不改变 anomaly map），故同一张异常图上
    直接对 4 个 q 重算 top-k 均值即可，无需重跑 fit/predict。
  以上把每类的前向次数从 24*(N_fit+N_val) 降到 1*(N_fit+N_val)，约 20 倍加速。

用法：
    python code/scripts/_exp_normal_only_validation.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import AnomalyDataset  # noqa: E402
from src.methods.factory import build_method  # noqa: E402
from src.modules.fgsc import image_score  # noqa: E402
from src.utils.common import ROOT, load_yaml, set_seed  # noqa: E402

BASE = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
FROZEN = dict(fusion="mean", cov_mode="diag", fgsc="lowpass", fgsc_lambda=0.6,
              fgsc_kernel=33, shrinkage=0.02, ref_smooth=0,
              image_score_topk=0.005, image_blur_k=11)

SHRINKAGES = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
TOPQS = [0.002, 0.005, 0.01, 0.02]


def make(shrinkage: float, topq: float):
    set_seed(42)
    cfg = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
    cfg.setdefault("model", {})["backbone"] = "dinov2"
    ov = dict(FROZEN)
    ov["shrinkage"] = shrinkage
    ov["image_score_topk"] = topq
    return build_method(cfg, "A2", device="cuda", override_method=ov)


def robust_spread(s: np.ndarray) -> float:
    """正常图 image_score 的稳健离散度（1.4826*MAD），越小说明正常图越不可能被判异常。"""
    s = np.asarray(s, dtype=np.float64)
    return float(1.4826 * np.median(np.abs(s - np.median(s))))


@torch.no_grad()
def extract_all(model, ds) -> tuple[dict, torch.Tensor]:
    """一次性抽完骨干特征（P3/P4/P5 [+ GLOBAL]），供后续多次 fit 复用；
    同时返回原始图像张量 [N,3,H,W]（predict 需要图像，不能只靠缓存特征）。"""
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    buf: dict[str, list] = {}
    imgs: list = []
    for b in dl:
        imgs.append(b["image"])
        fb = model.extract(b["image"].cuda())
        for k, v in fb.items():
            buf.setdefault(k, []).append(v.detach().float().cpu())
    return ({k: torch.cat(v, dim=0) for k, v in buf.items()},
            torch.cat(imgs, dim=0))


def main() -> None:
    data_root = os.path.join(ROOT, BASE["data"]["mvtec_root"])
    cats = list(BASE["data"]["mvtec_categories"])
    rng = np.random.default_rng(42)

    out = {"shrinkage_grid": SHRINKAGES, "topq_grid": TOPQS, "per_cat": {}, "frozen": FROZEN}
    rows = []

    for ci, cat in enumerate(cats, 1):
        probe = make(FROZEN["shrinkage"], FROZEN["image_score_topk"])
        bb = probe.backbone
        img_size = int(getattr(bb, "image_size", 256))
        mean = getattr(bb, "image_mean", (0.48145466, 0.4578275, 0.40821073))
        std = getattr(bb, "image_std", (0.26862954, 0.26130258, 0.27577711))
        ms = int(BASE["eval"]["map_size"])

        tr_all = AnomalyDataset(data_root, cat, "train", img_size, ms, mean, std, only_good=True)
        n = len(tr_all)
        idx = rng.permutation(n)
        n_val = max(1, int(round(0.2 * n)))
        val_idx, fit_idx = idx[:n_val], idx[n_val:]
        fit_ds = torch.utils.data.Subset(tr_all, fit_idx.tolist())
        val_ds = torch.utils.data.Subset(tr_all, val_idx.tolist())

        # ---- 特征只抽一次（骨干冻结，跨 shrinkage 复用）----
        fit_feats, _ = extract_all(probe, fit_ds)
        _, val_imgs = extract_all(probe, val_ds)
        n_val_imgs = val_imgs.shape[0]

        grid = []
        for lam in SHRINKAGES:
            m = make(lam, FROZEN["image_score_topk"])
            m.set_category(cat)
            m.fit(feats=fit_feats, max_samples=0)
            m.eval()
            # ---- 该 lambda 下，validation 的分数图只算一次 ----
            # 两张图各司其职，与旧版口径一致：
            #   raw_maps  = predict() 的 anomaly_map（校准后、图像模糊前）-> 像素项 npix
            #   blur_maps = 再经 image_blur_k 高斯模糊 -> top-q 图像项（与 predict 同链）
            raw_maps, blur_maps = [], []
            with torch.no_grad():
                for i in range(n_val_imgs):
                    amap = m.predict(val_imgs[i: i + 1].cuda())["anomaly_map"].float()
                    raw_maps.append(amap.cpu())
                    blur_maps.append(m._gaussian_blur(amap, m.image_blur_k).cpu())
            npix = float(np.quantile(torch.cat(raw_maps, 0).numpy(), 0.90))
            smaps = torch.cat(blur_maps, dim=0).numpy()
            for q in TOPQS:
                isc = image_score(torch.from_numpy(smaps), topk=q).numpy().ravel()
                grid.append({"shrinkage": lam, "topq": q,
                             "score": robust_spread(isc) + npix})

        best = min(grid, key=lambda g: g["score"])
        frozen_entry = next(g for g in grid
                            if g["shrinkage"] == FROZEN["shrinkage"]
                            and g["topq"] == FROZEN["image_score_topk"])
        rank = sorted(grid, key=lambda g: g["score"]).index(frozen_entry) + 1

        out["per_cat"][cat] = {
            "n_fit": len(fit_ds), "n_val": len(val_ds),
            "best": best, "frozen_score": frozen_entry["score"],
            "frozen_rank": rank, "n_grid": len(grid), "grid": grid,
        }
        with open(os.path.join(ROOT, "outputs",
                               "exp_normal_only_validation_partial.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        rows.append((cat, best["shrinkage"], best["topq"], frozen_entry["score"], rank))
        print(f"[{ci:2d}/{len(cats)}] {cat:12s} best lam={best['shrinkage']:<6} "
              f"q={best['topq']:<6} score={best['score']:.5f} | "
              f"frozen score={frozen_entry['score']:.5f} rank={rank}/{len(grid)}", flush=True)

    ranks = [r for *_, r in rows]
    n_best_frozen = sum(1 for _, l, q, _, _ in rows
                        if l == FROZEN["shrinkage"] and q == FROZEN["image_score_topk"])
    out["summary"] = {
        "n_cats_choosing_frozen": n_best_frozen,
        "n_cats": len(cats),
        "mean_frozen_rank": float(np.mean(ranks)),
        "median_frozen_rank": float(np.median(ranks)),
        "worst_frozen_rank": int(max(ranks)),
        "score_gap_best_vs_frozen": {
            c: float(b - f) for (c, _, _, f, _), b in
            zip(rows, [out["per_cat"][c]["best"]["score"] for c, *_ in rows])
        },
    }
    with open(os.path.join(ROOT, "outputs", "exp_normal_only_validation.json"),
              "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print(f"\n=== normal-only validation 重选汇总 (MVTec {len(cats)} 类) ===")
    print(f"选中冻结值的类别数: {n_best_frozen}/{len(cats)}")
    print(f"冻结值在 24 点网格中的平均排名: {np.mean(ranks):.1f} (最佳=1), "
          f"中位 {np.median(ranks):.0f}, 最差 {max(ranks)}")
    print("saved -> outputs/exp_normal_only_validation.json")


if __name__ == "__main__":
    main()