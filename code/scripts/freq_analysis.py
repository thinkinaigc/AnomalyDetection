"""频域分析：为 FGSC（纹理引导频域低通）提供经验证据。

关键事实（实测）：DINOv2 异常分数图由粗 patch 网格(28²/16²)上采样而来，本身已平滑、
几乎无高频斑噪；因此 FGSC 的机制不是“去除高频噪声”，而是**纹理自适应平滑**：
在纹理响应 T 大的区域降低低通强度 α（α=a0+a1·T），更强地压低该区域的异常分数，
从而抑制“纹理背景被误判为异常”的假阳性峰值 -> 提升像素 AP。

证据链：
  1) 正常/背景区域的平均异常分数随纹理 T 升高而升高（纹理背景更易被误判）-> 假警报来源；
  2) FGSC 对背景分数的压低量 red_bg 随 T 增大而增大 -> “纹理引导”成立；
  3) 残差图(raw−cal)集中在纹理区 -> 自适应平滑局部化于纹理。

输出：freq_summary.txt（相关性 + 纹理 tertile 拆分）+ vis_*.png（原始/校准图、FFT谱、残差）。
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.dataset import AnomalyDataset, list_categories
from src.methods.factory import build_method
from src.utils.common import ROOT, load_yaml

CFG = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
DEVICE = "cuda"


def main():
    out_dir = os.path.join(ROOT, "outputs", "freq_analysis")
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(CFG)
    cfg["method"] = dict(cfg.get("method", {}))
    cfg["method"]["fusion"] = "mean"
    cfg["method"]["fgsc"] = "lowpass"  # ours
    cfg["method"]["fgsc_lambda"] = 0.8
    cfg["method"]["fgsc_tau"] = 0.4
    method = build_method(cfg, "A2", device=DEVICE,
                         override_method={"fusion": "mean", "fgsc": "lowpass"})
    assert method.fgsc_mode == "lowpass", f"FGSC 未生效: {method.fgsc_mode}"

    data_root = os.path.join(ROOT, cfg["data"]["mvtec_root"])
    categories = list_categories(data_root)
    bb = method.backbone
    img_size = int(getattr(bb, "image_size", 256))
    mean = getattr(bb, "image_mean", (0.48145466, 0.4578275, 0.40821073))
    std = getattr(bb, "image_std", (0.26862954, 0.26130258, 0.27577711))
    map_size = int(cfg["eval"]["map_size"])

    records = []
    vis_cats = {"carpet", "bottle", "leather", "wood", "grid", "hazelnut"}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        HAVE_PLOT = True
    except Exception:
        HAVE_PLOT = False

    n_vis = 0
    for cat in categories:
        method.set_category(cat)
        train_ds = AnomalyDataset(data_root, cat, "train", img_size, map_size, mean, std)
        method.fit(DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=0))
        method.eval()
        test_ds = AnomalyDataset(data_root, cat, "test", img_size, map_size, mean, std)
        loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)
        for batch in loader:
            img = batch["image"].to(DEVICE)
            mask = batch["mask"].numpy()                      # [B,H,W]
            gt = (mask > 0.5).astype(np.float32)              # [B,H,W], 1=缺陷
            out = method.predict(img)
            raw = out["raw_map"].detach().cpu().numpy()
            cal = out["anomaly_map"].detach().cpu().numpy()
            tex = out["texture"].detach().cpu().numpy()
            raw = raw[:, 0] if raw.ndim == 4 else raw
            cal = cal[:, 0] if cal.ndim == 4 else cal
            tex = tex[:, 0] if tex.ndim == 4 else tex
            B = img.shape[0]
            for i in range(B):
                g = gt[i]                                     # [H,W]
                norm = g == 0                                 # 正常/背景
                if norm.sum() == 0:
                    continue
                T = float(tex[i].mean())
                rn = raw[i][norm]
                cn = cal[i][norm]
                k = max(int(rn.size * 0.05), 1)
                # 正常区高分尾（top-5%）与峰值：平滑主要压低这些“假阳性尖峰”
                tail_raw = float(np.sort(rn)[-k:].mean())
                tail_cal = float(np.sort(cn)[-k:].mean())
                max_raw = float(rn.max())
                max_cal = float(cn.max())
                # 同一张图内：高纹理像素(T>0.5) vs 低纹理像素(T<=0.5) 的 FGSC 分数降幅
                hi = (tex[i] > 0.5) & norm
                lo = (tex[i] <= 0.5) & norm
                rmap = raw[i] - cal[i]
                red_hi = float(rmap[hi].mean()) if hi.sum() > 0 else 0.0
                red_lo = float(rmap[lo].mean()) if lo.sum() > 0 else 0.0
                records.append(dict(cat=cat, T=T,
                                    tail_raw=tail_raw, tail_cal=tail_cal,
                                    red_tail=tail_raw - tail_cal,
                                    max_raw=max_raw, max_cal=max_cal,
                                    red_max=max_raw - max_cal,
                                    red_hi=red_hi, red_lo=red_lo))
                if HAVE_PLOT and cat in vis_cats and n_vis < 30 and np.random.rand() < 0.12:
                    try:
                        _save_vis(out_dir, cat, raw[i], cal[i], tex[i], g, n_vis)
                        n_vis += 1
                    except Exception as e:
                        print("vis skip:", e)

    R = np.array([(r["T"], r["tail_raw"], r["tail_cal"], r["red_tail"],
                   r["max_raw"], r["max_cal"], r["red_max"],
                   r["red_hi"], r["red_lo"]) for r in records], dtype=float)
    T, tail_raw, tail_cal, red_tail, max_raw, max_cal, red_max, red_hi, red_lo = R.T

    def pearson(a, b):
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    q = np.quantile(T, [1 / 3, 2 / 3])
    tert = np.digitize(T, q)
    frac_hi_gt_lo = float((red_hi > red_lo).mean())
    L = []
    L.append("=== 频域分析汇总（FGSC 纹理自适应压低背景假阳性尖峰）===")
    L.append(f"样本数(测试图,含正常区) = {len(records)}")
    L.append(f"整体: 正常区高分尾原始={tail_raw.mean():.4f} 校准={tail_cal.mean():.4f} "
             f"降幅={red_tail.mean():.4f} | 峰值降幅={red_max.mean():.4f}")
    L.append(f"同图纹理分层: 高纹理像素降幅 red_hi={red_hi.mean():.4f}  "
             f"低纹理像素降幅 red_lo={red_lo.mean():.4f}  | red_hi>red_lo 占比={frac_hi_gt_lo:.3f}")
    L.append(f"Pearson(T, 高分尾降幅 red_tail) = {pearson(T, red_tail):.3f}  (纹理越强, FGSC 压得略多)")
    L.append("--- 按纹理 T tertile 拆分 (低/中/高) ---")
    for k, name in enumerate(["low", "mid", "high"]):
        m = tert == k
        L.append(f"  {name}(n={m.sum()}): T={T[m].mean():.3f} | "
                 f"tailRaw={tail_raw[m].mean():.4f}->tailCal={tail_cal[m].mean():.4f} | "
                 f"red_tail={red_tail[m].mean():.4f} | red_hi={red_hi[m].mean():.4f} red_lo={red_lo[m].mean():.4f}")
    summary = "\n".join(L)
    print(summary)
    with open(os.path.join(out_dir, "freq_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    with open(os.path.join(out_dir, "freq_records.json"), "w", encoding="utf-8") as f:
        json.dump(records, f)
    print(f"\n已保存: {out_dir}  (vis={n_vis})")


def _save_vis(out_dir, cat, raw, cal, tex, gt, idx):
    import matplotlib.pyplot as plt
    raw = np.asarray(raw); cal = np.asarray(cal); tex = np.asarray(tex); gt = np.asarray(gt)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    spec_r = np.log(np.abs(np.fft.fftshift(np.fft.fft2(raw))) + 1e-6)
    spec_c = np.log(np.abs(np.fft.fftshift(np.fft.fft2(cal))) + 1e-6)
    for ax, data, title, cmap in [
        (axes[0, 0], raw, "raw anomaly", "magma"),
        (axes[0, 1], cal, "calibrated (FGSC)", "magma"),
        (axes[0, 2], tex, "texture T", "magma"),
        (axes[1, 0], spec_r, "raw FFT(log)", "inferno"),
        (axes[1, 1], spec_c, "cal FFT(log)", "inferno"),
        (axes[1, 2], (raw - cal), "raw-cal residual", "coolwarm"),
    ]:
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"vis_{cat}_{idx}.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
