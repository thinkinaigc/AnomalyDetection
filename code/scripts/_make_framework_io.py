# -*- coding: utf-8 -*-
"""生成 framework 图所需的真实输入/归一化/ours热图三元组。
挑选准则：ours 热图"最好看"优先（高亮精准落在真实缺陷上 + 背景干净），其次响应强。
输入与热图同一张图（一一对应），输入保存为完整正方形，不被裁剪。
输出覆盖 scientific_figure_output 下的 el_thumb.png / el_norm.png，并新增 ours_heat.png。
"""
import os, sys
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from torch.utils.data import DataLoader
import PIL.Image as Image

sys.path.insert(0, r"d:\cvs\pythoncode\AnomalyDetection\code")
from src.data.dataset import AnomalyDataset
from src.methods.factory import build_method
from src.utils.common import ROOT, load_yaml, set_seed

D = r"C:\Users\tiger\WorkBuddy\Research\scientific_figure_output"
BASE = load_yaml(r"d:\cvs\pythoncode\AnomalyDetection\code\configs\default.yaml")
set_seed(42)

cfg = dict(BASE)
cfg.setdefault("model", {})["backbone"] = "dinov2"
m = build_method(cfg, "A2", device="cuda",
                override_method={"fusion": "mean", "cov_mode": "diag", "fgsc": "lowpass"})
bb = m.backbone
img_size = int(getattr(bb, "image_size", 256))
mean = np.array(getattr(bb, "image_mean"), dtype=np.float32)
std = np.array(getattr(bb, "image_std"), dtype=np.float32)
ms = int(BASE["eval"]["map_size"])
mv_root = os.path.join(ROOT, BASE["data"]["mvtec_root"])

CAND = ["bottle", "capsule", "pill", "hazelnut", "metal_nut",
        "grid", "screw", "transistor", "tile", "cable"]

def resize_nearest(a, wh):
    return cv2.resize(a.astype(np.float32), wh, interpolation=cv2.INTER_NEAREST)

best = None          # (look, peak, cat, im, h28)
best_peak = None     # fallback: 仅响应最强
for cat in CAND:
    m.set_category(cat)
    tr = AnomalyDataset(mv_root, cat, "train", img_size, ms, mean, std, only_good=True)
    m.fit(DataLoader(tr, batch_size=8, shuffle=False, num_workers=0), max_samples=0)
    m.eval()
    te = AnomalyDataset(mv_root, cat, "test", img_size, ms, mean, std)
    dl = DataLoader(te, batch_size=4, shuffle=False, num_workers=0)
    with torch.no_grad():
        for b in dl:
            out = m.predict(b["image"].cuda())
            amap = out["anomaly_map"].float().cpu().numpy()[:, 0]
            img = b["image"].numpy()
            mask = b["mask"].numpy()
            lab = b["label"].numpy()
            for i in range(img.shape[0]):
                if int(lab[i]) != 1:
                    continue
                h = amap[i]
                g = mask[i]
                if g.ndim == 3:
                    g = g[0]
                gt28 = resize_nearest(g, (ms, ms)) > 0.5
                if gt28.sum() < 5:
                    continue
                hn = h / (h.max() + 1e-9)
                if gt28.sum() < 5:
                    continue
                in_mean = float(hn[gt28].mean())
                out_mean = float(hn[~gt28].mean())
                sep = in_mean - out_mean
                peak_in = float(hn[gt28].max())
                peak_out = float(hn[~gt28].max())
                peak_on_defect = peak_in >= peak_out
                gfrac = float(gt28.mean())
                peak = float(h.max())
                # 好看优先：缺陷是最热点 且 与背景明显分离(sep)；缺陷大小适中(可见但不占满)
                if peak_on_defect and sep > 0.12 and 0.01 < gfrac < 0.60:
                    size_w = 1.0 if 0.02 < gfrac < 0.45 else 0.7
                    look = sep * size_w
                    if best is None or (look, peak) > (best[0], best[1]):
                        im = (img[i].transpose(1, 2, 0) * std + mean).clip(0, 1).copy()
                        best = (look, peak, cat, im, h.copy())
                        best_meta = (gfrac, in_mean, out_mean, sep)
                if best_peak is None or peak > best_peak[1]:
                    best_peak = (cat, peak, img[i], h.copy())
    if best is not None:
        print(f"{cat}: best look so far {best[0]:.3f} peak {best[1]:.3f} ({best[2]})", flush=True)
    else:
        print(f"{cat}: no localized candidate yet", flush=True)

# 主选：最好看；若没有合格候选或最好看的其实很糟，退而求其次用响应最强的
if best is None or best[0] < 0.20:
    print("no/localized candidate poor; fallback to strongest response", flush=True)
    cat, peak, imt, h = best_peak
    im = (imt.transpose(1, 2, 0) * std + mean).clip(0, 1)
    best = (0.0, peak, cat, im.copy(), h.copy())
    best_meta = (1.0, 0.0, 0.0, 1.0)

look, peak, cat, im, h28 = best
gf, imn, otn, sep = best_meta
print("CHOSEN category:", cat, "| look(sep*w):", round(look, 3), "| peak:", round(peak, 3),
      "| gfrac:", round(gf, 3), "| in_mean:", round(imn, 3), "| out_mean:", round(otn, 3),
      "| sep:", round(sep, 3))

# el_thumb: 完整正方形未归一化输入（与热图同一张图）
thumb = (im * 255).astype(np.uint8)
Image.fromarray(thumb).save(os.path.join(D, "el_thumb.png"))
# el_norm: ImageNet 归一化后，用发散色图(0 居中)显示，使"归一化"效果明显可见
nv = np.clip((im - mean) / std, -2.0, 2.0)
s = nv.mean(axis=2)                 # 每像素归一化后均值(标量)
s = (s + 2.0) / 4.0                 # 映射到 0..1
rgb = (cm.coolwarm(s)[..., :3] * 255).astype(np.uint8)
Image.fromarray(rgb).save(os.path.join(D, "el_norm.png"))
# ours_heat: 28x28 校准异常图 -> 256x256，inferno 着色（真实 ours 输出）
h256 = cv2.resize(h28, (256, 256), interpolation=cv2.INTER_LINEAR)
rgb = (cm.inferno(h256)[:, :, :3] * 255).astype(np.uint8)
Image.fromarray(rgb).save(os.path.join(D, "ours_heat.png"))
print("saved el_thumb.png / el_norm.png / ours_heat.png", thumb.shape, "-> heat", rgb.shape)
