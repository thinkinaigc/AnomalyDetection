# -*- coding: utf-8 -*-
"""定位快速消融框架与 run_eval.py 的差异来源：同一进程内并排跑「真实前向」与「缓存特征」。"""
from __future__ import annotations
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _exp_fastgrid import FeatSrc, MaskDS, extract_feats, read_masks  # noqa: E402
from src.data.dataset import AnomalyDataset  # noqa: E402
from src.eval.evaluator import evaluate  # noqa: E402
from src.methods.factory import build_method  # noqa: E402
from src.utils.common import ROOT, load_yaml, set_seed  # noqa: E402

CAT = "bottle"
cfg = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
cfg["model"]["backbone"] = "dinov2"
set_seed(42)
OVR = {"fgsc": "lowpass", "fgsc_lambda": 0.6}
method = build_method(cfg, "A2", device="cuda", override_method=OVR)
bb = method.backbone
img_size = int(getattr(bb, "image_size", 256))
mean, std = bb.image_mean, bb.image_std
map_size = int(cfg["eval"]["map_size"])
bs = int(cfg["eval"]["batch_size"])
root = os.path.join(ROOT, cfg["data"]["mvtec_root"])
method.set_category(CAT)
train_ds = AnomalyDataset(root, CAT, split="train", img_size=img_size,
                          mask_size=map_size, mean=mean, std=std, only_good=True)
test_ds = AnomalyDataset(root, CAT, split="test", img_size=img_size,
                         mask_size=map_size, mean=mean, std=std)
tr_loader = DataLoader(train_ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=0)
te_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)

# --- A: 真实前向（等价 run_eval.py） ---
method.fit(tr_loader, epochs=0, max_samples=0)
resA = evaluate(method, test_ds, batch_size=bs, save_dir=None)
print("A real          img %.6f pix %.6f AP %.6f" % (resA["image_auroc"], resA["pixel_auroc"], resA["pixel_ap"]))

# --- 缓存特征与mask（第二个方法实例，权重/投影与第一个同种子） ---
method2 = build_method(cfg, "A2", device="cuda", override_method=OVR)
method2.set_category(CAT)
tr_f = extract_feats(method2, tr_loader, to_device=False)
te_f = extract_feats(method2, te_loader, to_device=True)
masks, labels = read_masks(root, CAT, map_size)
ds = MaskDS(masks, labels, img_size, map_size)
method2.fit(feats=tr_f, keep_device=False)
method2.eval()
method2.extract = FeatSrc(te_f)
resB = evaluate(method2, ds, batch_size=bs, save_dir=None)
print("B cached+MaskDS img %.6f pix %.6f AP %.6f" % (resB["image_auroc"], resB["pixel_auroc"], resB["pixel_ap"]))

# --- C: 缓存特征 + 真实 test_ds（真实图像、真实 mask），检验 mask 读取是否一致 ---
method2.extract = FeatSrc(te_f)
resC = evaluate(method2, test_ds, batch_size=bs, save_dir=None)
print("C cached+realDS img %.6f pix %.6f AP %.6f" % (resC["image_auroc"], resC["pixel_auroc"], resC["pixel_ap"]))

# --- D: 真实前向 + MaskDS（只换数据集，检验 mask 读取是否一致） ---
method.extract = method.extract
method.fit(tr_loader, epochs=0, max_samples=0)
resD = evaluate(method, ds, batch_size=bs, save_dir=None)
print("D real+MaskDS   img %.6f pix %.6f AP %.6f" % (resD["image_auroc"], resD["pixel_auroc"], resD["pixel_ap"]))

# --- E: 逐样本核对 mask 与标签 ---
mm = [test_ds[i]["mask"].numpy() for i in range(len(test_ds))]
lab_d = [test_ds[i]["label"] for i in range(len(test_ds))]
print("mask equal:", all(np.array_equal(a, b.numpy()) for a, b in zip(mm, masks)),
      "label equal:", lab_d == list(labels))