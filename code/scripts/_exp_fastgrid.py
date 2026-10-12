# -*- coding: utf-8 -*-
"""特征缓存式快速消融：每个类别只做一次 backbone 前向，之后所有配置只重算统计量与打分。

口径保证（与 run_eval.py 完全一致）
  * fit 通过新增的 feats= 参数走同一段统计量代码；
  * 打分通过替换 method.extract 返回缓存特征，继续走原始 predict()；
  * mask/label 直接由 collect_samples + 与 AnomalyDataset 相同的预处理读出。
因此 variants 里的 base 配置必须复现 outputs/mvtec15_a06/summary.csv 的数值
（DINOv2 骨干 + 对角协方差 + FGSC 固定低通 a0=0.6，即当前生产配置），脚本结束时会自动
做这项校验并打印逐类差异。

用法：
    python code/scripts/_exp_fastgrid.py --set A                # 默认 DINOv2 生产口径
    python code/scripts/_exp_fastgrid.py --cats grid,screw
    python code/scripts/_exp_fastgrid.py --set D --backbone mobileclip
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import AnomalyDataset, collect_samples  # noqa: E402
from src.eval.evaluator import evaluate  # noqa: E402
from src.methods.factory import build_method  # noqa: E402
from src.utils.common import ROOT, load_yaml, set_seed  # noqa: E402

PROD = {"fgsc": "lowpass", "fgsc_lambda": 0.6}  # A2 + FGSC 固定低通 a0=0.6（论文生产配置）

# 可被变体覆盖的实例属性（每次变体前先整体复位，避免相互污染）
ATTRS = (
    "fusion", "single_scale", "fgsc_mode", "lam", "tau", "fgsc_kernel", "tex_win", "use_dpa",
    "topk", "shrinkage", "ref_smooth", "nbr_agg", "nbr_score", "cov_mode", "use_knn",
    "knn_k", "bank_size", "image_blur_k", "image_area_t", "sym_avg", "nbr_ref_smooth",
)

SETS = {
    # A: 邻域聚合 + 对称平均（正交于 a0/K，两项都是"稳健性"方向）
    "A": [
        ("base", {}),
        ("nbr1", {"nbr_agg": 1}),
        ("nbr2", {"nbr_agg": 2}),
        ("nbr3", {"nbr_agg": 3}),
        ("sym4", {"sym_avg": 4}),
        ("sym8", {"sym_avg": 8}),
        ("nbr1+sym4", {"nbr_agg": 1, "sym_avg": 4}),
        ("nbr1+sym8", {"nbr_agg": 1, "sym_avg": 8}),
    ],
    # B: 参考统计口径（收缩、参考图平滑、协方差模式）
    "B": [
        ("base", {}),
        ("shrink0.02", {"shrinkage": 0.02}),
        ("shrink0.05", {"shrinkage": 0.05}),
        ("shrink0.20", {"shrinkage": 0.20}),
        ("shrink0.05+nbr1", {"shrinkage": 0.05, "nbr_agg": 1}),
        ("ref0", {"ref_smooth": 0}),
        ("ref3", {"ref_smooth": 3}),
        ("ref11", {"ref_smooth": 11}),
    ],
    # C: 打分口径（top-k 比例、平滑核、FGSC 强度）
    "C": [
        ("base", {}),
        ("q0.003", {"topk": 0.003}),
        ("q0.005", {"topk": 0.005}),
        ("q0.02", {"topk": 0.02}),
        ("q0.05", {"topk": 0.05}),
        ("blur5", {"image_blur_k": 5}),
        ("blur17", {"image_blur_k": 17}),
        ("a0.5", {"lam": 0.5}),
        ("a0.7", {"lam": 0.7}),
        ("K49", {"fgsc_kernel": 49}),
        ("K25", {"fgsc_kernel": 25}),
    ],
    # E: 尺度/融合口径 + 面积线索图像打分
    "E": [
        ("base", {}),
        ("singleP3", {"fusion": "single", "single_scale": "P3"}),
        ("singleP4", {"fusion": "single", "single_scale": "P4"}),
        ("ada", {"fusion": "ada"}),
        ("mean+nbr1", {"nbr_agg": 1}),
        ("ada+nbr1", {"fusion": "ada", "nbr_agg": 1}),
        ("area0.3", {"image_area_t": 0.3}),
        ("area0.4", {"image_area_t": 0.4}),
        ("area0.5", {"image_area_t": 0.5}),
    ],
    # F: 邻域聚合的细化（打分口径 / 与收缩、平滑核、q 的交互）
    "F": [
        ("base", {}),
        ("nbr1mean", {"nbr_agg": 1, "nbr_score": "mean"}),
        ("nbr2mean", {"nbr_agg": 2, "nbr_score": "mean"}),
        ("nbr1+refs3", {"nbr_agg": 1, "nbr_ref_smooth": 3}),
        ("nbr1+refs7", {"nbr_agg": 1, "nbr_ref_smooth": 7}),
        ("nbr1+sh0.05", {"nbr_agg": 1, "shrinkage": 0.05}),
        ("nbr1+sh0.02", {"nbr_agg": 1, "shrinkage": 0.02}),
        ("nbr1+K25", {"nbr_agg": 1, "fgsc_kernel": 25}),
        ("nbr1+K49", {"nbr_agg": 1, "fgsc_kernel": 49}),
        ("nbr1+q0.005", {"nbr_agg": 1, "topk": 0.005}),
        ("nbr1+blur17", {"nbr_agg": 1, "image_blur_k": 17}),
    ],
    # D: 协方差模式对照（full 为经典 PaDiM，diag 为当前生产口径）
    "D": [
        ("base", {}),
        ("nbr1", {"nbr_agg": 1}),
        ("sym8", {"sym_avg": 8}),
        ("full", {"cov_mode": "full"}),
        ("full+nbr1", {"cov_mode": "full", "nbr_agg": 1}),
        ("diag_nbr1mean", {"cov_mode": "diag", "nbr_agg": 1, "nbr_score": "mean"}),
    ],
    # G: 组合已验证的单项改进（ref_smooth 与 shrinkage 单独都已帕累托改善）
    "G": [
        ("base", {}),
        ("sh0.02", {"shrinkage": 0.02}),
        ("ref0", {"ref_smooth": 0}),
        ("ref0+sh0.02", {"ref_smooth": 0, "shrinkage": 0.02}),
        ("ref0+sh0.05", {"ref_smooth": 0, "shrinkage": 0.05}),
        ("ref3+sh0.02", {"ref_smooth": 3, "shrinkage": 0.02}),
        ("ref0+sh0.02+nbr1", {"ref_smooth": 0, "shrinkage": 0.02, "nbr_agg": 1}),
        ("ref0+sh0.02+K41", {"ref_smooth": 0, "shrinkage": 0.02, "fgsc_kernel": 41}),
        ("ref0+sh0.02+K49", {"ref_smooth": 0, "shrinkage": 0.02, "fgsc_kernel": 49}),
        ("ref0+sh0.02+a0.5", {"ref_smooth": 0, "shrinkage": 0.02, "lam": 0.5}),
        ("ref0+sh0.02+q0.005", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005}),
        ("ref0+sh0.02+q0.02", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.02}),
    ],
    # H: 在 G 的全支配组合(ref0+sh0.02+q0.005)附近做细化
    "H": [
        ("base", {}),
        ("best", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005}),
        ("best+blur5", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "image_blur_k": 5}),
        ("best+blur7", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "image_blur_k": 7}),
        ("best+a0.55", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "lam": 0.55}),
        ("best+a0.5", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "lam": 0.5}),
        ("best+a0.5+blur5", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005,
                              "lam": 0.5, "image_blur_k": 5}),
        ("best+K37", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "fgsc_kernel": 37}),
        ("best+K41", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "fgsc_kernel": 41}),
        ("best+nbr1mean", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005,
                            "nbr_agg": 1, "nbr_score": "mean"}),
        ("best+nbr1mean+blur5", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005,
                                  "nbr_agg": 1, "nbr_score": "mean", "image_blur_k": 5}),
        ("best+sh0.01", {"ref_smooth": 0, "shrinkage": 0.01, "topk": 0.005}),
        ("best+sh0.0", {"ref_smooth": 0, "shrinkage": 0.0, "topk": 0.005}),
        ("best+q0.003", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.003}),
    ],
    # I: DINOv2 口径的消融梯子（与新主结果同骨干，供论文消融表使用）
    "I": [
        ("A1_singleP3_noFGSC", {"fusion": "single", "single_scale": "P3", "fgsc_mode": "off"}),
        ("A2_mean_noFGSC", {"fgsc_mode": "off"}),
        ("A2_mean_FGSC_lowpass", {}),
        ("v2", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005}),
        ("v2+adaptive", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005,
                          "fgsc_mode": "adaptive", "tau": 0.4}),
        ("v2+adaptive+dpa", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005,
                              "fgsc_mode": "adaptive", "tau": 0.4, "use_dpa": True}),
        ("v2+ada", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "fusion": "ada"}),
        # 同骨干核对论文表里的基线行：PaDiM(全协方差) / kNN 记忆库 / A1 / A2(无 PSC)
        ("A2_noPSC_fullcov", {"fgsc_mode": "off", "cov_mode": "full"}),
        ("A2_noPSC_knn", {"fgsc_mode": "off", "use_knn": True, "knn_k": 3}),
        ("A1_noPSC_fullcov", {"fusion": "single", "single_scale": "P3",
                               "fgsc_mode": "off", "cov_mode": "full"}),
    ],
    # J: DINOv2+v2 口径的 PSC 增益分解(a) 与超参敏感性(b)，替换论文里不一致的老数值。
    #    每项 = v2 基(ref_smooth0/shrinkage0.02/topk0.005) + 被扫参数；PROD 提供固定低通 a0=0.6/K=33。
    "J": [
        # (a) PSC 增益分解：无校准 / 固定低通 a0=0.6 / 纹理门控 a1=0.4
        ("fgsc_none", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "fgsc_mode": "off"}),
        ("fgsc_lowpass", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005,
                          "fgsc_mode": "lowpass", "lam": 0.6}),
        ("fgsc_gated", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005,
                        "fgsc_mode": "adaptive", "tau": 0.4}),
        # (b) 敏感性：a0 / a1 / K / q / blur / shrinkage
        ("a0_0.2", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "lam": 0.2}),
        ("a0_0.4", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "lam": 0.4}),
        ("a0_0.6", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "lam": 0.6}),
        ("a0_0.8", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "lam": 0.8}),
        ("a0_1.0", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "lam": 1.0}),
        ("a1_0.0", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "fgsc_mode": "adaptive", "tau": 0.0}),
        ("a1_0.2", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "fgsc_mode": "adaptive", "tau": 0.2}),
        ("a1_0.4", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "fgsc_mode": "adaptive", "tau": 0.4}),
        ("a1_0.6", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "fgsc_mode": "adaptive", "tau": 0.6}),
        ("a1_0.8", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "fgsc_mode": "adaptive", "tau": 0.8}),
        ("K_9", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "fgsc_kernel": 9}),
        ("K_17", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "fgsc_kernel": 17}),
        ("K_33", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "fgsc_kernel": 33}),
        ("K_49", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "fgsc_kernel": 49}),
        ("K_81", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "fgsc_kernel": 81}),
        ("K_129", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "fgsc_kernel": 129}),
        ("q_0.001", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.001}),
        ("q_0.005", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005}),
        ("q_0.01", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.01}),
        ("q_0.05", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.05}),
        ("q_0.1", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.1}),
        ("blur_3", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "image_blur_k": 3}),
        ("blur_7", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "image_blur_k": 7}),
        ("blur_11", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "image_blur_k": 11}),
        ("blur_15", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "image_blur_k": 15}),
        ("blur_21", {"ref_smooth": 0, "shrinkage": 0.02, "topk": 0.005, "image_blur_k": 21}),
        ("sh_0.0", {"ref_smooth": 0, "shrinkage": 0.0, "topk": 0.005}),
        ("sh_0.05", {"ref_smooth": 0, "shrinkage": 0.05, "topk": 0.005}),
        ("sh_0.1", {"ref_smooth": 0, "shrinkage": 0.1, "topk": 0.005}),
        ("sh_0.2", {"ref_smooth": 0, "shrinkage": 0.2, "topk": 0.005}),
    ],
}


class MaskDS(Dataset):
    """只提供 mask/label 的轻量数据集；image 是占位张量（骨干前向已被缓存特征替代）。"""

    def __init__(self, masks, labels, img_size: int, mask_size: int):
        self.masks = masks
        self.labels = labels
        self.img_size = img_size
        self.mask_size = mask_size

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return {"image": torch.zeros(3, self.img_size, self.img_size),
                "mask": self.masks[i], "label": self.labels[i]}


class FeatSrc:
    """按 batch 顺序吐出缓存特征，调用签名与 backbone.forward 一致。"""

    def __init__(self, feats: dict):
        self.f = feats
        self.i = 0

    def __call__(self, img: torch.Tensor) -> dict:
        n = img.shape[0]
        out = {k: v[self.i:self.i + n] for k, v in self.f.items()}
        self.i += n
        return out


def read_masks(root: str, cat: str, mask_size: int):
    """与 AnomalyDataset 完全一致的 mask 预处理（阈值 127 -> 最近邻缩放 -> 0.5 二值化）。"""
    samples = collect_samples(os.path.join(root, cat), cat, "test", only_good=False)
    masks, labels = [], []
    for s in samples:
        if s.mask_path:
            m = cv2.imread(s.mask_path, cv2.IMREAD_GRAYSCALE)
            m = (m > 127).astype(np.float32)
            m = cv2.resize(m, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST)
            m = (m > 0.5).astype(np.float32)
        else:
            m = np.zeros((mask_size, mask_size), np.float32)
        masks.append(torch.from_numpy(m))
        labels.append(s.label)
    return masks, labels


@torch.no_grad()
def extract_feats(method, loader: DataLoader, to_device: bool = False) -> dict:
    """一次前向抽完特征；to_device=True 时直接留在 GPU 上（全协方差统计量在 GPU 上快得多）。"""
    dev = method.device
    buf: dict[str, list] = {}
    for batch in loader:
        f = method.extract(batch["image"].to(dev))
        for k, v in f.items():
            buf.setdefault(k, []).append(v.detach().float())
    return {k: (torch.cat(v, 0).to(dev) if to_device else torch.cat(v, 0).cpu())
            for k, v in buf.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="A", choices=sorted(SETS))
    ap.add_argument("--cats", default=None, help="逗号分隔；默认取配置里的 15 类")
    ap.add_argument("--dataset", default="mvtec")
    ap.add_argument("--backbone", default="dinov2", choices=["dinov2", "mobileclip"])
    ap.add_argument("--seed", type=int, default=None,
                    help="覆盖 seed；DINOv2 的随机固定投影依赖 RNG，故不同 seed 即不同投影")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
    cfg.setdefault("model", {})["backbone"] = args.backbone
    set_seed(int(args.seed if args.seed is not None else cfg.get("seed", 42)))
    device = cfg.get("device", "cuda")

    root_key = {"mvtec": "mvtec_root", "pcb": "pcb_root", "neu": "neu_root", "pvel": "pvel_root"}[args.dataset]
    data_root = os.path.join(ROOT, cfg["data"][root_key])
    cats = args.cats.split(",") if args.cats else list(cfg["data"]["mvtec_categories"])
    variants = SETS[args.set]

    method = build_method(cfg, "A2", device=device, override_method=dict(PROD))
    bb = method.backbone
    img_size = int(getattr(bb, "image_size", cfg["model"]["img_size"]))
    mean = getattr(bb, "image_mean", (0.48145466, 0.4578275, 0.40821073))
    std = getattr(bb, "image_std", (0.26862954, 0.26130258, 0.27577711))
    map_size = int(cfg["eval"]["map_size"])
    bs_test = int(cfg["eval"]["batch_size"])
    bs_train = int(cfg["train"]["batch_size"])
    snap = {a: getattr(method, a) for a in ATTRS}

    res: dict[str, list] = {name: [] for name, _ in variants}
    t_start = time.time()
    real_extract = method.extract
    for ci, cat in enumerate(cats):
        method.set_category(cat)
        method.extract = real_extract  # 复位可能被替换的 extract
        train_ds = AnomalyDataset(data_root, cat, split="train", img_size=img_size,
                                mask_size=map_size, mean=mean, std=std, only_good=True)
        test_ds = AnomalyDataset(data_root, cat, split="test", img_size=img_size,
                                 mask_size=map_size, mean=mean, std=std)
        tr_loader = DataLoader(train_ds, batch_size=bs_train, shuffle=False, num_workers=0)
        te_loader = DataLoader(test_ds, batch_size=bs_test, shuffle=False, num_workers=0)
        # 统计量严格在 CPU 上算（与 run_eval.py 的 feats.float().cpu() 完全一致），
        # 打分在 GPU 上算（与 run_eval.py 一致）。生产口径是对角协方差，CPU 上也很轻。
        tr_feats = extract_feats(method, tr_loader, to_device=False)
        te_feats = extract_feats(method, te_loader, to_device=True)
        masks, labels = read_masks(data_root, cat, map_size)
        ds = MaskDS(masks, labels, img_size, map_size)
        assert len(labels) == te_feats["P3"].shape[0], (cat, len(labels), te_feats["P3"].shape)

        for name, over in variants:
            for a in ATTRS:
                setattr(method, a, snap[a])
            for k, v in over.items():
                setattr(method, k, v)
            method.fit(feats=tr_feats, keep_device=False)
            method.eval()
            method.extract = FeatSrc(te_feats)
            r = evaluate(method, ds, batch_size=bs_test, save_dir=None)
            r.pop("latency_ms", None)
            r.pop("peak_mem_mb", None)
            res[name].append(r)
            print("[%2d/%2d %-9s] %-14s img %.4f  pix %.4f  AP %.4f  (%.1f min)"
                  % (ci + 1, len(cats), cat, name, r["image_auroc"], r["pixel_auroc"],
                     r["pixel_ap"], (time.time() - t_start) / 60), flush=True)
        method.extract = real_extract
        del tr_feats, te_feats, tr_loader, te_loader, train_ds, test_ds, ds
        torch.cuda.empty_cache()

    keys = ("image_auroc", "image_ap", "pixel_auroc", "pixel_ap", "fpr_at_95tpr", "tiny_defect_ap")
    print("\n" + "=" * 92)
    print("MACRO MEAN over %d categories  (%s)" % (len(cats), ",".join(cats)))
    print("=" * 92)
    print("%-16s" % "variant" + "".join("%11s" % k.replace("_", " ") for k in keys))
    table = {}
    for name, _ in variants:
        row = {k: float(np.mean([r[k] for r in res[name]])) for k in keys}
        table[name] = row
        print("%-16s" % name + "".join("%11.4f" % row[k] for k in keys), flush=True)
    base = table.get("base") or table.get("full_base") or table[variants[0][0]]
    print("\ndelta vs %s" % (variants[0][0],))
    for name, _ in variants[1:]:
        r = table[name]
        print("%-16s" % name + "".join("%+11.4f" % (r[k] - base[k]) for k in keys), flush=True)
    # 与正式流程的基线逐类对照，确认本框架口径一致（base 应复现 outputs/mvtec15_a06）
    ref_name = "mvtec15_a06" if args.backbone == "dinov2" else "mvtec15_A2fgscL"
    ref_csv = os.path.join(ROOT, "outputs", ref_name, "summary.csv")
    base_name = "base" if "base" in table else variants[0][0]
    if os.path.exists(ref_csv):
        import pandas as pd
        ref = pd.read_csv(ref_csv).set_index("category")
        got = pd.DataFrame(res[base_name], index=list(cats))
        both = [c for c in cats if c in ref.index]
        print("\n口径对照 %s vs outputs/%s (%d 类)" % (base_name, ref_name, len(both)))
        for k in ("image_auroc", "pixel_auroc", "pixel_ap"):
            d = float(np.abs(got.loc[both, k].astype(float).values
                              - ref.loc[both, k].astype(float).values).max())
            print("  %-14s macro ref %.6f  got %.6f  max|per-cat diff| %.2e"
                  % (k, ref.loc[both, k].astype(float).mean(),
                     got.loc[both, k].astype(float).mean(), d), flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"cats": cats, "set": args.set, "table": table, "per_cat": res},
                      f, indent=2, ensure_ascii=False)
        print("\nsaved: %s" % args.out)
    print("DONE %.1f min" % ((time.time() - t_start) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())