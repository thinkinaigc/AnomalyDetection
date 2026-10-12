"""在指定数据集/类别上运行某个方法变体并输出指标。

示例:
    python code/scripts/run_eval.py --dataset mvtec --categories bottle grid --variant A1
    python code/scripts/run_eval.py --dataset mvtec --categories bottle --variant A6 --fgsc-lambda 0.3 --fgsc-tau 0.6
    python code/scripts/run_eval.py --dataset pcb --variant A7 --no-fit
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import AnomalyDataset, list_categories  # noqa: E402
from src.eval.evaluator import evaluate  # noqa: E402
from src.methods.factory import build_method  # noqa: E402
from src.utils.common import ROOT, load_yaml, set_seed  # noqa: E402


def build_dataset(name: str, cfg: dict, category: str, split: str, img_size: int, mean, std, only_good: bool = False) -> AnomalyDataset:
    map_size = int(cfg["eval"]["map_size"])
    if name == "mvtec":
        root = os.path.join(ROOT, cfg["data"]["mvtec_root"])
    elif name == "pcb":
        root = os.path.join(ROOT, cfg["data"]["pcb_root"])
    elif name == "neu":
        root = os.path.join(ROOT, cfg["data"]["neu_root"])
    elif name == "pvel":
        root = os.path.join(ROOT, cfg["data"]["pvel_root"])
    else:
        raise ValueError(f"未知数据集: {name}")
    return AnomalyDataset(root, category, split=split, img_size=img_size, mask_size=map_size, mean=mean, std=std, only_good=only_good)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "code", "configs", "default.yaml"))
    ap.add_argument("--dataset", default="mvtec", choices=["mvtec", "pcb", "neu", "pvel"])
    ap.add_argument("--categories", nargs="*", default=None, help="默认取配置中的类别列表")
    ap.add_argument("--variant", default="A1", help="A1..A7 或 lizad/ours 等别名")
    ap.add_argument("--device", default=None)
    ap.add_argument("--backbone", default=None, help="mobileclip | dinov2（覆盖配置）")
    ap.add_argument("--dinov2-model-id", default=None, help="dinov2_vitb14_reg / dinov2_vitl14_reg（覆盖配置）")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--fit-epochs", type=int, default=None)
    ap.add_argument("--no-fit", action="store_true")
    ap.add_argument("--fgsc-lambda", type=float, default=None)
    ap.add_argument("--fgsc-tau", type=float, default=None)
    ap.add_argument("--fgsc-mode", default=None, help="off | lowpass | adaptive（覆盖配置）")
    ap.add_argument("--fusion", default=None, help="single | mean | ada（覆盖配置）")
    ap.add_argument("--cov-mode", default=None, help="diag | full（覆盖配置；full=经典 PaDiM 式全协方差）")
    ap.add_argument("--knn", action="store_true", help="使用 PatchCore 式 kNN 记忆库距离（替代马氏距离）")
    ap.add_argument("--knn-k", type=int, default=None, help="kNN 近邻数 k")
    ap.add_argument("--bank-size", type=int, default=None, help="kNN 记忆库抽样上限(0=全部)")
    ap.add_argument("--shrinkage", type=float, default=None, help="对角协方差收缩(覆盖配置)")
    ap.add_argument("--image-score-topk", type=float, default=None, help="图像级 top-k 比例 q(覆盖配置)")
    ap.add_argument("--image-blur-k", type=int, default=None, help="图像级打分前平滑核(覆盖配置)")
    ap.add_argument("--nbr-agg", type=int, default=None,
                    help="PaDiM 式邻域聚合半径 r(0=关闭)")
    ap.add_argument("--nbr-score", default=None, choices=["center", "mean"],
                    help="邻域聚合后的打分口径：center=只用中心 patch(PaDiM 原文)；mean=对偏移距离取均值")
    ap.add_argument("--ref-smooth", type=int, default=None,
                    help="正常参考均值的空域平滑核(0=关闭)")
    ap.add_argument("--fgsc-kernel", type=int, default=None, help="FGSC 低通核 K")
    ap.add_argument("--sym-avg", type=int, default=None, choices=[1, 4, 8],
                    help="测试期对称平均：1=关闭，4=四旋转，8=二面体群")
    ap.add_argument("--max-fit-samples", type=int, default=0, help="正常参考 fit 最多用多少训练图(0=全部；PCB 1866 张可抽样提速)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(args.seed or cfg.get("seed", 42))

    if args.backbone:
        cfg.setdefault("model", {})["backbone"] = args.backbone
    if args.dinov2_model_id:
        cfg.setdefault("model", {})["dinov2_model_id"] = args.dinov2_model_id

    if args.fgsc_lambda is not None:
        cfg.setdefault("method", {})["fgsc_lambda"] = args.fgsc_lambda
    if args.fgsc_tau is not None:
        cfg.setdefault("method", {})["fgsc_tau"] = args.fgsc_tau
    # CLI 覆盖：优先于 ABLATION 变体默认（用于组合最优配置，如 均值融合+FGSC）
    override_method = {}
    if args.fgsc_mode is not None:
        override_method["fgsc"] = args.fgsc_mode
    if args.fusion is not None:
        override_method["fusion"] = args.fusion
    if args.cov_mode is not None:
        override_method["cov_mode"] = args.cov_mode
    if args.knn:
        override_method["knn"] = True
    if args.knn_k is not None:
        override_method["knn_k"] = args.knn_k
    if args.bank_size is not None:
        override_method["bank_size"] = args.bank_size
    if args.shrinkage is not None:
        cfg.setdefault("method", {})["shrinkage"] = args.shrinkage
    if args.image_score_topk is not None:
        cfg.setdefault("method", {})["image_score_topk"] = args.image_score_topk
    if args.image_blur_k is not None:
        cfg.setdefault("method", {})["image_blur_k"] = args.image_blur_k
    if args.nbr_agg is not None:
        cfg.setdefault("method", {})["nbr_agg"] = args.nbr_agg
    if args.nbr_score is not None:
        cfg.setdefault("method", {})["nbr_score"] = args.nbr_score
    if args.sym_avg is not None:
        cfg.setdefault("method", {})["sym_avg"] = args.sym_avg
    if args.ref_smooth is not None:
        cfg.setdefault("method", {})["ref_smooth"] = args.ref_smooth
    if args.fgsc_kernel is not None:
        cfg.setdefault("method", {})["fgsc_kernel"] = args.fgsc_kernel
    if args.fit_epochs is not None:
        cfg.setdefault("train", {})["fit_epochs"] = args.fit_epochs

    device = args.device or cfg.get("device", "cuda")
    method = build_method(cfg, args.variant, device=device, override_method=override_method or None)
    bb = method.backbone
    img_size = int(getattr(bb, "image_size", cfg["model"]["img_size"]))
    norm_mean = getattr(bb, "image_mean", (0.48145466, 0.4578275, 0.40821073))
    norm_std = getattr(bb, "image_std", (0.26862954, 0.26130258, 0.27577711))

    out_dir = args.out or os.path.join(ROOT, "outputs", f"{args.dataset}_{args.variant}")
    os.makedirs(out_dir, exist_ok=True)

    root_key = {"mvtec": "mvtec_root", "pcb": "pcb_root", "neu": "neu_root", "pvel": "pvel_root"}[args.dataset]
    data_root = os.path.join(ROOT, cfg["data"][root_key])
    categories = args.categories or list_categories(data_root) or cfg["data"].get("mvtec_categories", [])

    rows = []
    for cat in categories:
        method.set_category(cat)
        test_ds = build_dataset(args.dataset, cfg, cat, "test", img_size, norm_mean, norm_std)
        # 零样本正常参考：train 只用正常图（only_good=True），排除 NEU 等 train 内的缺陷图
        train_ds = build_dataset(args.dataset, cfg, cat, "train", img_size, norm_mean, norm_std, only_good=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        # 零样本：用正常训练图训练轻量适配器（把正常区域异常概率压到 0）
        if not args.no_fit and int(cfg["train"]["fit_epochs"]) > 0:
            method.fit(
                train_loader,
                epochs=int(cfg["train"]["fit_epochs"]),
                lr=float(cfg["train"]["lr"]),
                max_samples=int(args.max_fit_samples),
            )
        method.eval()
        res = evaluate(
            method,
            test_ds,
            batch_size=args.batch_size or int(cfg["eval"]["batch_size"]),
            save_dir=os.path.join(out_dir, cat),
        )
        res.update({"dataset": args.dataset, "category": cat, "variant": args.variant})
        rows.append(res)
        print(json.dumps(res, ensure_ascii=False))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "summary.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {csv_path}")
    mean = df.select_dtypes(include=[np.number]).mean(numeric_only=True)
    print("均值:\n", mean.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
