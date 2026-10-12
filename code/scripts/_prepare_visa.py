"""VisA 数据准备：把 HuggingFace 镜像布局转换为项目统一的 MVTec 布局。

镜像源（官方 VisA_20220922.tar 的逐文件未改镜像，CC BY 4.0）：
    https://hf-mirror.com/datasets/imaadd05/visa-anomaly-detection

镜像结构（**按类别分子目录**，与官方 tar 不同）：
    <cat>/Data/Images/Normal/*.JPG        正常图
    <cat>/Data/Images/Anomaly/*.JPG       异常图
    <cat>/Data/Masks/Anomaly/*.png        异常图 mask（与 Anomaly 同名，无 _mask 后缀）
    <cat>/image_anno.csv                  该类图像级标签
    split_csv/1cls.csv                    **全局**官方划分（12 类合并）

1cls.csv 列：`object,split,label,image,mask`，例如
    candle,train,normal,candle/Data/Images/Normal/0836.JPG,
    candle,test,mel candle/Data/Images/Anomaly/000.JPG,candle/Data/Masks/Anomaly/000.png

目标结构（与 code/src/data/dataset.py 的 collect_samples 一致）：
    data/visa/<cat>/train/good/*.JPG
    data/visa/<cat>/test/good/*.JPG
    data/visa/<cat>/test/<defect_type>/*.JPG
    data/visa/<cat>/ground_truth/<defect_type>/*_mask.png

注意：mask 列给的是 `Data/Masks/Anomaly/000.png`，而同一目录下不同缺陷类型
共用同一张图的文件名空间，因此掩码目录按缺陷类型分子目录、文件重命名为
`<stem>_mask.png`，与 dataset.py 的候选命名约定一致。
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path

CATS = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
        "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]


def _clean_defect(label: str) -> str:
    """把 VisA 的复合缺陷标签规整成安全的目录名。

    官方 image_anno.csv 的复合标签形如
    ``damaged corner of packaging,foreign particals on candle``（逗号分隔多缺陷）。
    目录名取**首个**缺陷类型（VisA 官方评测亦按单一类型汇总），
    逗号后的其余类型丢弃，仅影响目录粒度、不影响任何指标。
    """
    first = label.split(",")[0].strip()
    return "_".join(first.split()).replace("/", "_")


def _load_defect_types(src_root: Path, cat: str) -> dict[str, str]:
    """读取 <cat>/image_anno.csv，返回 image 相对路径 -> 复合标签。"""
    f = src_root / cat / "image_anno.csv"
    if not f.is_file():
        return {}
    out = {}
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            img = (r.get("image") or "").strip()
            lab = (r.get("label") or "").strip()
            if img:
                out[img] = lab
    return out


def convert(src_root: Path, dst_root: Path, cats: list[str]) -> dict:
    split_csv = src_root / "split_csv" / "1cls.csv"
    if not split_csv.is_file():
        raise FileNotFoundError(f"缺少官方划分文件: {split_csv}")

    stats = defaultdict(lambda: dict(train_good=0, test_good=0, test_defect=0, mask=0))

    with open(split_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"1cls.csv 共 {len(rows)} 行")

    for cat in cats:
        cat_src = src_root / cat
        if not cat_src.is_dir():
            print(f"  [skip] 源目录不存在: {cat_src}")
            continue
        dst_cat = dst_root / cat
        for sub in ("train/good", "test/good"):
            (dst_cat / sub).mkdir(parents=True, exist_ok=True)
        (dst_cat / "ground_truth").mkdir(parents=True, exist_ok=True)

        # per-class 复合缺陷标签（1cls.csv 只给二值 anomaly）
        fine = _load_defect_types(src_root, cat)

        n = 0
        for r in rows:
            if r.get("object") != cat:
                continue
            split = (r.get("split") or "").strip().lower()
            label = (r.get("label") or "").strip()
            img_rel = (r.get("image") or "").strip()
            mask_rel = (r.get("mask") or "").strip()
            if split not in ("train", "test") or not img_rel:
                continue

            src_img = src_root / img_rel
            if not src_img.is_file():
                # 少数镜像条目可能缺失，跳过并计数
                continue

            if split == "train":
                # 训练集按官方口径只含正常图；若混入异常图则跳过（防污染正常参考）
                if label != "normal":
                    continue
                shutil.copy2(src_img, dst_cat / "train" / "good" / src_img.name)
                stats[cat]["train_good"] += 1
            else:
                if label == "normal":
                    shutil.copy2(src_img, dst_cat / "test" / "good" / src_img.name)
                    stats[cat]["test_good"] += 1
                else:
                    defect = _clean_defect(fine.get(img_rel, label))
                    (dst_cat / "test" / defect).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_img, dst_cat / "test" / defect / src_img.name)
                    stats[cat]["test_defect"] += 1

                    if mask_rel:
                        src_mask = src_root / mask_rel
                        if src_mask.is_file():
                            gt_dir = dst_cat / "ground_truth" / defect
                            gt_dir.mkdir(parents=True, exist_ok=True)
                            stem = os.path.splitext(src_img.name)[0]
                            shutil.copy2(src_mask, gt_dir / f"{stem}_mask.png")
                            stats[cat]["mask"] += 1
                        else:
                            print(f"  [warn] mask 缺失: {src_mask}")
            n += 1
        print(f"  {cat:12s} 处理 {n} 行")

    return dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/visa_raw", help="镜像下载根目录")
    ap.add_argument("--dst", default="data/visa", help="转换后的 MVTec 布局目录")
    ap.add_argument("--cats", default="")
    args = ap.parse_args()

    root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    src = Path(args.src)
    src = src if src.is_absolute() else root / src
    dst = Path(args.dst)
    dst = dst if dst.is_absolute() else root / dst

    cats = [c.strip() for c in args.cats.split(",") if c.strip()] or CATS
    dst.mkdir(parents=True, exist_ok=True)

    stats = convert(src, dst, cats)

    print("\n=== 转换统计 ===")
    tot = defaultdict(int)
    for cat in cats:
        s = stats.get(cat)
        if not s:
            continue
        line = (f"{cat:12s} train_good={s['train_good']:4d} test_good={s['test_good']:4d} "
                f"test_defect={s['test_defect']:4d} masks={s['mask']:4d}")
        print(line)
        for k, v in s.items():
            tot[k] += v
    print(f"{'TOTAL':12s} train_good={tot['train_good']:4d} test_good={tot['test_good']:4d} "
          f"test_defect={tot['test_defect']:4d} masks={tot['mask']:4d}")


if __name__ == "__main__":
    main()
