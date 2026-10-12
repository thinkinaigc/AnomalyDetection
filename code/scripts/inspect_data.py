"""P1 数据体检：打印三个数据集的目录结构、类别分布、图像尺寸，检查是否符合 MVTec 布局。

用法:
    python code/scripts/inspect_data.py [--data-root data] [--out logs/inspect_data.txt]
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]


def _line(print_fn, s=""):
    print_fn(s)


def inspect_mvtec(data_root, print_fn):
    root = os.path.join(data_root, "mvtec_ad")
    if not os.path.isdir(root):
        print_fn("[mvtec_ad] 缺失")
        return
    cats = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    _line(print_fn, f"=== MVTec-AD  (root={root})")
    _line(print_fn, f"类别数: {len(cats)} / 15 -> {cats}")
    missing = [c for c in MVTEC_CATEGORIES if c not in cats]
    _line(print_fn, f"缺失类别: {missing}")
    for c in cats:
        cd = os.path.join(root, c)
        n_train = len(glob.glob(os.path.join(cd, "train", "good", "*.*")))
        test_dirs = sorted(
            d for d in os.listdir(os.path.join(cd, "test"))
            if os.path.isdir(os.path.join(cd, "test", d))
        ) if os.path.isdir(os.path.join(cd, "test")) else []
        n_test = {d: len(glob.glob(os.path.join(cd, "test", d, "*.*"))) for d in test_dirs}
        n_gt = len(glob.glob(os.path.join(cd, "ground_truth", "*", "*.*")))
        _line(print_fn, f"  {c:12s} train/good={n_train:4d} test={n_test} gt={n_gt}")
    _line(print_fn)


def inspect_yolo(data_root, ds, print_fn):
    root = os.path.join(data_root, ds)
    if not os.path.isdir(root):
        print_fn(f"[{ds}] 缺失")
        return
    _line(print_fn, f"=== {ds}  (root={root})")
    for split in ["train", "valid", "val", "test"]:
        img_dir = os.path.join(root, split, "images")
        lab_dir = os.path.join(root, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        imgs = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        labels = sorted(glob.glob(os.path.join(lab_dir, "*.txt"))) if os.path.isdir(lab_dir) else []
        cls_counter = collections.Counter()
        n_boxes = 0
        sizes = collections.Counter()
        for f in labels:
            with open(f) as fh:
                for line in fh:
                    t = line.split()
                    if not t:
                        continue
                    cls_counter[int(t[0])] += 1
                    n_boxes += 1
        for f in imgs[: min(50, len(imgs))]:
            im = cv2.imread(f)
            if im is not None:
                sizes[(im.shape[1], im.shape[0])] += 1
        _line(print_fn, f"  [{split}] images={len(imgs)} labels={len(labels)} boxes={n_boxes}")
        _line(print_fn, f"      class_counts={dict(sorted(cls_counter.items()))}")
        _line(print_fn, f"      sizes(sample)={dict(list(sizes.items())[:5])}")
    _line(print_fn)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "inspect_data.txt"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fh = open(args.out, "w", encoding="utf-8")

    def print_fn(s=""):
        print(s)
        fh.write(s + "\n")

    inspect_mvtec(args.data_root, print_fn)
    inspect_yolo(args.data_root, "PKU-Market-PCB", print_fn)
    inspect_yolo(args.data_root, "neu_det_split", print_fn)
    fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
