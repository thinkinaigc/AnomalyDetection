"""抽样可视化：按类别画出带标注框的样例网格，用于核对类别名与标注质量。

用法:
    python code/scripts/visualize_samples.py --dataset neu_det_split --out logs/class_check_neu.png
    python code/scripts/visualize_samples.py --dataset PKU-Market-PCB --out logs/class_check_pcb.png --crop-boxes
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.common import ROOT  # noqa: E402


def draw_boxes(img, label_path, color=(0, 0, 255), thickness=2):
    h, w = img.shape[:2]
    if label_path and os.path.isfile(label_path):
        with open(label_path) as f:
            for line in f:
                t = line.split()
                if len(t) < 5:
                    continue
                cx, cy, bw, bh = map(float, t[1:5])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="data/ 下的 YOLO 格式数据集目录名")
    ap.add_argument("--split", default="train")
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--cell", type=int, default=200)
    ap.add_argument("--crop-boxes", action="store_true", help="裁剪并放大标注框区域（看清小缺陷）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    img_dir = os.path.join(ROOT, "data", args.dataset, args.split, "images")
    lab_dir = os.path.join(ROOT, "data", args.dataset, args.split, "labels")
    files = sorted(glob.glob(os.path.join(lab_dir, "*.txt")))
    if not files:
        print("未找到标签文件")
        return 1

    by_cls: dict[int, list[str]] = {}
    for f in files:
        with open(f) as fh:
            cls = {int(l.split()[0]) for l in fh if l.strip()}
        for c in cls:
            by_cls.setdefault(c, []).append(f)

    classes = sorted(by_cls)
    cell = args.cell
    canvas = np.full((len(classes) * cell, args.per_class * cell, 3), 255, dtype=np.uint8)
    for r, c in enumerate(classes):
        picks = random.sample(by_cls[c], min(args.per_class, len(by_cls[c])))
        for j, lf in enumerate(picks):
            stem = os.path.splitext(os.path.basename(lf))[0]
            ip = None
            for ext in (".jpg", ".png", ".jpeg", ".bmp"):
                cand = os.path.join(img_dir, stem + ext)
                if os.path.isfile(cand):
                    ip = cand
                    break
            if ip is None:
                continue
            img = cv2.imread(ip)
            if args.crop_boxes:
                h, w = img.shape[:2]
                boxes = []
                with open(lf) as fh:
                    for line in fh:
                        t = line.split()
                        if len(t) >= 5 and int(t[0]) == c:
                            cx, cy, bw, bh = map(float, t[1:5])
                            boxes.append((cx, cy, bw, bh))
                if boxes:
                    cx, cy, bw, bh = max(boxes, key=lambda b: b[2] * b[3])
                    pad = 3.0
                    x1 = max(int((cx - bw * pad / 2) * w), 0)
                    y1 = max(int((cy - bh * pad / 2) * h), 0)
                    x2 = min(int((cx + bw * pad / 2) * w), w)
                    y2 = min(int((cy + bh * pad / 2) * h), h)
                    img = img[y1:y2, x1:x2]
            img = draw_boxes(img.copy(), None if args.crop_boxes else lf)
            img = cv2.resize(img, (cell, cell))
            cv2.putText(img, f"cls{c}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            canvas[r * cell:(r + 1) * cell, j * cell:(j + 1) * cell] = img

    out = args.out or os.path.join(ROOT, "logs", f"class_check_{args.dataset}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, canvas)
    print("已保存:", out, canvas.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
