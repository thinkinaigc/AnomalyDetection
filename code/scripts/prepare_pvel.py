"""把 ELPV(pvel_ad, VOC 框标注) 转成 MVTec 布局，供统一 Dataset 读取。

布局: <out>/<class>/{train/good, test/good, test/<class>, ground_truth/<class>}
- 正常图来自 othertypes/good，按固定种子切分为 train_good / test_good（各类共享同一划分）。
- 异常图为含该类缺陷的图；像素 mask 由该类 bbox 并集填充为矩形（ELPV 仅提供框，像素指标为近似）。
- 一个图可含多类缺陷，会分别归入各自类别（每类用其自身的框生成 mask）。
"""
from __future__ import annotations
import os, re, glob, shutil
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from collections import defaultdict

B = r"D:\cvs\pythoncode\IndustryDetector\indu-vision\paper\data\pvel_ad"
OUT = os.path.join("data", "processed", "pvel_ad")
SEED = 42
TRAIN_GOOD = 250
TEST_GOOD = 100
TEST_DEFECT_CAP = 600
MIN_CLASS_IMGS = 30
rng = np.random.RandomState(SEED)


def clean(txt: str) -> str:
    return re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad]", "", txt)


def parse_all():
    recs = []
    for p in glob.glob(os.path.join(B, "trainval", "Annotations", "*.xml")):
        root = ET.fromstring(clean(open(p, "r", encoding="utf-8", errors="ignore").read()))
        fn = (root.findtext("filename") or "").strip()
        sz = root.find("size")
        W = int(sz.findtext("width")); H = int(sz.findtext("height"))
        objs = defaultdict(list)  # name -> [(x1,y1,x2,y2)]
        for o in root.findall("object"):
            nm = (o.findtext("name") or "").strip()
            bb = o.find("bndbox")
            if bb is None or bb.findtext("xmin") is None:
                continue
            x1 = int(bb.findtext("xmin")); y1 = int(bb.findtext("ymin"))
            x2 = int(bb.findtext("xmax")); y2 = int(bb.findtext("ymax"))
            objs[nm].append((x1, y1, x2, y2))
        if fn and objs:
            recs.append((fn, W, H, dict(objs)))
    return recs


def main():
    recs = parse_all()
    print("parsed labeled images:", len(recs))
    good_files = sorted(glob.glob(os.path.join(B, "othertypes", "good", "*")))
    good_files = [f for f in good_files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    print("othertypes/good:", len(good_files))
    perm = rng.permutation(len(good_files))
    train_good = [good_files[i] for i in perm[:TRAIN_GOOD]]
    test_good = [good_files[i] for i in perm[TRAIN_GOOD:TRAIN_GOOD + TEST_GOOD]]

    # 按类组织异常图
    by_cls = defaultdict(list)  # name -> [(fn,W,H,boxes)]
    for fn, W, H, objs in recs:
        for nm, boxes in objs.items():
            by_cls[nm].append((fn, W, H, boxes))

    stats = {}
    for cls in sorted(by_cls):
        items = by_cls[cls]
        if len(items) < MIN_CLASS_IMGS:
            print(f"skip {cls}: only {len(items)} imgs")
            continue
        idx = rng.permutation(len(items))[:TEST_DEFECT_CAP]
        sel = [items[i] for i in idx]
        base = os.path.join(OUT, cls)
        for sub in ("train/good", "test/good", f"test/{cls}", f"ground_truth/{cls}"):
            os.makedirs(os.path.join(base, sub), exist_ok=True)
        # 正常图（每类拷贝一份；MVTec 风格按类独立）
        for f in train_good:
            shutil.copyfile(f, os.path.join(base, "train/good", os.path.basename(f)))
        for f in test_good:
            shutil.copyfile(f, os.path.join(base, "test/good", os.path.basename(f)))
        n_def = 0
        for fn, W, H, boxes in sel:
            src = os.path.join(B, "trainval", "JPEGImages", fn)
            if not os.path.isfile(src):
                continue
            shutil.copyfile(src, os.path.join(base, "test", cls, fn))
            stem = os.path.splitext(fn)[0]
            m = np.zeros((H, W), dtype=np.uint8)
            for (x1, y1, x2, y2) in boxes:
                x1 = max(0, min(x1, W - 1)); y1 = max(0, min(y1, H - 1))
                x2 = max(x1 + 1, min(x2, W - 1)); y2 = max(y1 + 1, min(y2, H - 1))
                m[y1:y2 + 1, x1:x2 + 1] = 255
            cv2.imwrite(os.path.join(base, "ground_truth", cls, stem + "_mask.png"), m)
            n_def += 1
        stats[cls] = dict(train_good=len(train_good), test_good=len(test_good), test_defect=n_def)
        print(f"  {cls:24s} train_good={len(train_good)} test_good={len(test_good)} test_defect={n_def}")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
