"""P1 数据预处理：把 PCB / NEU-DET 转成 MVTec 布局 + MVTec 构造方法可靠性验证。

只读取 data/ 下的原始数据，全部输出写到 data/processed/，绝不修改/删除原始文件。

子命令
------
pcb     : PKU-Market-PCB 高分辨率图切片 -> data/processed/pcb/<defect>/{train/good,test/...}
neu     : NEU-DET 构造正常样本（无缺陷区域 2x2 拼接 200x200）-> data/processed/neu/<class>/...
check   : MVTec 构造方法可靠性验证（4.3 节）-> data/processed/mvtec_check/<category>

示例
----
    python code/scripts/prepare_datasets.py pcb   --tile 640
    python code/scripts/prepare_datasets.py neu   --n-per-class 100
    python code/scripts/prepare_datasets.py check --category bottle
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

from src.utils.common import ROOT, get_logger  # noqa: E402

logger = get_logger("prepare", log_file=os.path.join(ROOT, "logs", "prepare.log"))

PCB_CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
NEU_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
NEU_DIRNAME = {c: c.replace("-", "_") for c in NEU_CLASSES}  # 目录名避免连字符


# ----------------------------------------------------------------------------- 通用
def _read_yolo_labels(label_path: str) -> list[tuple[int, float, float, float, float]]:
    out = []
    if not os.path.isfile(label_path):
        return out
    with open(label_path) as f:
        for line in f:
            t = line.split()
            if len(t) >= 5:
                out.append((int(t[0]), *map(float, t[1:5])))
    return out


def _ensure(*dirs: str) -> None:
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def _save_sample(img: np.ndarray, mask: np.ndarray, img_path: str, mask_path: str | None) -> None:
    _ensure(os.path.dirname(img_path))
    cv2.imwrite(img_path, img)
    if mask_path is not None:
        _ensure(os.path.dirname(mask_path))
        cv2.imwrite(mask_path, mask)


# ----------------------------------------------------------------------------- PCB
def build_pcb(tile: int = 640, max_good_per_image: int = 6, seed: int = 42) -> None:
    """高分辨率 PCB 图切片。

    * test 划分沿用用户已有的 train/valid/test 文件划分；
    * 每张图按 tile x tile 无重叠切片（边缘补齐），缺陷框中心所在 tile 记为该类异常；
    * 无缺陷 tile 作为正常样本（PCB 背景与缺陷类别无关，跨类共享正常背景）：
        train/good 只取 3 个已知类图像的空 tile（跨类别零样本设定）；
        test/good 取 test 集全部空 tile。
    """
    random.seed(seed)
    src = os.path.join(ROOT, "data", "PKU-Market-PCB")
    dst = os.path.join(ROOT, "data", "processed", "pcb")
    known = {"missing_hole", "mouse_bite", "spur"}

    for split in ("train", "valid", "test"):
        img_dir = os.path.join(src, split, "images")
        lab_dir = os.path.join(src, split, "labels")
        files = sorted(glob.glob(os.path.join(lab_dir, "*.txt")))
        n_good, n_defect = 0, 0
        for lf in files:
            stem = os.path.splitext(os.path.basename(lf))[0]
            ip = next(
                (os.path.join(img_dir, stem + e) for e in (".jpg", ".png", ".jpeg") if os.path.isfile(os.path.join(img_dir, stem + e))),
                None,
            )
            if ip is None:
                continue
            img = cv2.imread(ip)
            h, w = img.shape[:2]
            boxes = _read_yolo_labels(lf)
            ny, nx = (h + tile - 1) // tile, (w + tile - 1) // tile

            good_tiles: list[tuple[int, int]] = []
            for iy in range(ny):
                for ix in range(nx):
                    y0, x0 = iy * tile, ix * tile
                    y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                    tile_img = img[y0:y1, x0:x1]
                    if tile_img.size == 0:
                        continue
                    # 找中心落在本 tile 内的框
                    inner = []
                    for c, cx, cy, bw, bh in boxes:
                        pcx, pcy = cx * w, cy * h
                        if x0 <= pcx < x1 and y0 <= pcy < y1:
                            inner.append((c, cx * w - x0, cy * h - y0, bw * w, bh * h))
                    if not inner:
                        good_tiles.append((iy, ix))
                        continue

                    mask = np.zeros(tile_img.shape[:2], dtype=np.uint8)
                    for c, bx, by, bw, bh in inner:
                        x_1 = int(bx - bw / 2)
                        y_1 = int(by - bh / 2)
                        x_2 = int(bx + bw / 2)
                        y_2 = int(by + bh / 2)
                        cv2.rectangle(mask, (max(x_1, 0), max(y_1, 0)), (min(x_2, tile_img.shape[1]), min(y_2, tile_img.shape[0])), 255, -1)
                    if split != "test":
                        # 仅测试划分需要异常 tile；train/valid 只需正常背景 tile
                        continue
                    cls_name = PCB_CLASSES[inner[0][0]]
                    tag = f"{stem}_r{iy}c{ix}"
                    _save_sample(
                        tile_img, mask,
                        os.path.join(dst, cls_name, "test", cls_name, f"{tag}.jpg"),
                        os.path.join(dst, cls_name, "ground_truth", cls_name, f"{tag}_mask.png"),
                    )
                    n_defect += 1

            # 空 tile -> 正常样本（每图限量）
            random.shuffle(good_tiles)
            for k, (iy, ix) in enumerate(good_tiles[: max_good_per_image]):
                y0, x0 = iy * tile, ix * tile
                tile_img = img[y0:min(y0 + tile, h), x0:min(x0 + tile, w)]
                tag = f"{stem}_r{iy}c{ix}"
                if split == "test":
                    _save_sample(tile_img, None, os.path.join(dst, "_good", "test", "good", f"{tag}.jpg"), None)
                else:
                    # 训练用正常背景：仅取已知类别图像（零样本类别背景与其同类）
                    first_cls = PCB_CLASSES[boxes[0][0]] if boxes else None
                    if first_cls in known:
                        _save_sample(tile_img, None, os.path.join(dst, "_good", "train", "good", f"{tag}.jpg"), None)
                n_good += 1
        logger.info(f"PCB [{split}] defect_tiles={n_defect} good_tiles_kept={n_good}")

    # _good 的正常样本同时挂到每个类别的 train/good 与 test/good
    import shutil

    for sub in ("train", "test"):
        good_src = os.path.join(dst, "_good", sub, "good")
        if not os.path.isdir(good_src):
            continue
        for cls_name in PCB_CLASSES:
            tgt = os.path.join(dst, cls_name, sub, "good")
            _ensure(tgt)
            for f in sorted(os.listdir(good_src)):
                shutil.copyfile(os.path.join(good_src, f), os.path.join(tgt, f))
    logger.info("PCB 处理完成 -> " + dst)


# ----------------------------------------------------------------------------- NEU
def _largest_empty_rect(occ: np.ndarray, size: int) -> tuple[int, int] | None:
    """在占用图 occ（1=不可用）中找 size x size 全空区域，返回左上角。"""
    h, w = occ.shape
    if size > h or size > w:
        return None
    occ = np.ascontiguousarray(occ, dtype=np.int64)
    integ = np.zeros((occ.shape[0] + 1, occ.shape[1] + 1), dtype=np.int64)
    integ[1:, 1:] = occ.cumsum(axis=0).cumsum(axis=1)  # (h+1, w+1) 积分图
    s = integ
    for y in range(0, h - size + 1, 4):
        for x in range(0, w - size + 1, 4):
            if s[y + size, x + size] - s[y, x + size] - s[y + size, x] + s[y, x] == 0:
                return y, x
    return None


def build_neu(n_per_class: int = 100, patch: int = 100, margin: int = 8, seed: int = 42) -> None:
    """NEU-DET：构造正常样本 + 类别划分。

    * 异常测试样本直接来自用户划分好的 test/（带框 -> 矩形 mask）；
    * 正常样本 = 从该类图像的无缺陷区域裁 patch，2x2 拼接成 200x200：
        - 已知类（patches/pitted_surface/rolled-in_scale）: train/good 用本类构造样本
        - 零样本类（crazing/inclusion/scratches）: train/good 用已知类的构造样本（模型完全没见过目标类）
        - test/good 用 val 集图像构造（与训练互不重叠）
    """
    random.seed(seed)
    src = os.path.join(ROOT, "data", "neu_det_split")
    dst = os.path.join(ROOT, "data", "processed", "neu")
    known = ["patches", "pitted_surface", "rolled-in_scale"]

    def make_normals(img_dir: str, lab_dir: str, files: list[str], n_out: int, out_dir: str, tag: str) -> int:
        """从给定图像集合构造 n_out 张 200x200 正常样本。"""
        _ensure(out_dir)
        made, tries = 0, 0
        files = list(files)
        random.shuffle(files)
        while made < n_out and tries < n_out * 30 and files:
            tries += 1
            lf = files[(made + tries) % len(files)]
            stem = os.path.splitext(os.path.basename(lf))[0]
            ip = next(
                (os.path.join(img_dir, stem + e) for e in (".jpg", ".png") if os.path.isfile(os.path.join(img_dir, stem + e))),
                None,
            )
            if ip is None:
                continue
            img = cv2.imread(ip)
            h, w = img.shape[:2]
            occ = np.zeros((h, w), dtype=np.int32)
            for c, cx, cy, bw, bh in _read_yolo_labels(lf):
                x1 = max(int((cx - bw / 2) * w) - margin, 0)
                y1 = max(int((cy - bh / 2) * h) - margin, 0)
                x2 = min(int((cx + bw / 2) * w) + margin, w)
                y2 = min(int((cy + bh / 2) * h) + margin, h)
                occ[y1:y2, x1:x2] = 1
            tiles = []
            for size in (patch, 80, 64, 48):
                tiles = []
                ok = True
                for _ in range(4):
                    pos = _largest_empty_rect(occ, size)
                    if pos is None:
                        ok = False
                        break
                    y, x = pos
                    tiles.append(img[y:y + size, x:x + size])
                if ok:
                    break
            if not ok or len(tiles) != 4:
                continue
            sizes = {t.shape[0] for t in tiles}
            if len(sizes) == 1:
                size = tiles[0].shape[0]
                canvas = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
            else:  # 尺寸不一则统一缩放到 patch
                tiles = [cv2.resize(t, (patch, patch)) for t in tiles]
                canvas = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
            _ensure(out_dir)
            cv2.imwrite(os.path.join(out_dir, f"normal_{tag}_{made:04d}.jpg"), canvas)
            made += 1
        return made

    # 异常样本（train/val/test 直接搬运用户划分，并把 YOLO 框转成 mask）
    for split in ("train", "val", "test"):
        img_dir = os.path.join(src, split, "images")
        lab_dir = os.path.join(src, split, "labels")
        for lf in sorted(glob.glob(os.path.join(lab_dir, "*.txt"))):
            stem = os.path.splitext(os.path.basename(lf))[0]
            ip = next(
                (os.path.join(img_dir, stem + e) for e in (".jpg", ".png") if os.path.isfile(os.path.join(img_dir, stem + e))),
                None,
            )
            if ip is None:
                continue
            img = cv2.imread(ip)
            h, w = img.shape[:2]
            boxes = _read_yolo_labels(lf)
            if not boxes:
                continue
            cls_name = NEU_DIRNAME[NEU_CLASSES[boxes[0][0]]]
            mask = np.zeros((h, w), dtype=np.uint8)
            for c, cx, cy, bw, bh in boxes:
                x1, y1 = max(int((cx - bw / 2) * w), 0), max(int((cy - bh / 2) * h), 0)
                x2, y2 = min(int((cx + bw / 2) * w), w), min(int((cy + bh / 2) * h), h)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            out_split = "test" if split == "val" else split
            _save_sample(
                img, mask,
                os.path.join(dst, cls_name, out_split, cls_name, f"{stem}.jpg"),
                os.path.join(dst, cls_name, "ground_truth", cls_name, f"{stem}_mask.png"),
            )

    # 正常样本
    def class_files(split: str, cls_idx: int) -> list[str]:
        return sorted(
            f for f in glob.glob(os.path.join(src, split, "labels", "*.txt"))
            if any(int(l.split()[0]) == cls_idx for l in open(f) if l.strip())
        )

    known_idx = [NEU_CLASSES.index(c) for c in known]
    for ci, cls_name in enumerate(NEU_CLASSES):
        dn = NEU_DIRNAME[cls_name]
        if cls_name in known:
            made = make_normals(
                os.path.join(src, "train", "images"), os.path.join(src, "train", "labels"),
                class_files("train", ci), n_per_class,
                os.path.join(dst, dn, "train", "good"), dn,
            )
        else:  # 零样本类：训练用已知类的正常样本
            for kj in known_idx:
                make_normals(
                    os.path.join(src, "train", "images"), os.path.join(src, "train", "labels"),
                    class_files("train", kj), max(n_per_class // len(known_idx), 1),
                    os.path.join(dst, dn, "train", "good"), f"known_{NEU_DIRNAME[NEU_CLASSES[kj]]}",
                )
            made = len(glob.glob(os.path.join(dst, dn, "train", "good", "*.jpg")))
        logger.info(f"NEU [{dn}] train/good = {made}")

    # test/good：用 val 集全类图像构造
    all_val = []
    for ci in range(len(NEU_CLASSES)):
        all_val += class_files("val", ci)
    made = make_normals(
        os.path.join(src, "val", "images"), os.path.join(src, "val", "labels"),
        all_val, 6 * n_per_class // 3,
        os.path.join(ROOT, "data", "processed", "neu", "_test_good", "test", "good"), "testgood",
    )
    import shutil

    for cls_name in NEU_CLASSES:
        dn = NEU_DIRNAME[cls_name]
        src_good = os.path.join(dst, "_test_good", "test", "good")
        tgt = os.path.join(dst, dn, "test", "good")
        _ensure(tgt)
        for f in sorted(os.listdir(src_good)):
            shutil.copyfile(os.path.join(src_good, f), os.path.join(tgt, f))
    logger.info(f"NEU test/good = {made}  -> {dst}")


# ------------------------------------------------------------------- MVTec 构造可靠性验证
def build_mvtec_check(category: str = "bottle", n_construct: int = 100, seed: int = 42) -> None:
    """4.3 节：构造正常 vs 真实正常 对照实验的数据准备。

    把该类 test 异常图对半分（固定种子）：
        A 半 -> 裁无缺陷区域拼接，构造"伪正常"训练集
        B 半 -> 与 good 一起作为测试集
    输出 data/processed/mvtec_check/<category>/，其中
        constructed_normal/ 伪正常训练图
        test/ 与 ground_truth/ 只含 B 半异常图
    """
    random.seed(seed)
    src = os.path.join(ROOT, "data", "mvtec_ad", category)
    dst = os.path.join(ROOT, "data", "processed", "mvtec_check", category)
    defect_types = sorted(
        d for d in os.listdir(os.path.join(src, "test"))
        if d != "good" and os.path.isdir(os.path.join(src, "test", d))
    )
    anomalous = []
    for d in defect_types:
        for p in sorted(glob.glob(os.path.join(src, "test", d, "*.png"))):
            anomalous.append((d, p))
    random.shuffle(anomalous)
    half = len(anomalous) // 2
    build_half, eval_half = anomalous[:half], anomalous[half:]
    logger.info(f"[{category}] 异常图 {len(anomalous)} -> 构造用 {len(build_half)} / 测试用 {len(eval_half)}")

    # 伪正常训练集：从 build_half 的异常图中裁无缺陷区域
    out_train = os.path.join(dst, "constructed_normal")
    _ensure(out_train)
    per_img = max(n_construct // max(len(build_half), 1), 1)
    made = 0
    for d, p in build_half:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        gt = os.path.join(src, "ground_truth", d, os.path.basename(p).replace(".png", "_mask.png"))
        occ = np.zeros((h, w), dtype=np.int32)
        if os.path.isfile(gt):
            m = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
            occ[m > 0] = 1
        for k in range(per_img):
            pos = None
            for size in (256, 192, 128, 96):
                pos = _largest_empty_rect(occ, size)
                if pos is not None:
                    y, x = pos
                    crop = img[y:y + size, x:x + size]
                    crop = cv2.resize(crop, (h, w))  # 拉伸回整图尺寸，模拟"整张正常图"
                    break
            if pos is None:
                break
            cv2.imwrite(os.path.join(out_train, f"construct_{made:04d}.png"), crop)
            made += 1
    logger.info(f"[{category}] 构造伪正常 {made} 张")

    # 测试集：B 半异常图 + 全部 good（软链接省空间，Windows 用 copy）
    import shutil

    for d, p in eval_half:
        stem = os.path.splitext(os.path.basename(p))[0]
        _ensure(os.path.join(dst, "test", d))
        shutil.copyfile(p, os.path.join(dst, "test", d, stem + ".png"))
        gt = os.path.join(src, "ground_truth", d, stem + "_mask.png")
        if os.path.isfile(gt):
            _ensure(os.path.join(dst, "ground_truth", d))
            shutil.copyfile(gt, os.path.join(dst, "ground_truth", d, stem + "_mask.png"))
    for p in sorted(glob.glob(os.path.join(src, "test", "good", "*.png"))):
        _ensure(os.path.join(dst, "test", "good"))
        shutil.copyfile(p, os.path.join(dst, "test", "good", os.path.basename(p)))
    # 真实正常训练集直接引用原目录（AnomalyDataset 支持 root 指向 data/mvtec_ad）
    logger.info(f"[{category}] check 数据就绪 -> {dst}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["pcb", "neu", "check"])
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.task == "pcb":
        build_pcb(tile=args.tile, seed=args.seed)
    elif args.task == "neu":
        build_neu(n_per_class=args.n_per_class, seed=args.seed)
    else:
        build_mvtec_check(category=args.category, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
