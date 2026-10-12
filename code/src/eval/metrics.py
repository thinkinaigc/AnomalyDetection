"""评价指标：像素 AUROC / AP，图像 AUROC / AP，FPR@95TPR，AUPRO，微小缺陷 AP。"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _as_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x).ravel()


def pixel_auroc(masks, scores) -> float:
    y_true, y_score = _as_np(masks), _as_np(scores)
    if y_true.min() == y_true.max():
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def pixel_ap(masks, scores) -> float:
    y_true, y_score = _as_np(masks), _as_np(scores)
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def image_auroc(labels, scores) -> float:
    y_true, y_score = _as_np(labels), _as_np(scores)
    if y_true.min() == y_true.max():
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def image_ap(labels, scores) -> float:
    y_true, y_score = _as_np(labels), _as_np(scores)
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def fpr_at_95tpr(masks, scores) -> float:
    """像素级 FPR@95TPR（越低越好）。"""
    y_true, y_score = _as_np(masks), _as_np(scores)
    if y_true.min() == y_true.max():
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])



def _label_regions(mask2d: np.ndarray) -> tuple[np.ndarray, int]:
    """对二值 mask 做 4-连通域标记，返回 (label_map, n_labels)。标签从 1 开始，0 为背景。"""
    try:
        import cv2

        n, lab = cv2.connectedComponents(mask2d.astype(np.uint8), connectivity=4)
        return lab.astype(np.int32), int(n - 1)
    except Exception:
        # 退化实现：整体当作一个区域
        return mask2d.astype(np.int32), int(mask2d.any())


def aupro(masks_2d: list, scores_2d: list) -> float:
    """AUPRO（Area Under the Per-Region Overlap curve），MVTec 定位评测标准指标。

    与像素 AUROC 的区别：AUPRO 先对每个 GT 连通缺陷**区域**单独计算重叠率，
    再对区域重叠率取平均，因此小缺陷区域不会像在像素 AUROC 中那样被大面积
    背景像素淹没——这正是工业微缺陷评测要求单列 AUPRO 的原因。

    算法（等价于官方 mupro 的向量化实现）：
      1) 把所有图的分数展平后整体降序排序，得到阈值序列；
      2) 沿该序列逐步激活像素，维护每个缺陷区域已覆盖的像素数；
      3) 在每个包含新区域激活的阈值处记录一个 (FPR, 区域平均重叠率) 点；
      4) 对PRO 曲线做梯形积分并按FPR 全区间归一化。

    参数
    ----
    masks_2d  : [N] 每张图的二值 GT mask（H*W）
    scores_2d : [N] 每张图的连续异常图（H*W），与 mask 同尺寸

    返回 PRO 曲线下面积（0..1，越高越好）；无有效缺陷区域时返回 nan。
    """
    if len(masks_2d) == 0 or len(masks_2d) != len(scores_2d):
        return float("nan")

    flat_scores = []
    flat_labels = []          # 全局区域 id（0 = 背景），跨图统一编号
    base = 0                  # 当前图之前的区域总数；本图区域 id = base + lab
    for m, s in zip(masks_2d, scores_2d):
        m = np.asarray(m).astype(bool)
        s = np.asarray(s, dtype=np.float64)
        if m.shape != s.shape:
            k = int(min(m.size, s.size))
            m = m.ravel()[:k]
            s = s.ravel()[:k]
        lab, nreg = _label_regions(m)
        flat_scores.append(s.ravel())
        # 全局区域 id = base + 本图 lab，且**只平移前景**：背景（lab=0）必须保持 0。
        # 若整体加 base，本图背景会被错记为上一个区域的延续，污染 region_sizes。
        lab64 = lab.ravel().astype(np.int64)
        flat_labels.append(np.where(lab64 > 0, lab64 + base, 0))
        base += int(nreg)

    scores = np.concatenate(flat_scores)
    labels = np.concatenate(flat_labels)
    n_total_regions = base
    if n_total_regions == 0:
        return float("nan")

    n_bg_total = int((labels == 0).sum())
    if n_bg_total == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    l_sorted = labels[order]

    region_sizes = np.bincount(l_sorted[l_sorted > 0],
                               minlength=n_total_regions + 1).astype(np.float64)

    # 沿分数降序、在每个"不同分值"的边界处取一个 (FPR, PRO) 点。
    # 同分值必须整批激活，否则并列分数会造成 PRO 抖动。
    hit_acc = np.zeros(n_total_regions + 1, dtype=np.int64)
    n_bg_hit = 0
    fprs = [0.0]
    pros = [0.0]
    i = 0
    N = s_sorted.size
    while i < N:
        v = s_sorted[i]
        j = i
        while j < N and s_sorted[j] == v:
            j += 1
        seg = l_sorted[i:j]
        n_bg_hit += int((seg == 0).sum())
        seg = seg[seg > 0]
        if seg.size:
            hit_acc += np.bincount(seg, minlength=n_total_regions + 1)
        overlap = hit_acc[1:] / region_sizes[1:]
        fprs.append(n_bg_hit / n_bg_total)
        pros.append(float(np.mean(overlap)))
        i = j

    f = np.asarray(fprs, dtype=np.float64)
    p = np.asarray(pros, dtype=np.float64)

    # PRO 是阈值的阶梯函数：在同一个 FPR 上，阈值继续下降会让 PRO 竖直跳变，
    # 而这些竖直段在 FPR 轴上宽度为 0，不贡献积分面积。因此不能对 (FPR, PRO)
    # 直接做梯形积分（那会把跳变摊成虚假斜坡），必须按阶梯函数积分：
    #   对每个相异 FPR 值 f_i，取"FPR <= f_i 的所有点中 PRO 的最大值"作为
    #   区间 [f_i, f_{i+1}) 上的有效 PRO（PRO 随 FPR 非减，故取累积最大），
    #   再以左端 Riemann 和求面积。该约定同时满足两个方向性检验：
    #   完美检测器 -> 1.0（PRO 在 FPR=0 即达 1），反向检测器 -> 0.0（PRO 直到
    #   FPR=1 才升起，该跳变宽度为 0）。
    order_f = np.argsort(f, kind="mergesort")
    f, p = f[order_f], p[order_f]
    cummax = np.maximum.accumulate(p)
    # 每个相异 FPR 分组的最后一个下标
    grp_end = np.flatnonzero(np.concatenate([np.diff(f) > 0, [True]]))
    f_g = f[grp_end]
    p_g = cummax[grp_end]
    if f_g.size < 2:
        return float("nan")
    area = float(np.sum(np.diff(f_g) * p_g[:-1]))
    return float(area)


def subset_indices(areas: np.ndarray, max_area_ratio: float, image_area: float) -> np.ndarray:
    areas = np.asarray(areas)
    return np.where((areas > 0) & (areas <= max_area_ratio * image_area))[0]


def tiny_defect_ap(labels, scores, areas, max_area_ratio: float = 0.001, image_area: float = 224 * 224) -> float:
    """微小缺陷子集上的图像级 AP（正负样本混合，负样本取全部正常图）。"""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    areas = np.asarray(areas)
    idx = subset_indices(areas, max_area_ratio, image_area)
    keep = np.concatenate([idx, np.where(labels == 0)[0]])
    if keep.size == 0 or labels[keep].sum() == 0:
        return float("nan")
    return float(average_precision_score(labels[keep], scores[keep]))


def compute_all(
    image_labels,
    image_scores,
    pixel_masks,
    pixel_scores,
    areas,
    image_area: float = 224 * 224,
    tiny_ratio: float = 0.001,
) -> dict:
    """汇总所有指标。

    image_labels / image_scores : [N] 图像级标签与分数
    pixel_masks / pixel_scores  : [M] 全部图像的像素级 GT 与分数（已展平拼接）
    areas                       : [N] 每张图的缺陷像素数（用于微小缺陷子集）
    """
    return {
        "image_auroc": image_auroc(image_labels, image_scores),
        "image_ap": image_ap(image_labels, image_scores),
        "pixel_auroc": pixel_auroc(pixel_masks, pixel_scores),
        "pixel_ap": pixel_ap(pixel_masks, pixel_scores),
        "fpr_at_95tpr": fpr_at_95tpr(pixel_masks, pixel_scores),
        "tiny_defect_ap": tiny_defect_ap(image_labels, image_scores, areas, tiny_ratio, image_area),
    }
