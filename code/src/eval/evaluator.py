"""评测器：跑完一个类别的测试集，汇总指标 / 延迟 / 峰值显存。"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..utils.common import MemoryTracker
from . import metrics


@torch.no_grad()
def evaluate(
    method,
    dataset,
    batch_size: int = 1,
    save_dir: Optional[str] = None,
    vis_num: int = 0,
    logger=None,
) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    device = method.device

    image_labels, image_scores, areas = [], [], []
    pixel_masks, pixel_scores = [], []
    latencies = []
    mem = MemoryTracker(device)

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"]
        t0 = time.perf_counter()
        out = method.predict(img)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0 / img.shape[0])

        amap = out["anomaly_map"].float().cpu()
        iscore = out["image_score"].float().cpu()
        for i in range(img.shape[0]):
            m = mask[i].numpy()
            s = amap[i, 0].numpy()
            if m.shape != s.shape:
                import cv2

                s = cv2.resize(s, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_LINEAR)
            image_labels.append(int(batch["label"][i]))
            image_scores.append(float(iscore[i]))
            areas.append(float(m.sum()))
            # 标准协议：像素指标在全部测试图像上计算（正常图掩码全 0，作为负样本）
            pixel_masks.append(m.ravel())
            pixel_scores.append(s.ravel())
        mem.update()

    # 像素指标必须有正样本；若该类别测试集无异常像素则整类跳过
    if pixel_masks:
        pm = np.concatenate(pixel_masks)
        ps = np.concatenate(pixel_scores)
    else:
        pm = np.zeros(1, dtype=np.float32)
        ps = np.zeros(1, dtype=np.float32)

    res = metrics.compute_all(
        np.array(image_labels),
        np.array(image_scores),
        pm,
        ps,
        np.array(areas),
        image_area=float(dataset.mask_size * dataset.mask_size),
    )
    res.update(
        {
            "n_images": len(image_labels),
            "n_anomaly": int(np.sum(image_labels)),
            "latency_ms": float(np.mean(latencies)),
            "peak_mem_mb": float(mem.peak),
            "trainable_params": int(method.trainable_params()),
        }
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(save_dir, "scores.npz"),
            image_labels=np.array(image_labels),
            image_scores=np.array(image_scores),
            areas=np.array(areas),
        )
        with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

    if logger:
        logger.info(
            f"[{dataset.category}] imgAUROC={res['image_auroc']:.4f} pixAUROC={res['pixel_auroc']:.4f} "
            f"pixAP={res['pixel_ap']:.4f} fpr95={res['fpr_at_95tpr']:.4f} "
            f"{res['latency_ms']:.1f}ms peak={res['peak_mem_mb']:.0f}MB"
        )
    return res
