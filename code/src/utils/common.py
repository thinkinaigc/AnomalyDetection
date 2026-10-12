"""通用工具：路径解析、随机种子、日志、计时与显存统计。"""

from __future__ import annotations

import logging
import os
import random
import time
from contextlib import contextmanager

import numpy as np
import torch

# 项目根目录（AnomalyDetection/）
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def path(*parts: str) -> str:
    """相对项目根目录拼路径。"""
    return os.path.join(ROOT, *parts)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(name: str = "fgsc_ada", log_file: str | None = None, level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def get_device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@contextmanager
def timer(name: str = "", logger=None):
    t0 = time.perf_counter()
    yield
    dt = (time.perf_counter() - t0) * 1000.0
    if logger is not None:
        logger.info(f"{name} 耗时 {dt:.2f} ms")
    else:
        print(f"{name} 耗时 {dt:.2f} ms")


class MemoryTracker:
    """CUDA 峰值显存统计（单位 MB）。"""

    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.enabled = device is not None and device.type == "cuda"
        self.peak = 0.0
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(device)

    def update(self) -> float:
        if not self.enabled:
            return 0.0
        self.peak = max(self.peak, torch.cuda.max_memory_allocated(self.device) / (1024 ** 2))
        return self.peak


def count_params(module: torch.nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def load_yaml(file: str) -> dict:
    import yaml

    with open(file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
