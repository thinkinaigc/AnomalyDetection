"""下载本项目所需的预训练权重（离线可复用）。

- DINOv2 ViT-S/14（冻结视觉主干，Meta 官方直链）
- MobileCLIP-S1（冻结文本编码器，HF 镜像）

用法:
    python code/scripts/download_weights.py            # 下载全部
    python code/scripts/download_weights.py --only dinov2
    python code/scripts/download_weights.py --only mobileclip
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request

# HF 国内镜像（直连 huggingface.co 在本机会超时）
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS = os.path.join(ROOT, "weights")

DINOV2_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"

MOBILECLIP_REPO = "apple/MobileCLIP-S1-OpenCLIP"
MOBILECLIP_FILES = [
    "open_clip_config.json",
    "open_clip_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]


def _download(url: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"[skip] {dst} 已存在 ({os.path.getsize(dst) / 1e6:.1f} MB)")
        return
    tmp = dst + ".part"
    print(f"[get ] {url}")
    print(f"    -> {dst}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r    {done / 1e6:.1f}/{total / 1e6:.1f} MB", end="", flush=True)
    if total:
        print()
    os.replace(tmp, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["all", "dinov2", "mobileclip"], default="all")
    args = ap.parse_args()

    if args.only in ("all", "dinov2"):
        _download(DINOV2_URL, os.path.join(WEIGHTS, "dinov2_vits14_pretrain.pth"))

    if args.only in ("all", "mobileclip"):
        out = os.path.join(WEIGHTS, "mobileclip_s1")
        for name in MOBILECLIP_FILES:
            url = f"{HF_ENDPOINT}/{MOBILECLIP_REPO}/resolve/main/{name}"
            _download(url, os.path.join(out, name))

    print("[done] 权重目录:", WEIGHTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
