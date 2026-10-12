"""并行下载 VisA 数据集（12 类）到 MVTec 布局。

数据源为 HuggingFace 镜像 imaadd05/visa-anomaly-detection（官方
VisA_20220922.tar 的逐文件未改镜像，CC BY 4.0）。官方 AWS S3 直连在
本机仅约 7 KB/s，镜像约 366 KB/s，故改走镜像 + 多线程。

用法：
    python code/scripts/_download_visa.py --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE = "https://hf-mirror.com/datasets/imaadd05/visa-anomaly-detection/resolve/main"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DST = os.path.join(ROOT, "data", "visa_raw")
MANIFEST = os.path.join(ROOT, "outputs", "visa_files.json")

CATS = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
        "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]


def fetch_one(rel: str, session: requests.Session, retries: int = 4) -> tuple[str, bool, str]:
    """下载单个文件到 DST/<rel>；已存在且非空则跳过。"""
    out = os.path.join(DST, rel.replace("/", os.sep))
    if os.path.isfile(out) and os.path.getsize(out) > 0:
        return rel, True, "skip"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    last = ""
    for attempt in range(retries):
        try:
            r = session.get(f"{BASE}/{rel}", timeout=120)
            if r.status_code != 200:
                last = f"HTTP {r.status_code}"
                time.sleep(2 + 3 * attempt)
                continue
            tmp = out + ".part"
            with open(tmp, "wb") as fh:
                fh.write(r.content)
            os.replace(tmp, out)
            return rel, True, "ok"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
            time.sleep(2 + 3 * attempt)
    return rel, False, last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--cats", default="", help="逗号分隔类别；缺省全部 12 类")
    args = ap.parse_args()

    cats = [c.strip() for c in args.cats.split(",") if c.strip()] or CATS

    with open(MANIFEST, encoding="utf-8") as fh:
        files = [s["rfilename"] for s in json.load(fh)["siblings"]]

    todo = [f for f in files
            if f.endswith((".JPG", ".png", ".csv")) and (f.split("/")[0] in cats)]
    # 每类的 image_anno.csv 与官方划分都要
    todo += ["split_csv/1cls.csv"]
    todo = sorted(set(todo))
    print(f"待下载 {len(todo)} 个文件 -> {DST}", flush=True)

    ok = fail = skip = 0
    fails = []
    t0 = time.time()
    with requests.Session() as session:
        session.mount("https://", requests.adapters.HTTPAdapter(
            pool_maxsize=args.workers * 2, max_retries=0))
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_one, rel, session): rel for rel in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                rel, good, msg = fut.result()
                if good:
                    ok += 1
                    if msg == "skip":
                        skip += 1
                else:
                    fail += 1
                    fails.append((rel, msg))
                if i % 200 == 0 or i == len(todo):
                    el = time.time() - t0
                    rate = i / max(el, 1e-6)
                    eta = (len(todo) - i) / max(rate, 1e-6)
                    print(f"  {i}/{len(todo)}  ok={ok} skip={skip} fail={fail}  "
                          f"{rate:.1f} file/s  ETA {eta / 60:.1f} min", flush=True)

    print(f"\n完成: ok={ok} skip={skip} fail={fail}  用时 {(time.time() - t0) / 60:.1f} min")
    if fails:
        print("失败清单（前 20）:")
        for rel, msg in fails[:20]:
            print(f"  {rel}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
