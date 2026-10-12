# -*- coding: utf-8 -*-
"""生产配置 v2 的正式流程验证驱动（串行、可断点续跑、避免显存竞争）。

v2   = DINOv2 + diag + fusion=mean + FGSC 固定低通 a0=0.6/K=33
       + shrinkage 0.02 + ref_smooth 0 + image_score_topk 0.005
base = 现行生产配置(DINOv2 + diag + a0=0.6/K=33 + shrinkage 0.1 + ref_smooth 7 + q=0.01)
两者只差 shrinkage / ref_smooth / q 三项，逐项隔离其贡献；NEU/ELPV 未参与调参，
在新旧配置上的对照构成独立检验。
"""
from __future__ import annotations
import os
import subprocess
import time

PY = r"D:\cvs\pythoncode\yolo-insulator\.venv\Scripts\python.exe"
ROOT = r"d:\cvs\pythoncode\AnomalyDetection"
LOG = os.path.join(ROOT, "logs", "v2rest.log")

COMMON = ("--variant A2 --backbone dinov2 --fgsc-mode lowpass --fusion mean "
          "--fgsc-lambda 0.6 --fgsc-kernel 33 --image-blur-k 11")
V2 = COMMON + " --shrinkage 0.02 --ref-smooth 0 --image-score-topk 0.005"
BASE = COMMON + " --shrinkage 0.1 --image-score-topk 0.01"

JOBS = [
    ("v2_s7", V2, "mvtec", 7),
    ("base_s1", BASE, "mvtec", 1),
    ("base_s7", BASE, "mvtec", 7),
    ("neu_v2", V2, "neu", 42),
    ("neu_base", BASE, "neu", 42),
    ("pvel_v2", V2, "pvel", 42),
    ("pvel_base", BASE, "pvel", 42),
]


def main() -> None:
    with open(LOG, "a", encoding="utf-8", buffering=1) as lg:
        for tag, opts, ds, seed in JOBS:
            name = ("mvtec15_%s" % tag) if ds == "mvtec" else ("%s_%s" % (ds, tag))
            out = os.path.join(ROOT, "outputs", name)
            if os.path.exists(os.path.join(out, "summary.csv")):
                lg.write("SKIP %s (summary.csv already exists)\n" % tag)
                continue
            cmd = [PY, os.path.join(ROOT, "code", "scripts", "run_eval.py")] + opts.split() + [
                   "--dataset", ds, "--seed", str(seed), "--out", out]
            lg.write("\n===== START %s  %s =====\n" % (tag, time.strftime("%H:%M:%S")))
            t0 = time.time()
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            lg.write(p.stdout[-3000:] + "\n")
            if p.returncode != 0:
                lg.write("STDERR:\n" + p.stderr[-2000:] + "\n")
            lg.write("===== END %s rc=%d %.1f min =====\n"
                     % (tag, p.returncode, (time.time() - t0) / 60))
            lg.flush()
    with open(os.path.join(ROOT, "logs", "v2rest.done"), "w") as f:
        f.write("done\n")


if __name__ == "__main__":
    main()