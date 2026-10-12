# -*- coding: utf-8 -*-
"""Item 2 (IEEE-TIM checklist): fully-supervised, SAME-backbone head-to-head.

We train a per-pixel logistic-regression probe on the *frozen* DINOv2-B features
(the identical backbone + random projection our method uses) with ground-truth
masks. MVTec train is normal-only (no masks), so we follow a within-category
leave-half-out image split on the test set: train the probe on one half, evaluate
both the probe and our training-free method on the other half (no leakage). This
isolates the value of supervision on the identical representation.

Outputs exp/rev_supervised.json  ->  head-to-head table data.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "code"))   # src package lives at code/src
from src.data.dataset import AnomalyDataset, list_categories
from src.methods.factory import build_method
from src.utils.common import load_yaml, set_seed, ROOT as SRC_ROOT
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

CFG = load_yaml(os.path.join(ROOT, "code", "configs", "default.yaml"))
# The paper's primary backbone is DINOv2 ViT-B/14 (the config default says mobileclip,
# which is stale) -- force it so the supervised probe and our map share the same backbone.
CFG["model"]["backbone"] = "dinov2"
DEVICE = "cuda"
set_seed(42)

OUT = os.path.join(ROOT, "exp", "rev_supervised.json")

def build_ours():
    m = build_method(CFG, "A2", device=DEVICE,
                     override_method=dict(fusion="mean", fgsc="lowpass",
                                          fgsc_lambda=0.6, fgsc_tau=0.0,
                                          fgsc_kernel=33, shrinkage=0.02,
                                          cov_mode="diag", image_score_topk=0.005))
    return m

def topq_score(maps, q=0.005):
    # maps: [N,1,H,W] tensor
    B = maps.shape[0]
    flat = maps.reshape(B, -1)
    k = max(1, int(q * flat.shape[1]))
    v, _ = torch.topk(flat, k, dim=1)
    return v.mean(dim=1).cpu().numpy()

def main():
    method = build_ours()
    bb = method.backbone
    img_size = int(getattr(bb, "image_size", 256))
    mean = tuple(float(x) for x in bb.image_mean)
    std = tuple(float(x) for x in bb.image_std)
    map_size = int(CFG.get("eval", {}).get("map_size", 256))
    data_root = os.path.join(ROOT, CFG["data"]["mvtec_root"])
    cats = list_categories(data_root)
    scales = method.scale_names

    results = {}
    t0 = time.time()
    for ci, cat in enumerate(cats):
        method.set_category(cat)
        train_ds = AnomalyDataset(data_root, cat, "train", img_size, map_size, mean, std, only_good=True)
        method.fit(DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=0))

        test_ds = AnomalyDataset(data_root, cat, "test", img_size, map_size, mean, std)
        n = len(test_ds)
        idxA = list(range(0, n, 2)); idxB = list(range(1, n, 2))

        # ---- collect split-A features + masks for probe training ----
        feat_buf = {s: [] for s in scales}
        mask_buf = {s: [] for s in scales}
        for i in idxA:
            s = test_ds[i]
            img = s["image"].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feats = method.extract(img)      # {scale:[1,C,H,W]} projected
            mask256 = s["mask"].numpy().astype(np.float32)
            for sc in scales:
                f = feats[sc][0].permute(1, 2, 0).reshape(-1, feats[sc].shape[1]).cpu().numpy().astype(np.float32)
                H, W = feats[sc].shape[2], feats[sc].shape[3]
                mg = (np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) if False else None)
                # downsample mask to grid HxW via nearest
                mh = cv2_down(mask256, (W, H))
                feat_buf[sc].append(f); mask_buf[sc].append(mh.reshape(-1))

        probes = {}
        for sc in scales:
            X = np.concatenate(feat_buf[sc], 0); Y = (np.concatenate(mask_buf[sc], 0) > 0.5).astype(int)
            if X.shape[0] > 80000:
                sel = np.random.choice(X.shape[0], 80000, replace=False)
                X, Y = X[sel], Y[sel]
            # balance: keep all defects + subsample normals
            pos = np.where(Y == 1)[0]; neg = np.where(Y == 0)[0]
            if len(pos) == 0:
                probes[sc] = None; continue
            if len(neg) > 4 * len(pos):
                neg = np.random.choice(neg, 4 * len(pos), replace=False)
            sel = np.concatenate([pos, neg])
            lr = LogisticRegression(max_iter=400, class_weight="balanced", C=0.1, solver="saga")
            lr.fit(X[sel], Y[sel])
            probes[sc] = lr

        # ---- evaluate on split B ----
        our_img, our_pix, sup_img, sup_pix = [], [], [], []
        our_pixap, sup_pixap = [], []
        for i in idxB:
            s = test_ds[i]
            img = s["image"].unsqueeze(0).to(DEVICE)
            mask = s["mask"].numpy().astype(np.float32).reshape(-1)
            lab = int(s["label"])
            with torch.no_grad():
                out = method.predict(img)
            amap = out["anomaly_map"][0, 0].cpu().numpy().reshape(-1)
            iscore = float(out["image_score"][0])
            our_img.append((iscore, lab))
            our_pix.append((amap, mask)); our_pixap.append((amap, mask))
            # probe maps
            with torch.no_grad():
                feats = method.extract(img)
            pmaps = []
            for sc in scales:
                if probes[sc] is None:
                    continue
                f = feats[sc][0].permute(1, 2, 0).reshape(-1, feats[sc].shape[1]).cpu().numpy().astype(np.float32)
                H, W = feats[sc].shape[2], feats[sc].shape[3]
                p = probes[sc].predict_proba(f)[:, 1].reshape(H, W)
                p256 = cv2_up(p, (map_size, map_size))
                pmaps.append(p256)
            if pmaps:
                pmap = np.mean(pmaps, 0).reshape(-1)
                sup_img.append((float(topq_score(torch.from_numpy(pmap).reshape(1, 1, map_size, map_size))[0]), lab))
                sup_pix.append((pmap, mask)); sup_pixap.append((pmap, mask))

        def au(list_tuples):
            s = np.concatenate([np.asarray(x[0]).ravel() for x in list_tuples])
            l = np.concatenate([np.asarray(x[1]).ravel() for x in list_tuples])
            return float(roc_auc_score(l, s)) if len(np.unique(l)) > 1 else float("nan")
        def ap(list_tuples):
            s = np.concatenate([np.asarray(x[0]).ravel() for x in list_tuples])
            l = np.concatenate([np.asarray(x[1]).ravel() for x in list_tuples])
            return float(average_precision_score(l, s))

        res = dict(
            category=cat,
            ours_image_auroc=au(our_img), ours_pixel_auroc=au(our_pix), ours_pixel_ap=ap(our_pix),
            sup_image_auroc=au(sup_img), sup_pixel_auroc=au(sup_pix), sup_pixel_ap=ap(sup_pix),
            n_test_splitB=len(idxB),
        )
        results[cat] = res
        print(f"[{ci+1}/{len(cats)}] {cat}: ours I/px/pxAP={res['ours_image_auroc']:.4f}/{res['ours_pixel_auroc']:.4f}/{res['ours_pixel_ap']:.4f}"
              f"  sup I/px/pxAP={res['sup_image_auroc']:.4f}/{res['sup_pixel_auroc']:.4f}/{res['sup_pixel_ap']:.4f}", flush=True)
        json.dump(results, open(OUT, "w"), indent=2)
    print(f"done in {time.time()-t0:.0f}s -> {OUT}")

def cv2_down(m, sz):
    import cv2
    return cv2.resize(m, sz, interpolation=cv2.INTER_NEAREST)
def cv2_up(m, sz):
    import cv2
    return cv2.resize(m, sz, interpolation=cv2.INTER_LINEAR)

if __name__ == "__main__":
    main()
