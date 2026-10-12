# -*- coding: utf-8 -*-
"""Geometry validator for the framework figure.

Checks (1) text overflow: every text bbox must sit fully inside the smallest
module box containing its centre; (2) sibling box overlap; (3) gutter labels
must not spill onto a panel border.  Reads paper/figures/framework_geo.json.
"""
import json, sys

GEO = r"d:\cvs\pythoncode\AnomalyDetection\paper\figures\framework_geo.json"
g = json.load(open(GEO, encoding="utf-8"))
boxes = g["boxes"]
TOL = 1.0

def inside(t, b, tol=TOL):
    x, y, w, h = b[0], b[1], b[2], b[3]
    return (t["x0"] >= x - tol and t["x1"] <= x + w + tol and
            t["y0"] >= y - tol and t["y1"] <= y + h + tol)

def contains_center(t, b):
    cx, cy = (t["x0"] + t["x1"]) / 2, (t["y0"] + t["y1"]) / 2
    return b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3]

# ---- 1. text overflow -------------------------------------------------
bad = []
for t in g["texts"]:
    cands = [(k, b) for k, b in boxes.items() if contains_center(t, b)]
    if not cands:
        continue
    bid, smallest = min(cands, key=lambda kb: kb[1][2] * kb[1][3])
    if not inside(t, smallest):
        bad.append((t["t"][:34], bid,
                    round(t["x1"] - (smallest[0] + smallest[2]), 1),
                    round((smallest[1] + smallest[3]) - t["y1"], 1)))

print("=" * 72)
print("1. TEXT OVERFLOW (expect none)")
print("=" * 72)
print("  none" if not bad else "")
for t, bid, dx, dy in bad:
    print(f"  [{bid}] {t!r:38s} right+{dx:6.1f}px  bottom+{dy:6.1f}px")

# ---- 2. sibling box overlap ------------------------------------------
PARENTS = {"M05": {"M05a", "M05b", "M05c", "M05d"}}
ov = []
keys = list(boxes)
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        a, b = keys[i], keys[j]
        if a in PARENTS.get(b, ()) or b in PARENTS.get(a, ()):
            continue
        ax, ay, aw, ah = boxes[a][:4]
        bx, by, bw, bh = boxes[b][:4]
        ox = min(ax + aw, bx + bw) - max(ax, bx)
        oy = min(ay + ah, by + bh) - max(ay, by)
        if ox > TOL and oy > TOL:
            ov.append((a, b, round(ox, 1), round(oy, 1)))

print("=" * 72)
print("2. BOX OVERLAP (expect none)")
print("=" * 72)
print("  none" if not ov else "")
for a, b, ox, oy in ov:
    print(f"  {a} <-> {b}: {ox} x {oy} px")

# ---- 3. gutter label spill -------------------------------------------
GUT = [(590, 700), (1260, 1370)]
gut_bad = []
for t in g["texts"]:
    for gx0, gx1 in GUT:
        if t["x1"] > gx0 and t["x0"] < gx1 and (t["x0"] < gx0 - 0.5 or t["x1"] > gx1 + 0.5):
            gut_bad.append((t["t"][:30], round(t["x0"], 1), round(t["x1"], 1)))
            break

print("=" * 72)
print("3. GUTTER LABEL SPILL (expect none)")
print("=" * 72)
print("  none" if not gut_bad else "")
for t, x0, x1 in gut_bad:
    print(f"  {t!r:34s} x=[{x0}, {x1}]")

print("=" * 72)
print("TOTAL VIOLATIONS:", len(bad) + len(ov) + len(gut_bad))
