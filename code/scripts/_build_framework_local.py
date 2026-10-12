# -*- coding: utf-8 -*-
"""FGSC framework figure — landscape 3-col x 2-row SNAKE layout (v8, 2026-10-10)

Row1 (L->R): Input | Backbone (Frozen) | Normal Reference
Row2 (R->L): Scoring & Fusion | PSC | Output
Main flow = adjacent-panel arrows; column/row gutters widened to 110 px so every
edge label sits inside an empty gutter (never covers a panel border).
Canvas 1960x1450 -> placed at width=\textwidth, upright.
Input/Output panels are 80% of row height, vertically centred.
Fonts scaled by FS inside txt()/rich() so formula offsets stay correct.
"""
import math, os, base64, json
from PIL import ImageFont

W, H = 1960, 1450
FS = 1.6
D = r"C:\Users\tiger\WorkBuddy\Research\scientific_figure_output"
OUTDIR = r"d:\cvs\pythoncode\AnomalyDetection\paper\figures"

INK, SUB, MENU = "#12263A", "#4B5F73", "#33475B"
BLUE, BLUE_F = "#2E6CA5", "#EAF2FB"
SLAB, SLAB_F = "#7FA8CE", "#F3F8FD"
GREEN, GREEN_F = "#1F8A4C", "#EAF7EF"
ORANGE, ORANGE_F, ORANGE_S = "#D97706", "#FDF2E6", "#C2410C"
VIOLET = "#7C4DC4"
DASH, WHITE, LINE, TBAR = "#CBD5E1", "#FFFFFF", "#2C4A6E", "#3D5E8B"
GRAY, GRAY_F, GRAY_S = "#94A3B8", "#F1F5F9", "#64748B"
FONT = "Arial, Helvetica, sans-serif"

_MEAS = 256
_FC = {}
def _mfont(bold):
    if bold not in _FC:
        p = "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"
        _FC[bold] = ImageFont.truetype(p, _MEAS)
    return _FC[bold]

def tw(s, size, bold=False):
    return _mfont(bold).getlength(s) * size / _MEAS

def _isbold(w):
    return str(w) in ("600", "700", "800", "900", "bold") or \
           (isinstance(w, (int, float)) and w >= 600)

def halo_w(size):
    """Thin outline: wide halos swallow small glyphs."""
    return min(3.2, max(1.6, size * 0.14))

out = []
def add(s): out.append(s)

def rect(x, y, w, h, fill=WHITE, stroke=None, sw=2, rx=12, dash=None, shadow=True):
    a = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" fill="{fill}"'
    if stroke: a += f' stroke="{stroke}" stroke-width="{sw}"'
    if dash:   a += f' stroke-dasharray="{dash}"'
    if shadow: a += ' filter="url(#sh)"'
    add(a + '/>')

def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

TEXTS = []
def txt(x, y, s, size=20, fill=INK, weight="400", anchor="middle", halo=None,
        reg=False, sw=None, rot=None):
    size = size * FS
    tr = f' transform="rotate({rot} {x} {y})"' if rot else ''
    common = (f'xml:space="preserve" x="{x}" y="{y}" font-family="{FONT}" font-size="{size:.2f}" '
              f'font-weight="{weight}" text-anchor="{anchor}"')
    if halo:
        # two-pass: cairosvg ignores paint-order, so a single element would paint
        # the outline ON TOP of the fill and eat the glyph.
        add(f'<text {common} fill="none" stroke="{halo}" stroke-width="{halo_w(size):.2f}" '
            f'stroke-linejoin="round"{tr}>{_esc(s)}</text>')
        add(f'<text {common} fill="{fill}"{tr}>{_esc(s)}</text>')
    elif sw:
        add(f'<text {common} fill="{fill}" paint-order="stroke" stroke="{fill}" '
            f'stroke-width="{sw}" stroke-linejoin="round"{tr}>{_esc(s)}</text>')
    else:
        add(f'<text {common} fill="{fill}"{tr}>{_esc(s)}</text>')
    w = tw(s, size, _isbold(weight))
    x0 = x - w / 2 if anchor == "middle" else (x - w if anchor == "end" else x)
    TEXTS.append({"t": s, "x0": round(x0, 1), "y0": round(y - size * 0.78, 1),
                  "x1": round(x0 + w, 1), "y1": round(y + size * 0.22, 1)})
    if reg:
        lbl(x, y, s, size, anchor)

SUBR, SHR = 0.68, 0.32
FORMULAS = []
def rich(x, y, segs, size=20, fill=INK, weight="400", anchor="middle", halo=None, reg=False):
    size = size * FS
    bold = _isbold(weight)
    parts = []
    for t, k in segs:
        fs = size if k == 'n' else size * SUBR
        parts.append((t, k, fs, tw(t, fs, bold)))
    total = sum(p[3] for p in parts)
    if anchor == "middle":   cx = x - total / 2
    elif anchor == "end":    cx = x - total
    else:                    cx = x
    meta = []
    for t, k, fs, w in parts:
        dy = 0.0 if k == 'n' else (SHR * size if k == 'sub' else -SHR * size * 0.92)
        meta.append({"t": t, "k": k, "cx": cx + w / 2, "by": y + dy, "fs": fs, "w": w})
        cx += w
    FORMULAS.append({"size": size, "base_by": y, "pieces": meta, "label": bool(reg)})
    if meta:
        TEXTS.append({"t": "".join(t for t, _ in segs)[:28], "x0": round(x - total / 2, 1),
                      "y0": round(y - size * 0.95, 1), "x1": round(x + total / 2, 1),
                      "y1": round(y + size * 0.30, 1)})

    def emit(stroke_fill, stroke_w, paint_order):
        nonlocal cx
        cx = x - total / 2 if anchor == "middle" else (x - total if anchor == "end" else x)
        s = ""
        for t, k, fs, w in parts:
            dy = 0.0 if k == 'n' else (SHR * size if k == 'sub' else -SHR * size * 0.92)
            extra = (f' paint-order="{paint_order}" stroke="{stroke_fill}" stroke-width="{stroke_w}"'
                     if stroke_w else '')
            s += (f'<text xml:space="preserve" x="{cx:.2f}" y="{y + dy:.2f}" font-family="{FONT}" font-size="{fs:.2f}" '
                  f'font-weight="{weight}" fill="{stroke_fill if stroke_w else fill}" '
                  f'text-anchor="start"{extra}>{_esc(t)}</text>')
            cx += w
        return s

    add(emit(halo, halo_w(size), "stroke") if halo else emit(None, 0, None))
    add(emit(None, 0, None))
    if reg:
        lbl(x, y, "".join(t for t, _ in segs), size, anchor)

def photo(b64s, x, y, w, h, rx=10, cid="clipA", stroke=BLUE):
    add(f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}"/></clipPath>')
    add(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice" '
        f'clip-path="url(#{cid})" href="data:image/png;base64,{b64s}"/>')
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" fill="none" '
        f'stroke="{stroke}" stroke-width="2"/>')
    PHOTOS.append([float(x), float(y), float(w), float(h)])

PHOTOS = []
BOXES, CONNS, LABELS = {}, {}, []
def mod(bid, x, y, w, h, fill, stroke, sw, rx, dash=None):
    rect(x, y, w, h, fill=fill, stroke=stroke, sw=sw, rx=rx, dash=dash)
    BOXES[bid] = [float(x), float(y), float(w), float(h), fill]

def lbl(x, y, s, size, anchor="start"):
    w = len(s) * size * 0.56
    x0 = x - w/2 if anchor == "middle" else (x - w if anchor == "end" else x)
    LABELS.append([round(x0, 1), round(y - size*0.80, 1), round(x0+w, 1), round(y + size*0.22, 1), s])

HEAD_L, HEAD_W, LINE_W = 16.0, 16.0, 4.4
def arrow(pts, cid, nrm, lw=None, color=None):
    lw = lw or LINE_W
    scol = color or LINE
    tip, prev = pts[-1], pts[-2]
    dx, dy = tip[0]-prev[0], tip[1]-prev[1]
    n = math.hypot(dx, dy) or 1
    ux, uy = dx/n, dy/n
    bx, by = tip[0]-ux*HEAD_L, tip[1]-uy*HEAD_L
    shaft = list(pts[:-1]) + [(bx, by)]
    d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(shaft))
    add(f'<path d="{d}" fill="none" stroke="{scol}" stroke-width="{lw}" '
        f'stroke-linejoin="round" stroke-linecap="butt"/>')
    px, py = -uy, ux
    hw = HEAD_W/2
    poly = [(tip[0], tip[1]), (bx+px*hw, by+py*hw), (bx-px*hw, by-py*hw)]
    add('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in poly) + f'" fill="{scol}"/>')
    CONNS[cid] = {"pts": [[float(x), float(y)] for x, y in pts],
                  "tip": [float(tip[0]), float(tip[1])], "nrm": [float(nrm[0]), float(nrm[1])]}

def b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

EL_RAW = b64(os.path.join(D, "el_thumb.png"))
EL_NRM = b64(os.path.join(D, "el_norm.png"))
OURS_HEAT = b64(os.path.join(D, "ours_heat.png"))

# ================= canvas =================
add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
add('''<defs>
<filter id="sh" x="-25%" y="-25%" width="150%" height="160%">
  <feDropShadow dx="0" dy="3.5" stdDeviation="4.5" flood-color="#1F3A56" flood-opacity="0.20"/>
</filter>
</defs>''')
add(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>')

# columns: col1 30..590 | gutter 110 | col2 700..1260 | gutter 110 | col3 1370..1930
# rows:    row1 30..670  | gutter 110 | row2 780..1420
panels = [("R01", "Input", 30, 94, 560, 512),
          ("R02", "Backbone (Frozen)", 700, 30, 560, 640),
          ("R03", "Normal Reference", 1370, 30, 560, 640),
          ("R04", "Scoring & Multi-scale Fusion", 1370, 780, 560, 640),
          ("R05", "PSC", 700, 780, 560, 640),
          ("R06", "Output", 30, 844, 560, 512)]
TBH = 40
P = {}
FLOW = {"R01": 1, "R02": 2, "R03": 3, "R04": 4, "R05": 5, "R06": 6}
for rid, title, x0, y0, w, h in panels:
    P[rid] = (x0, x0 + w)
    hi = (rid == "R05")
    rect(x0, y0, w, h, fill=("#FFFBF3" if hi else WHITE),
         stroke=(ORANGE if hi else DASH), sw=(2.4 if hi else 1.5), rx=22)
    if hi:
        rect(x0+7, y0+7, w-14, h-14, fill="none", stroke=ORANGE_S, sw=1.1, rx=17,
             dash="9 6", shadow=False)
    rect(x0+6, y0+6, w-12, TBH, fill=(ORANGE_S if hi else TBAR), stroke=None, rx=12, shadow=False)
    tsz = 18 if len(title) > 22 else 22
    fsz = tsz * FS
    # stage number: identical font/size/weight as the title; circle hugs the digit
    cy = y0 + 6 + TBH / 2
    dh = tw("8", fsz, True)                    # digit advance width at title size
    r = dh / 2 + 0.17 * fsz                    # circle around the digit
    cx = x0 + w / 2 - tw(title, fsz, True) / 2 - 0.34 * fsz - r
    add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="none" '
        f'stroke="{WHITE}" stroke-width="{0.055 * fsz:.2f}"/>')
    txt(cx, cy + 0.35 * fsz, str(FLOW[rid]), size=tsz, fill=WHITE, weight="700")
    txt(x0+w/2, y0+6+28, title, size=tsz, fill=WHITE, weight="700")

# ================= R01 Input (80% height, vertically centred) =================
mod("M01", 60, 205, 240, 330, BLUE_F, BLUE, 2.2, 14)
txt(180, 247, "Input image x", size=17, weight="700")
photo(EL_RAW, 80, 265, 200, 200, cid="clipIn")
txt(180, 497, "raw industrial image", size=13, fill=SUB)
mod("M02", 325, 205, 240, 330, BLUE_F, BLUE, 2.2, 14)
txt(445, 247, "ImageNet normalisation", size=10.5, weight="700")
photo(EL_NRM, 345, 265, 200, 200, cid="clipNm")
txt(445, 497, "mean/std standardised", size=13, fill=SUB)
arrow([(300, 370), (325, 370)], "C1", (1, 0))

# ================= R02 Backbone =================
mod("M03", 740, 105, 470, 150, BLUE_F, BLUE, 2.6, 14)
txt(975, 155, "DINOv2 ViT-B/14", size=22, weight="700")
txt(975, 195, "frozen backbone · no gradient", size=15, fill=SUB)
txt(975, 228, "multi-scale patch features (P3, P4)", size=15, fill=SUB)
for cx_, tag, res, grid in ((857, "P3", "392", "28 × 28"), (1117, "P4", "224", "16 × 16")):
    bid = "S0" + ("1" if tag == "P3" else "2")
    mod(bid, cx_-112, 300, 225, 200, SLAB_F, SLAB, 2, 12)
    txt(cx_, 340, tag, size=17, weight="700")
    txt(cx_, 378, f"resize → {res} px", size=13, fill=MENU)
    txt(cx_, 414, "patch stride 14", size=13, fill=MENU)
    txt(cx_, 450, f"→ {grid} tokens", size=14, fill=MENU)
    txt(cx_, 484, "C = 768 channels", size=13, fill=SUB)
    arrow([(cx_, 255), (cx_, 300)], "C3" if tag == "P3" else "C4", (0, 1))
    arrow([(cx_, 500), (cx_, 545)], "C5" if tag == "P3" else "C6", (0, 1))
mod("M04", 740, 545, 490, 105, BLUE_F, BLUE, 2.6, 14)
txt(985, 592, "Fixed random 1×1 projection", size=20, weight="700")
txt(985, 628, "d = 256 · one frozen basis, never trained", size=14, fill=SUB)

# ================= R03 Normal Reference =================
mod("M05", 1410, 105, 470, 530, GREEN_F, GREEN, 2.6, 16)
txt(1645, 152, "Position-wise Gaussian reference", size=15, weight="700")
mod("M05a", 1430, 195, 430, 100, WHITE, SLAB, 1.5, 10)
rich(1645, 250, [("μ","n"),("k","sub"),(" = E[φ(x)","n"),("k","sub"),("]","n")], size=20, weight="600")
txt(1645, 285, "per-position normal mean", size=13, fill=SUB)
mod("M05b", 1430, 312, 430, 100, WHITE, SLAB, 1.5, 10)
rich(1645, 367, [("Σ","n"),("k","sub"),(" = diag(σ","n"),("2","sup"),("k","sub"),(")","n")], size=20, weight="600")
txt(1645, 402, "shrinkage λ = 0.02", size=13, fill=SUB)
mod("M05c", 1430, 429, 430, 100, WHITE, SLAB, 1.5, 10)
rich(1645, 484, [("ref_smooth(μ","n"),("k","sub"),(")","n")], size=20, weight="600")
txt(1645, 519, "spatial smoothing (off)", size=13, fill=SUB)
mod("M05d", 1430, 546, 430, 72, WHITE, GREEN, 1.5, 10, dash="8 5")
txt(1645, 578, "normal (defect-free) images only", size=14, fill=GREEN)
txt(1645, 608, "no labels, no gradients used", size=14, fill=GREEN)

# ================= R04 Scoring & Fusion =================
mod("M06", 1410, 855, 470, 305, BLUE_F, BLUE, 2.6, 14)
txt(1645, 907, "Diagonal Mahalanobis distance", size=19, weight="700")
rich(1645, 995, [("d","n"),("k","sub"),(" = (φ","n"),("k","sub"),(" − μ","n"),("k","sub"),
                 (")","n"),("T","sup"),(" Σ","n"),("k","sub"),("−1","sup"),(" (φ","n"),("k","sub"),
                 (" − μ","n"),("k","sub"),(")","n")], size=16, weight="600")
txt(1645, 1050, "per-position standardisation", size=14, fill=SUB)
txt(1645, 1085, "sigmoid → score in (0, 1)", size=14, fill=SUB)
txt(1645, 1120, "diag Σ_k from the normal reference", size=14, fill=SUB)
arrow([(1645, 1160), (1645, 1210)], "C9", (0, 1))
txt(1665, 1190, "d_k · per scale", size=13, fill=MENU, weight="600", anchor="start", halo=WHITE, reg=True)
mod("M07", 1410, 1210, 470, 180, BLUE_F, BLUE, 2.6, 14)
txt(1645, 1262, "Multi-scale mean fusion", size=20, weight="700")
rich(1645, 1335, [("ẑ","n"),(" = ( Up(ẑ","n"),("P3","sub"),(") + Up(ẑ","n"),("P4","sub"),
                  (") ) / 2","n")], size=17, weight="600")
txt(1645, 1370, "resample → 28×28 · equal weights 1/2 + 1/2", size=14, fill=SUB)

# ================= R05 PSC — post-hoc module: orange = activated, grey dashed = disabled branch =================
mod("M08", 740, 855, 470, 145, GRAY_F, GRAY, 2, 12, dash="8 5")
txt(975, 907, "Texture energy T", size=20, weight="700", fill=GRAY_S)
txt(975, 947, "Haar high-frequency energy", size=14, fill=SUB)
txt(975, 982, "of the original input image", size=14, fill=SUB)
arrow([(975, 1000), (975, 1040)], "C12", (0, 1), color=GRAY)
txt(992, 1025, "T", size=14, fill=MENU, weight="600", anchor="start", halo=WHITE, reg=True)
mod("M09", 740, 1040, 470, 150, GRAY_F, GRAY, 2, 12, dash="8 5")
rich(975, 1100, [("α","n"),(" = clip(a","n"),("0","sub"),(" + a","n"),("1","sub"),
                 (" · T, 0, 1)","n")], size=18, weight="600")
rich(975, 1140, [("a","n"),("0","sub"),(" = 0.6,   a","n"),("1","sub"),(" = 0 → gate OFF","n")],
     size=15, fill=GRAY_S)
arrow([(975, 1190), (975, 1230)], "C13", (0, 1), color=GRAY)
rich(992, 1215, [("α","n")], size=14, fill=MENU, anchor="start", weight="600", halo=WHITE, reg=True)
mod("M10", 740, 1230, 470, 170, ORANGE_F, ORANGE, 2.2, 12)
rich(975, 1285, [("ẑ","n"),("cal","sub"),(" = (1 − α) · ẑ + α · Blur","n"),("K","sub"),
                 ("(ẑ)","n")], size=17, weight="600")
txt(975, 1322, "K = 33 · Gaussian low-pass (DEPLOYED)", size=14, fill=SUB)
txt(975, 1352, "α = a0 = 0.6; texture gate off (a1 = 0)", size=13, fill=ORANGE_S)
txt(975, 1378, "fixed low-pass smoothing", size=13, fill=ORANGE_S)

# ================= R06 Output (80% height, vertically centred) =================
txt(310, 930, "Ours: anomaly map ẑ_cal", size=17, weight="700", fill=VIOLET, sw=1.3)
photo(OURS_HEAT, 180, 955, 260, 260, cid="clipOut", stroke=VIOLET)
txt(310, 1262, "Image-level score", size=17, weight="700")
txt(310, 1298, "top-q mean (q = 0.5%)", size=14, fill=SUB)
txt(310, 1328, "dense localisation", size=13, fill=SUB)

# ================= main flow (labels live inside the 110px gutters) =================
arrow([(590, 370), (700, 370)], "A_IB", (1, 0))
txt(645, 348, "normalised", size=12, fill=MENU, weight="600", anchor="middle", halo=WHITE, reg=True)
arrow([(1230, 597), (1370, 597)], "A_BN", (1, 0))
rich(1300, 578, [("φ","n"),("P3","sub"),(", φ","n"),("P4","sub")], size=12, fill=MENU,
     weight="600", anchor="middle", halo=WHITE, reg=True)
arrow([(1645, 670), (1645, 780)], "A_NS", (0, 1))
rich(1665, 730, [("μ","n"),("k","sub"),(", Σ","n"),("k","sub")], size=12, fill=MENU,
     weight="600", anchor="start", halo=WHITE, reg=True)
arrow([(1410, 1300), (1260, 1300)], "A_SP", (-1, 0))
txt(1315, 1282, "ẑ", size=14, fill=MENU, weight="600", anchor="middle", halo=WHITE, reg=True)
arrow([(700, 1000), (590, 1000)], "A_PO", (-1, 0))
rich(645, 982, [("ẑ","n"),("cal","sub")], size=12, fill=MENU, weight="600",
     anchor="middle", halo=WHITE, reg=True)

# ================= bypass: input image -> texture energy (row gutter) =================
arrow([(175, 606), (175, 725), (975, 725), (975, 780)], "C11", (0, -1), color=GRAY)
txt(450, 712, "input image", size=14, fill=MENU, weight="600", anchor="middle", halo=WHITE, reg=True)

add('</svg>')
svg = "\n".join(out)
p = os.path.join(OUTDIR, "framework_src.svg")
with open(p, "w", encoding="utf-8") as f:
    f.write(svg)

geo = {"canvas": [W, H], "panels": {r: list(v) for r, v in P.items()},
       "boxes": BOXES, "conns": CONNS, "photos": PHOTOS, "labels": LABELS,
       "texts": TEXTS, "line": LINE, "head_l": HEAD_L, "head_w": HEAD_W,
       "line_w": LINE_W, "formulas": FORMULAS, "shr": SHR, "subr": SUBR, "fs": FS}
with open(os.path.join(OUTDIR, "framework_geo.json"), "w", encoding="utf-8") as f:
    json.dump(geo, f, ensure_ascii=False, indent=1)

print("SVG written:", p, len(svg), "bytes")
print("box count:", len(BOXES), "| conn count:", len(CONNS), "| texts:", len(TEXTS))
