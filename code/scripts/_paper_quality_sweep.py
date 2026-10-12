# -*- coding: utf-8 -*-
"""Paper quality sweep: LaTeX hygiene, AI-flavour markers, citation integrity.

Run before packaging. Reports findings only; edits are made by hand afterwards.
"""
import os, re, glob, collections

PAPER = r"d:\cvs\pythoncode\AnomalyDetection\paper"
main = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
secs = sorted(glob.glob(os.path.join(PAPER, "sections", "*.tex")))
sec_text = {os.path.basename(p): open(p, encoding="utf-8").read() for p in secs}
sec_order = re.findall(r"\\(?:input|include)\{([^}]+)\}", main)
print("main.tex inputs:", sec_order)
print("section files on disk:", list(sec_text))
print("=" * 74)

def strip_tex(s):
    s = re.sub(r"(?m)%.*$", "", s)
    s = re.sub(r"\\begin\{(figure|table|equation\*?|align\*?)\}.*?\\end\{\1\}", " ", s, flags=re.S)
    s = re.sub(r"\\cite[tp]?\*?(\[[^\]]*\])*\{[^}]*\}", " CITE ", s)
    s = re.sub(r"\\(ref|autoref|eqref)\{[^}]*\}", " REF ", s)
    s = re.sub(r"\\label\{[^}]*\}", " ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", s)
    s = re.sub(r"[{}$&\\~^]", " ", s)
    return s

def words(s):
    return re.findall(r"[A-Za-z][A-Za-z'\-]*", s)

all_prose = {f: strip_tex(t) for f, t in sec_text.items()}
all_prose["main.tex"] = strip_tex(main)
wc = {f: len(words(t)) for f, t in all_prose.items()}
print("word count per file:", wc)
print("TOTAL words:", sum(wc.values()))
print("=" * 74)

FLAGGED = ["delve", "tapestry", "pivotal", "crucial", "foster", "showcase",
           "testament", "leverage", "realm", "embark", "underscore", "multifaceted",
           "nuanced", "comprehensive", "robust", "intricate", "cornerstone",
           "synergy", "holistic", "streamline", "cutting-edge", "groundbreaking",
           "landscape", "navigate"]
print("1. AI-FLAGGED TERMS")
hits = collections.Counter()
for f, t in all_prose.items():
    low = " " + t.lower() + " "
    for term in FLAGGED:
        for m in re.finditer(r"\b" + re.escape(term) + r"\w{0,3}\b", low):
            ctx = low[max(0, m.start()-70):m.end()+70].replace("\n", " ")
            hits[term] += 1
            print(f"   [{f}] {m.group(0).strip()!r} :: ...{ctx.strip()}...")
print("   none" if not hits else f"   -> {sum(hits.values())} occurrence(s): {dict(hits)}")
print("=" * 74)

THROAT = ["it is important to note", "it should be noted", "it is worth mentioning",
          "in order to", "in today's", "in the realm", "it goes without saying",
          "as a matter of fact", "when it comes to", "with that being said",
          "this section will", "the following paragraph", "we now turn our attention",
          "this serves as a testament", "at the end of the day", "in this section, we will"]
print("2. THROAT-CLEARING / META-COMMENTARY")
n2 = 0
for f, t in all_prose.items():
    low = " " + t.lower() + " "
    for p in THROAT:
        for m in re.finditer(re.escape(p), low):
            n2 += 1
            print(f"   [{f}] {p!r} :: ...{low[max(0,m.start()-60):m.end()+80].strip()}...")
print("   none" if not n2 else f"   -> {n2} occurrence(s)")
print("=" * 74)

em = sum(t.count("\u2014") for t in all_prose.values())
semi = sum(t.count(";") for t in all_prose.values())
tot_w = sum(wc.values())
print("3. PUNCTUATION")
print(f"   em dashes: {em}  (limit <=3, recommend 0-1)")
print(f"   semicolons: {semi}  (limit <=2/1000w -> {2*tot_w/1000:.1f} allowed)")
print("=" * 74)

print("4. BURSTINESS (5+ consecutive sentences within a 5-word band)")
def sentences(t):
    parts = re.split(r"(?<=[.!?])\s+", " ".join(t.split()))
    return [s for s in parts if len(words(s)) >= 4]
bad_runs = 0
for f, t in all_prose.items():
    ss = [len(words(s)) for s in sentences(t)]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and max(ss[i:j+1] + [ss[j+1]]) - min(ss[i:j+1] + [ss[j+1]]) <= 5:
            j += 1
        if j - i + 1 >= 5:
            bad_runs += 1
            print(f"   [{f}] {j-i+1} sentences within {min(ss[i:j+1])}-{max(ss[i:j+1])} words")
        i = max(j + 1, i + 1)
print("   none" if not bad_runs else f"   -> {bad_runs} monotonous run(s)")
print("=" * 74)

print("5. PARAGRAPH LENGTH SPREAD")
for f, t in all_prose.items():
    paras = [p for p in t.split("\n\n") if len(words(p)) >= 25]
    if len(paras) >= 4:
        ls = [len(words(p)) for p in paras]
        spread = (max(ls) - min(ls)) / max(1, sum(ls) / len(ls))
        flag = "  <-- UNIFORM" if spread < 0.45 else ""
        print(f"   [{f}] {len(paras)} paras, min {min(ls)} / mean {sum(ls)//len(ls)} / max {max(ls)}, spread {spread:.2f}{flag}")
print("=" * 74)

print("6. CITATION INTEGRITY")
bib = open(os.path.join(PAPER, "references.bib"), encoding="utf-8").read()
bib_keys = set(re.findall(r"(?m)^@\w+\s*\{\s*([^,\s]+)\s*,", bib))
entries = re.findall(r"(?m)^@\w+\s*\{\s*([^,\s]+)\s*,((?:.|\n)*?)(?=^@\w|\Z)", bib)
bib_nodoi = [k for k, body in entries if "doi" not in body.lower()]
cited, per_file = set(), collections.Counter()
for f, raw in list(sec_text.items()) + [("main.tex", main)]:
    for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", raw):
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                cited.add(k)
                per_file[f] += 1
print("   bib entries:", len(bib_keys), "| distinct cited:", len(cited))
print("   uncited bib entries (bloat):", sorted(bib_keys - cited) or "none")
print("   cited but MISSING from bib (FATAL):", sorted(cited - bib_keys) or "none")
print("   entries without DOI:", len(bib_nodoi), bib_nodoi[:12])
print("   citations per section:", dict(per_file))
print("=" * 74)

print("7. LABEL / REF INTEGRITY")
allraw = main + "".join(sec_text.values())
labels = set(re.findall(r"\\label\{([^}]+)\}", allraw))
refs = set(re.findall(r"\\(?:ref|autoref|eqref)\{([^}]+)\}", allraw))
print("   labels:", len(labels), "| refs:", len(refs))
print("   dangling refs:", sorted(refs - labels) or "none")
print("   unused labels:", sorted(labels - refs) or "none")
print("=" * 74)

print("8. LATEX HYGIENE")
for f, raw in [("main.tex", main)] + list(sec_text.items()):
    if not raw.endswith("\n"):
        print(f"   [{f}] missing trailing newline")
print("   double spaces:", sum(len(re.findall(r"[a-z]  +[a-z]", t)) for t in all_prose.values()))
print("   'et al.' in prose:", sum(t.count("et al.") for t in all_prose.values()))
print("   figure w/o \\label:",
      sum(1 for m in re.finditer(r"\\begin\{figure\}.*?\\end\{figure\}", allraw, re.S)
          if "\\label" not in m.group(0)))
print("=" * 74)
