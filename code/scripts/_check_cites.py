"""校验：tex 中所有 \cite key 是否都在 bib 中；并统计 bib 年份分布。"""
import re, glob, collections

tex = ""
for f in glob.glob("sections/*.tex") + ["main.tex"]:
    tex += open(f, encoding="utf-8").read()
cited = set()
for m in re.finditer(r"\\cite\{([^}]+)\}", tex):
    for k in m.group(1).split(","):
        cited.add(k.strip())

bib = open("references.bib", encoding="utf-8").read()
keys = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib))

dangling = sorted(cited - keys)
unused = sorted(keys - cited)
print("cited unique keys:", len(cited))
print("bib entries:", len(keys))
print("DANGLING (cited but not in bib):", dangling if dangling else "none")
print("unused (in bib but not cited):", unused if unused else "none")

ys = collections.Counter(re.findall(r"year\s*=\s*\{(\d{4})\}", bib))
print("year distribution:", dict(sorted(ys.items())))
n_recent = sum(c for y, c in ys.items() if int(y) >= 2023)
print(f"refs >= 2023: {n_recent}/{len(keys)}")