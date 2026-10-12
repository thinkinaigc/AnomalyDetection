"""从 references.bib 移除指定 key 的完整条目（按 @type{key, ... 平衡大括号）。"""
import re

DROP = ["generalizing2026clip","aspclip2025","fpclip2025","gpspclip2025",
        "prompting2025eswa","hplclip2026","kanoclip2025","diclip2025",
        "adaptclip2025","sowa2024","entroAD2026","ijcai2025vlm",
        "pad2023","imiad2023","rad2024","capsvit2026","pico2026"]

path = "references.bib"
t = open(path, encoding="utf-8").read()

def entry_span(text, key):
    m = re.search(r"@\w+\{" + re.escape(key) + r"\s*,", text)
    if not m:
        return None
    i = m.start()
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return (i, j + 1)
    return None

removed = []
for k in DROP:
    span = entry_span(t, k)
    if span:
        s, e = span
        while e < len(t) and t[e] in "\r\n":
            e += 1
        t = t[:s] + t[e:]
        removed.append(k)

open(path, "w", encoding="utf-8").write(t)
print("removed:", len(removed), removed)
print("remaining entries:", len(re.findall(r"@\w+\{", t)))