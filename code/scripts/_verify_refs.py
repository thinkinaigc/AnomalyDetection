# -*- coding: utf-8 -*-
"""Verify every bib entry against OpenAlex by title match (DOI preferred)."""
from __future__ import annotations
import json, os, re, sys, time, urllib.parse, urllib.request

BIB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "paper", "references.bib")
MAIL = "mailto:tangjun@xhsysu.edu.cn"


def openalex(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": f"AnomalyDetection-refcheck ({MAIL})"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return json.load(r)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def main():
    only = set(sys.argv[1:])
    txt = open(BIB, encoding="utf-8").read()
    entries = re.findall(r"@(\w+)\{([^,]+),(.*?)\n\}", txt, flags=re.S)
    print(f"{len(entries)} bib entries parsed\n")
    bad = []
    for etype, key, body in entries:
        if only and key not in only:
            continue
        def f(name):
            m = re.search(name + r"\s*=\s*\{(.*?)\}\s*,?\s*\n", body, flags=re.S)
            return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""
        title, doi, journal, year = f("title"), f("doi"), f("journal"), f("year")
        # ---- 1) DOI must resolve ----
        doi_ok, doi_note = False, ""
        if doi:
            try:
                j = openalex(f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi)}")
                doi_ok = True
                doi_note = j.get("title", "")[:60]
            except Exception as e:
                doi_note = f"DOI not in OpenAlex ({type(e).__name__})"
        # ---- 2) title must exist somewhere in OpenAlex ----
        title_ok, best = False, ""
        try:
            q = urllib.parse.quote(title)
            j = openalex(f"https://api.openalex.org/works?search={q}&per-page=5")
            nt = norm(title)
            for w in j.get("results", []):
                nt2 = norm(w.get("title") or "")
                if nt2 and (nt2 == nt or nt in nt2 or nt2 in nt):
                    title_ok = True
                    best = f"{w.get('title','')[:55]} | {w.get('publication_year')} | {(w.get('primary_location') or {}).get('source',{}) and ((w.get('primary_location') or {}).get('source') or {}).get('display_name')}"
                    break
        except Exception as e:
            best = f"search failed {type(e).__name__}"
        flag = "OK " if (doi_ok and title_ok) else "!! "
        if not (doi_ok and title_ok):
            bad.append(key)
        print(f"{flag}{key:24s} doi={'Y' if doi_ok else 'N'} title={'Y' if title_ok else 'N'}  {title[:52]}")
        if not doi_ok:
            print(f"      doi note: {doi_note}")
        if not title_ok:
            print(f"      title not found in OpenAlex: {best}")
        time.sleep(0.15)
    print("\nSUSPECT ENTRIES:", bad if bad else "none")


if __name__ == "__main__":
    main()
