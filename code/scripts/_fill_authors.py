# -*- coding: utf-8 -*-
"""Fill {{Authors}} placeholders in references.bib with real author names
fetched from OpenAlex by DOI. Also validates year/venue when available."""
import re, json, time, urllib.request, sys

BIB = r"d:\cvs\pythoncode\AnomalyDetection\paper\references.bib"
TODO = r"d:\cvs\pythoncode\AnomalyDetection\paper\_todo_authors.txt"

def fetch(doi):
    u = "https://api.openalex.org/works/https://doi.org/%s?select=authorships,publication_year" % doi
    req = urllib.request.Request(u, headers={"User-Agent": "bib-fill/1.0 (mailto:tangjun@xhsysu.edu.cn)"})
    for attempt in range(3):
        try:
            return json.load(urllib.request.urlopen(req, timeout=30))
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))

def fmt(names):
    out = []
    for n in names:
        n = " ".join(n.split())
        if "," in n:  # already "Last, First"
            out.append(n); continue
        parts = n.split(" ")
        if len(parts) == 1:
            out.append(n)
        else:
            out.append("%s, %s" % (parts[-1], " ".join(parts[:-1])))
    return " and ".join(out)

lines = [l.strip() for l in open(TODO, encoding="utf-8") if l.strip()]
bib = open(BIB, encoding="utf-8").read()
report = []
for line in lines:
    key, doi, title = line.split("|", 2)
    try:
        d = fetch(doi)
        names = [a.get("author", {}).get("display_name", "") for a in d.get("authorships", [])]
        names = [n for n in names if n]
        if not names:
            report.append("FAIL no-authorships %s (%s)" % (key, doi)); continue
        astr = fmt(names)
        # replace the placeholder inside the entry block for this key
        pat = re.compile(r"(@\w+\{%s,.*?author\s*=\s*)\{\{Authors\}\}" % re.escape(key), re.S)
        if not pat.search(bib):
            report.append("FAIL no-placeholder %s" % key); continue
        bib = pat.sub(lambda m: m.group(1) + "{" + astr + "}", bib, count=1)
        report.append("OK   %-22s %s | year=%s" % (key, astr[:80], d.get("publication_year")))
    except Exception as e:
        report.append("FAIL %s (%s): %s" % (key, doi, type(e).__name__))
    time.sleep(0.4)

open(BIB, "w", encoding="utf-8").write(bib)
open(r"d:\cvs\pythoncode\AnomalyDetection\paper\_authors_report.txt", "w", encoding="utf-8").write("\n".join(report))
print("done; report in _authors_report.txt")
