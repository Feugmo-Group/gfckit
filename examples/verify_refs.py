"""Verify a bibliography against Crossref, field by field.

A completeness check -- does every entry have a volume, a year, a page range --
passes an entry that is complete and wrong. The failure that actually occurred in
this project was of that kind: a citation correct in subject, era and research
group, but pointing at a glass-ceramic paper when the measurement was on glasses.
Only comparison against the record catches that.

For each entry this queries Crossref on the title, then compares the returned
journal, volume, first page and year with what the bibliography claims, and
reports a DOI. Titles are matched on a token overlap rather than exactly, since
bibliographies abbreviate and Crossref does not.

Verdicts:
  OK        every field present in the entry agrees with Crossref
  CHECK     Crossref found a close title but one or more fields disagree
  NO-MATCH  no candidate with a convincing title overlap; may be a book,
            dataset, thesis or preprint, which Crossref indexes unevenly

Usage:
    python examples/verify_refs.py <file.bib|file.tex> [--limit N]
"""
import json
import re
import subprocess
import sys
import time
import urllib.parse

MAILTO = "giresse.feugmo@gmail.com"        # Crossref polite pool
STOP = {"the", "a", "an", "of", "on", "in", "for", "and", "to", "with", "by",
        "from", "at", "as", "is", "are", "its", "their"}


def first_page(p):
    """Leading page number, so that '373--386' and '373' compare equal."""
    m = re.match(r"\s*([A-Za-z]?\d+)", (p or "").replace("--", "-"))
    return m.group(1) if m else ""


def tokens(s):
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return {w for w in s.split() if w not in STOP and len(w) > 2}


def parse_bib(text):
    """BibTeX -> [(key, title, journal, volume, page, year)]."""
    out = []
    for m in re.finditer(r"@\w+\s*\{\s*([^,]+),(.*?)\n\}", text, re.S):
        key, body = m.group(1).strip(), m.group(2)
        f = {}
        for fm in re.finditer(r"(\w+)\s*=\s*[{\"](.+?)[}\"]\s*,?\s*\n", body, re.S):
            f[fm.group(1).lower()] = " ".join(fm.group(2).split())
        out.append((key, f.get("title", ""), f.get("journal", ""),
                    f.get("volume", ""), f.get("pages", ""), f.get("year", "")))
    return out


def parse_bibitem(text):
    """LaTeX \\bibitem -> the same tuple, parsed loosely."""
    out = []
    body = text[text.index("\\begin{thebibliography}"):] \
        if "\\begin{thebibliography}" in text else text
    for chunk in re.split(r"\\bibitem\{", body)[1:]:
        key = chunk[:chunk.index("}")]
        rest = " ".join(chunk[chunk.index("}") + 1:].split())
        rest = re.sub(r"\\emph\{([^}]*)\}", r"\1", rest)
        rest = re.sub(r"\\textbf\{([^}]*)\}", r"|VOL|\1|", rest)
        year = (re.search(r"\((\d{4})\)", rest) or [None, ""])[1]
        vol = (re.search(r"\|VOL\|([^|]+)\|", rest) or [None, ""])[1]
        page = (re.search(r"(\d+)\s*--\s*\d+", rest) or [None, ""])[1]
        # title: between the author list (ends at the first ': ') and the venue
        t = re.split(r":\s", rest, maxsplit=1)
        title = t[1] if len(t) > 1 else rest
        title = re.split(r"\.\s|\{", title)[0]
        journal = (re.search(r"\{([^}]+)\}", rest) or [None, ""])[1]
        out.append((key, title, journal, vol, page, year))
    return out


def crossref(title):
    q = urllib.parse.quote(title[:200])
    url = (f"https://api.crossref.org/works?query.bibliographic={q}&rows=3"
           f"&mailto={MAILTO}"
           "&select=title,container-title,volume,page,issued,DOI")
    try:
        r = subprocess.run(["curl", "-s", "--max-time", "25", url],
                           capture_output=True, text=True)
        return json.loads(r.stdout)["message"]["items"]
    except Exception:
        return []


def main(path, limit=None):
    text = open(path, encoding="utf-8").read()
    entries = parse_bib(text) if path.endswith(".bib") else parse_bibitem(text)
    if limit:
        entries = entries[:limit]
    print(f"{path}: {len(entries)} entries\n")
    tally = {"OK": 0, "CHECK": 0, "NO-MATCH": 0}
    for key, title, journal, vol, page, year in entries:
        if not title.strip():
            print(f"  NO-MATCH  {key:22s} (no parsable title)")
            tally["NO-MATCH"] += 1
            continue
        best, best_ov = None, 0.0
        for it in crossref(title):
            ct = (it.get("title") or [""])[0]
            ov = (len(tokens(title) & tokens(ct)) / max(1, len(tokens(title))))
            if ov > best_ov:
                best, best_ov = it, ov
        time.sleep(0.15)
        if not best or best_ov < 0.6:
            print(f"  NO-MATCH  {key:22s} {title[:52]}")
            tally["NO-MATCH"] += 1
            continue
        cv = str(best.get("volume", "") or "")
        cp = first_page(str(best.get("page", "") or ""))
        cy = str((best.get("issued", {}).get("date-parts") or [[""]])[0][0])
        cj = (best.get("container-title") or [""])[0]
        mp = first_page(page)

        # A high title overlap is not enough: Crossref will happily return a
        # meeting abstract or a later book chapter with nearly the same title.
        # Require the journal to agree too before trusting the record.
        jrn_ok = (not journal or not cj
                  or len(tokens(journal) & tokens(cj)) >= 1)
        if not jrn_ok:
            print(f"  CHECK     {key:22s} journal {journal!r} vs Crossref "
                  f"{cj!r}  doi:{best.get('DOI')}")
            tally["CHECK"] += 1
            continue

        bad = []
        if vol and cv and vol != cv:
            bad.append(f"vol {vol} vs {cv}")
        if mp and cp and mp != cp:
            bad.append(f"first page {mp} vs {cp}")
        if year and cy and abs(int(year) - int(cy)) > 1:
            bad.append(f"year {year} vs {cy}")
        if bad:
            print(f"  CHECK     {key:22s} {'; '.join(bad)}  doi:{best.get('DOI')}")
            tally["CHECK"] += 1
        else:
            tally["OK"] += 1
    print(f"\n  OK {tally['OK']}   CHECK {tally['CHECK']}   "
          f"NO-MATCH {tally['NO-MATCH']}")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    lim = next((int(x.split("=")[1]) for x in sys.argv if x.startswith("--limit")),
               None)
    main(a[0], lim)
