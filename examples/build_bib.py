"""Build a BibTeX file from Crossref, so entries are correct by construction.

Verifying a hand-written bibliography means comparing it against the record and
hoping the comparison is sensitive enough. Generating it from the record instead
removes the failure mode: authors, journal, volume, pages, year and DOI all come
from Crossref in one query, and nothing is transcribed by hand.

Each wanted reference is given as (citekey, query, expected_year). The year is
not used to search; it is checked afterwards, so that a query returning a
plausible but wrong paper -- a later review with a similar title, a meeting
abstract, the supporting information rather than the article -- is caught rather
than silently written into the file.

Usage:
    python examples/build_bib.py wanted.json out.bib
"""
import json
import re
import subprocess
import sys
import time
import urllib.parse

MAILTO = "giresse.feugmo@gmail.com"
STOP = {"the", "a", "an", "of", "on", "in", "for", "and", "to", "with", "by",
        "from", "at", "as", "is", "are"}


def toks(s):
    return {w for w in re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split()
            if w not in STOP and len(w) > 2}


def query(q, rows=5):
    url = ("https://api.crossref.org/works?query.bibliographic="
           + urllib.parse.quote(q[:250])
           + f"&rows={rows}&mailto={MAILTO}"
           "&select=title,author,container-title,volume,issue,page,issued,DOI,type,"
           "article-number")
    try:
        r = subprocess.run(["curl", "-s", "--max-time", "30", url],
                           capture_output=True, text=True)
        return json.loads(r.stdout)["message"]["items"]
    except Exception:
        return []


def clean(s):
    """Strip the HTML Crossref embeds, decode entities, escape for BibTeX."""
    s = re.sub(r"<[^>]+>", "", s or "")
    for a, b in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"')):
        s = s.replace(a, b)
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def by_doi(doi):
    """Fetch one work by DOI. Unambiguous, unlike a title search."""
    url = (f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
           f"?mailto={MAILTO}")
    try:
        r = subprocess.run(["curl", "-s", "--max-time", "30", url],
                           capture_output=True, text=True)
        return json.loads(r.stdout)["message"]
    except Exception:
        return None


def pick(items, q, year):
    """Best candidate: title overlap, penalising SI records and wrong years."""
    best, score = None, 0.0
    for it in items:
        t = (it.get("title") or [""])[0]
        if not t:
            continue
        s = len(toks(q) & toks(t)) / max(1, len(toks(q)))
        if str(it.get("DOI", "")).endswith((".s001", ".s002")):
            s -= 0.5                                  # supporting information
        if not it.get("author"):
            s -= 0.3                                  # SI records lack authors
        if year and it.get("issued", {}).get("date-parts"):
            y = it["issued"]["date-parts"][0][0]
            if y and abs(int(y) - int(year)) > 1:
                s -= 0.4
        if s > score:
            best, score = it, s
    return best, score


def to_bib(key, it, extra=None):
    au = " and ".join(f"{a.get('family','')}, {a.get('given','')}".strip(", ")
                      for a in it.get("author", []))
    y = (it.get("issued", {}).get("date-parts") or [[""]])[0][0]
    f = [f"  author = {{{au}}}", f"  title = {{{clean((it.get('title') or [''])[0])}}}",
         f"  journal = {{{clean((it.get('container-title') or [''])[0])}}}",
         f"  year = {{{y}}}"]
    if it.get("volume"):
        f.append(f"  volume = {{{it['volume']}}}")
    # Physical Review, Nature Communications and the like paginate by article
    # number, which Crossref reports separately from `page`. Without this the
    # entry silently loses its locator and BibTeX prints "PRL 94, (2005)".
    page = it.get("page") or it.get("article-number")
    if page:
        f.append(f"  pages = {{{str(page).replace('-', '--')}}}")
    if it.get("DOI"):
        f.append(f"  doi = {{{it['DOI']}}}")
    # A few publishers deposit no locator at all (Frontiers gives neither page
    # nor article-number). Those fields are supplied from the article itself and
    # flagged, so the file still says which lines Crossref did not vouch for.
    note = ""
    for k, v in (extra or {}).items():
        if not any(x.startswith(f"  {k} =") for x in f):
            f.append(f"  {k} = {{{v}}}")
            note = f"% {key}: {', '.join(extra)} not in Crossref; from the article\n"
    return note + "@article{" + key + ",\n" + ",\n".join(f) + "\n}\n"


def main(wanted_path, out_path):
    wanted = json.load(open(wanted_path))
    out, failed, seen_doi = [], [], {}
    for entry in wanted:
        key, q, year = entry[0], entry[1], entry[2]
        doi_hint = entry[3] if len(entry) > 3 else None
        extra = entry[4] if len(entry) > 4 else None
        if doi_hint:
            best = by_doi(doi_hint)
            score = 1.0 if best else 0.0
        else:
            best, score = pick(query(q), q, year)
        time.sleep(0.15)
        if not best or score < 0.55:
            failed.append((key, q, f"no confident match (score {score:.2f})"))
            continue
        y = (best.get("issued", {}).get("date-parts") or [[""]])[0][0]
        if year and y and abs(int(y) - int(year)) > 1:
            failed.append((key, q, f"year {y} but expected {year}"))
            continue
        d = str(best.get("DOI", "")).lower()
        if d in seen_doi:
            failed.append((key, q, f"duplicate of {seen_doi[d]} (same DOI {d})"))
            continue
        seen_doi[d] = key
        out.append(to_bib(key, best, extra))
        print(f"  ok  {key:22s} {(best.get('container-title') or [''])[0][:38]:38s} "
              f"{best.get('volume','?')}, {str(best.get('page','?'))[:12]} ({y})")
    open(out_path, "w", encoding="utf-8").write(
        "% Generated from Crossref by gfckit/examples/build_bib.py.\n"
        "% Every field came from the Crossref record, except where a %-comment\n"
        "% marks a field Crossref does not hold, which is taken from the article.\n\n"
        + "\n".join(out))
    print(f"\n  wrote {len(out)} entries to {out_path}")
    if failed:
        print(f"  {len(failed)} NOT written -- resolve by hand:")
        for k, q, why in failed:
            print(f"    {k:22s} {why}  [{q[:44]}]")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
