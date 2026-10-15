#!/usr/bin/env python3
"""Answer classes of the retrieval-family items (writes data/analysis/answer_classes.jsonl).
"""
import ast
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from khat.paths import ROOT, DATA, LOCAL

REL = ROOT
CB = LOCAL / "runs/cb_claude_prompts.jsonl"
HERE = Path(__file__).resolve().parent

jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
lit = lambda x: ast.literal_eval(x) if isinstance(x, str) and x[:1] in "[{" else x
ids = lambda p: {l.strip() for l in open(p, encoding="utf-8") if l.strip() and not l.startswith("#")}

ALL = {}
for s in ("dev", "test_public"):
    for r in jl(REL / f"data/qa_{s}.jsonl"):
        ALL[r["qa_id"]] = r
G = {q: r for q, r in ALL.items() if r["split"] == "test_public"}
EX = ids(REL / "data/exclusions/eval.txt")
RET = ("long_context_retrieval", "long_distance_retrieval")
ITEMS = sorted(q for q, g in G.items() if g["task_type"] in RET and q not in EX)
assert len(ITEMS) == 167, len(ITEMS)

THOUS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
WSP = re.compile(r"\s+")
norm = lambda s: WSP.sub("", THOUS.sub("", s or ""))
BOUND = re.compile(r"[0-9A-Za-z_\-]")


def matches(term, text):
    """start offsets in text (both normalized) where term occurs with a clean boundary on both sides."""
    out, i = [], text.find(term)
    while i >= 0:
        a = text[i - 1] if i > 0 else ""
        b = text[i + len(term)] if i + len(term) < len(text) else ""
        if not (a and BOUND.match(a)) and not (b and BOUND.match(b)):
            out.append(i)
        i = text.find(term, i + 1)
    return out


def terms_of(g):
    ev = lit(g["evaluation"]) or {}
    t = ev.get("gold_terms") or ev.get("gold_numbers") or ([g["answer"]] if g.get("answer") else [])
    return [norm(x) for x in t if norm(x)]


anns = defaultdict(set)
for q, r in ALL.items():
    for a in (lit(r["announcement_ids"]) or lit(r["source_ids"])):
        anns[WSP.sub("", str(r["answer"]))].add(a)
recurring = lambda q: len(anns[WSP.sub("", str(G[q]["answer"]))]) >= 2

cbp = {r["qa_id"]: r["prompt"] for r in jl(CB) if r.get("regime") == "cb"}
pages = {}
for f in glob.glob(str(LOCAL / "*-announcements*/*/document_pages.jsonl")):
    for r in jl(f):
        pages[r["page_id"]] = r.get("text", "")
TITLE = re.compile(r"「([^」]{1,80}?(?:법|법률|시행령|시행규칙|규칙|고시|기준))」")

rows, nrep = [], []
for q in ITEMS:
    g = G[q]
    terms = terms_of(g)
    prompt = norm(cbp[q])
    L = bool(terms) and all(matches(t, prompt) for t in terms)
    R = recurring(q)
    own = {norm(t) for t in re.findall(r"「([^」]+)」", g["question"])}
    ptext = norm("\n".join(pages.get(p, "") for p in lit(g["page_ids"])))
    hits = {}
    for t in terms:
        found = None
        for i in matches(t, ptext):
            win = ptext[max(0, i - 200): i + len(t) + 200]
            ts = [m for m in TITLE.findall(win) if norm(m) not in own]
            if ts:
                found = sorted(set(ts))
                break
        hits[t] = found
    N = bool(terms) and all(hits[t] for t in terms)
    Sp = not L and not R
    rows.append({"qa_id": q, "task_type": g["task_type"], "n_terms": len(terms), "L": L, "R": R, "N": N,
                 "S_prime": Sp, "S": Sp and not N, "announcement_ids": lit(g["announcement_ids"]),
                 "page_ids": lit(g["page_ids"]), "pages_found": sum(p in pages for p in lit(g["page_ids"]))})
    if N:
        nrep.append({"qa_id": q, "page_ids": lit(g["page_ids"]), "titles": sorted({x for v in hits.values() for x in v})})

with open(DATA / "analysis/answer_classes.jsonl", "w", encoding="utf-8") as fo:
    for r in rows:
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(DATA / "analysis/answer_classes_report.jsonl", "w", encoding="utf-8") as fo:
    for r in nrep:
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
c = Counter()
for r in rows:
    for k in ("L", "R", "N", "S_prime", "S"):
        c[k] += r[k]
print(f"items {len(rows)}; L {c['L']}; R {c['R']} (unique {len(rows) - c['R']}); N {c['N']}; S' {c['S_prime']}; S {c['S']}")
print(f"items whose cited pages are all in the corpus: {sum(r['pages_found'] == len(r['page_ids']) for r in rows)}")
print("N by family:", Counter(r["task_type"] for r in rows if r["N"]))
