#!/usr/bin/env python3
"""Items whose question carries an announcement number or page marker, and the stripped question.
"""
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from khat.paths import ROOT, DATA

REL = ROOT
HERE = Path(__file__).resolve().parent
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
ids = lambda p: {l.strip() for l in open(p, encoding="utf-8") if l.strip() and not l.startswith("#")}

RULES = [(re.compile(r"\s*\(\s*p\.?\s*\d+\s*\)"), ""),
         (re.compile(r"(입주자\s*모집\s*공고)\s+\d{10}(?=」)"), r"\1"),
         (re.compile(r"\(\s*(LH)\s*,\s*관리번호\s*\d{10}\s*\)"), r"(\1)")]
CHECKS = [re.compile(r"(?<!\d)\d{10}(?!\d)"), re.compile(r"\(?\s*p\.?\s*\d+\s*\)?"),
          re.compile(r"\d+\s*(쪽|페이지)"), re.compile(r"(?i)page\s*\d+")]
TITLE = re.compile(r"「([^」]+)」")


def strip(q):
    for rx, rep in RULES:
        q = rx.sub(rep, q)
    return q


ALL = {}
for s in ("dev", "test_public"):
    for r in jl(REL / f"data/qa_{s}.jsonl"):
        ALL[r["qa_id"]] = r
STATUTES = {json.loads(l)["title"] for l in open(REL / "data/source_manifest.jsonl", encoding="utf-8")
            if json.loads(l).get("source_type") == "statute_or_rule"}

origins = defaultdict(set)
for q, r in ALL.items():
    o, s = TITLE.findall(r["question"]), TITLE.findall(strip(r["question"]))
    assert len(o) == len(s), q
    for a, b in zip(o, s):
        if a not in STATUTES:
            origins[b].add(a)
ambiguous_titles = {b for b, a in origins.items() if len(a) >= 2}

EX = ids(REL / "data/exclusions/eval.txt")
BEX = ids(REL / "data/exclusions/bundle.txt")
TEST = sorted(q for q, r in ALL.items() if r["split"] == "test_public"
              and r["task_type"] != "cross_document_legal_reasoning" and q not in EX)
rows = []
for q in TEST:
    r = ALL[q]
    sq = strip(r["question"])
    bad = [rx.pattern for rx in CHECKS if rx.search(sq)]
    assert not bad, (q, bad)
    if r["task_type"] == "answerability_detection":
        cls = "excluded"
    else:
        ts = [t for t in TITLE.findall(sq) if t not in STATUTES]
        cls = "ambiguous" if any(t in ambiguous_titles for t in ts) else "resolvable"
    rows.append({"qa_id": q, "task_type": r["task_type"], "changed": sq != r["question"], "class": cls,
                 "with_bundle": q not in BEX})
with open(DATA / "analysis/no_marker_items.jsonl", "w", encoding="utf-8") as fo:
    for r in rows:
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
c = Counter(r["class"] for r in rows)
print(f"test_public items {len(rows)} (with a bundle {sum(r['with_bundle'] for r in rows)}): {dict(c)}")
print("changed by stripping:", Counter((r["class"], r["changed"]) for r in rows))
print("changed resolvable by family:", Counter(r["task_type"] for r in rows if r["changed"] and r["class"] == "resolvable"))
print("changed ambiguous by family:", Counter(r["task_type"] for r in rows if r["changed"] and r["class"] == "ambiguous"))
print(f"ambiguous stripped titles (all 1,997 items): {len(ambiguous_titles)}")
