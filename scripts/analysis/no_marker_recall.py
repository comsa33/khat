#!/usr/bin/env python3
"""Retrieval recall without the announcement number and page marker in the question.
"""
import csv
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from khat.paths import ROOT, DATA, LOCAL

HERE = Path(__file__).resolve().parent
REL = ROOT
sys.path.insert(0, str(HERE))
import rag_lib as lib
import rag_rows as rows

spec = importlib.util.spec_from_file_location("no_marker_items", HERE / "no_marker_items.py")
src = open(spec.origin, encoding="utf-8").read().split("ALL = {}")[0]
NS = {"__file__": spec.origin}
exec(src, NS)
strip = NS["strip"]

CLS = {json.loads(l)["qa_id"]: json.loads(l) for l in open(DATA / "analysis/no_marker_items.jsonl", encoding="utf-8")}
G = {json.loads(l)["qa_id"]: json.loads(l) for l in open(REL / "data/qa_test_public.jsonl", encoding="utf-8")}
qa = lib.test_qa()
assert all(qa[q]["question"] == G[q]["question"] for q in CLS), "question text differs between the retrieval QA records and the released QA file"
KS = (5, 10, 20)
rk = lib.Ranker()
agg_items, row_checks = rows.build(qa)
AGG = {it["qa_id"]: it for it in agg_items}
scope = [r for r in lib.eval_scope(qa) if r["qa_id"] in CLS and CLS[r["qa_id"]]["class"] != "excluded"]
print(f"scope {len(scope)} (aggregation {sum(r['qa_id'] in AGG for r in scope)}); row checks {row_checks}", flush=True)


def ranking(retr, rec, query):
    return rk.full(retr, rec, rec["bundle_id"], 1200, query)[:max(KS)]


def page_recall(rec, idx):
    ch = lib.chunks_for(rec["bundle_id"], 1200)
    got = {ch[i][0] for i in idx}
    g = set(rec.get("page_ids") or [])
    return len(got & g) / len(g)


def measures(rec, query):
    out = {}
    for retr in ("bm25", "dense_bge"):
        top = ranking(retr, rec, query)
        for k in KS:
            out[f"{retr}_page@{k}"] = page_recall(rec, top[:k])
            if rec["qa_id"] in AGG:
                it = AGG[rec["qa_id"]]
                full = rows.covered(it, top[:k], 1200) == it["n_rows"]
                out[f"{retr}_rows@{k}"] = float(full)
                out[f"{retr}_pagerows@{k}"] = float(full and out[f"{retr}_page@{k}"] == 1.0)
    return out


# ---------------------------------------------------------------- step 1: original questions
orig = {}
for i, rec in enumerate(scope, 1):
    orig[rec["qa_id"]] = measures(rec, None)
    if i % 50 == 0:
        print(f"  original {i}/{len(scope)}", flush=True)
        lib.save_cache()
lib.save_cache()
ref = {r["qa_id"]: r for r in csv.DictReader(open(REL / "results/retrieval/nonagg_recall_test.csv", encoding="utf-8"))}
mism = []
for q, m in orig.items():
    if q in AGG or q not in ref:
        continue
    for retr, col in (("bm25", "bm25_1200"), ("dense_bge", "dense_bge_1200")):
        for k in KS:
            if abs(m[f"{retr}_page@{k}"] - float(ref[q][f"{col}_r@{k}"])) > 1e-9:
                mism.append((q, retr, k, m[f"{retr}_page@{k}"], ref[q][f"{col}_r@{k}"]))
agg_full = {f"{retr}_{m}": [int(sum(orig[q][f"{retr}_{m}@{k}"] for q in AGG if q in orig)) for k in KS]
            for retr in ("bm25", "dense_bge") for m in ("rows", "pagerows")}
fam = defaultdict(list)
for q in orig:
    if q not in AGG:
        fam[G[q]["task_type"]].append(q)
print("original recall@5 by family (BM25 / bge-m3, n):",
      {f: (round(100 * sum(orig[q]['bm25_page@5'] for q in v) / len(v), 1),
           round(100 * sum(orig[q]['dense_bge_page@5'] for q in v) / len(v), 1), len(v)) for f, v in fam.items()})
print("aggregation full-row coverage k=5/10/20:", agg_full)
assert not mism, mism[:20]

# ---------------------------------------------------------------- step 2: stripped questions
strp = {}
for rec in scope:
    sq = strip(rec["question"])
    strp[rec["qa_id"]] = orig[rec["qa_id"]] if sq == rec["question"] else measures(rec, sq)
lib.save_cache()
(LOCAL / "analysis").mkdir(parents=True, exist_ok=True)
with open(LOCAL / "analysis/no_marker_recall_items.jsonl", "w", encoding="utf-8") as fo:
    for rec in scope:
        q = rec["qa_id"]
        fo.write(json.dumps({"qa_id": q, "task_type": G[q]["task_type"], "class": CLS[q]["class"],
                             "changed": CLS[q]["changed"], "cluster_id": G[q].get("cluster_id"),
                             "original": orig[q], "stripped": strp[q]}, ensure_ascii=False) + "\n")

groups = defaultdict(list)
for rec in scope:
    q = rec["qa_id"]
    if CLS[q]["changed"]:
        groups[(G[q]["task_type"], CLS[q]["class"])].append(q)
print("\n== changed items: original -> stripped (%), by family and class")
for (f, c), qs in sorted(groups.items()):
    keys = ["bm25_page@5", "dense_bge_page@5", "bm25_page@10", "dense_bge_page@10", "bm25_page@20", "dense_bge_page@20"]
    if f == "cross_source_aggregation":
        keys += ["bm25_rows@5", "dense_bge_rows@5", "bm25_rows@20", "dense_bge_rows@20",
                 "bm25_pagerows@20", "dense_bge_pagerows@20"]
    cells = [f"{k} {100 * sum(orig[q][k] for q in qs) / len(qs):.1f}->{100 * sum(strp[q][k] for q in qs) / len(qs):.1f}"
             for k in keys]
    print(f"{f} | {c} | n={len(qs)} | " + "; ".join(cells))

# ---------------------------------------------------------------- step 3: cluster bootstrap
print("\n== families with >= 20 changed resolvable items: mean change in recall@5 (stripped - original), points")
rng = random.Random(20261013)
for f in sorted({f for (f, c) in groups if c == "resolvable"}):
    qs = groups[(f, "resolvable")]
    if len(qs) < 20:
        print(f"{f}: {len(qs)} changed resolvable items (< 20): descriptive only")
        continue
    for k in ("bm25_page@5", "dense_bge_page@5") + (("bm25_rows@5", "dense_bge_rows@5") if f == "cross_source_aggregation" else ()):
        d = {q: 100 * (strp[q][k] - orig[q][k]) for q in qs}
        by = defaultdict(list)
        for q in qs:
            by[G[q].get("cluster_id")].append(q)
        cl = list(by)
        est = sum(d.values()) / len(d)
        boots = []
        for _ in range(10000):
            s = [q for c in (rng.choice(cl) for _ in cl) for q in by[c]]
            boots.append(sum(d[q] for q in s) / len(s))
        boots.sort()
        lo, hi = boots[249], boots[9749]
        flag = est <= -5 and hi < 0
        print(f"{f} {k}: n={len(qs)}, clusters={len(cl)}, change {est:+.1f} [{lo:+.1f}, {hi:+.1f}]"
              + ("  -> named in Section 6 (drop >= 5 points, interval excludes zero)" if flag else ""))
