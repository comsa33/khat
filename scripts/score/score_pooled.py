#!/usr/bin/env python3
"""Pooled (dev + test_public) and dev accuracy of the four OpenAI models, overall and by context tier.

Usage: python3 scripts/score/score_pooled.py [--bundle-exclusions FILE] [--json OUT.json]
"""
import json
import sys
from pathlib import Path

from khat.paths import ROOT as R
LEGAL = "cross_document_legal_reasoning"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def ids(p):
    return {l.strip() for l in open(p) if l.strip() and not l.startswith("#")}


G = {d["qa_id"]: d for sp in ("dev", "test_public") for d in jl(R / f"data/qa_{sp}.jsonl") + [x for x in jl(R / "data/auxiliary/legal_reasoning.jsonl") if x["split"] == sp]}
EX = ids(R / "data/exclusions/eval.txt")


def _opt(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


_bx = _opt("--bundle-exclusions")
if _bx is None:
    _bx = Path("data/exclusions/bundle.txt")
B_FINAL = ids(Path(_bx) if Path(_bx).is_absolute() else R / _bx)
JUDGES = ("claude-sonnet-5-5", "gpt-4.1-mini")


def V(judge, tag):
    p = R / f"results/verdicts/{judge}/{tag}.jsonl"
    return {r["qa_id"]: bool(r["judge_correct"]) for r in jl(p)} if p.exists() else {}


M = {"gpt-4.1-mini": ("closed_book_gpt-4.1-mini", "bm25_gpt-4.1-mini", "full_context_gpt-4.1-mini"),
     "gpt-5.4-mini": ("closed_book_gpt-5.4-mini", "bm25_gpt-5.4-mini", "full_context_gpt-5.4-mini"),
     "gpt-5.4-nano": ("closed_book_gpt-5.4-nano", "bm25_gpt-5.4-nano", "full_context_gpt-5.4-nano"),
     "gpt-5.5": ("closed_book_gpt-5.5", "bm25_gpt-5.5", "full_context_gpt-5.5")}


def verdicts(judge, m, setting):
    cb, rag, fc = M[m]
    if setting == "cb":
        v = V(judge, cb)
        for sp in ("dev", "test_public"):
            v.update(V(judge, f"closed_book_no_row_ids_{m}_{sp}"))
        if m == "gpt-5.5":
            v.update(V(judge, "closed_book_second_call_gpt-5.5_dev"))
            v.update(V(judge, "closed_book_second_call_gpt-5.5_test_public"))
        v.update(V(judge, f"{cb}_dev_corrected_gold"))
        if m == "gpt-5.5":
            v.update(V(judge, "closed_book_second_call_gpt-5.5_dev_dev_corrected_gold"))
        return v
    tag = rag if setting == "bm25" else fc
    v = V(judge, tag)
    v.update(V(judge, f"{tag}_dev_corrected_gold"))
    return v


def score(v, keep):
    q = [k for k in v if k in keep]
    n = len(q)
    h = sum(v[k] for k in q)
    wn = sum(float(G[k].get("cluster_weight", 1)) for k in q)
    wh = sum(float(G[k].get("cluster_weight", 1)) for k in q if v[k])
    return {"n": n, "hits": h, "plain": 100 * h / n if n else 0, "cw": 100 * wh / wn if wn else 0}


def scope(split, setting):
    s = {q for q, g in G.items() if g["task_type"] != LEGAL and q not in EX and (split == "pooled" or g["split"] == split)}
    if setting != "cb":
        s -= B_FINAL
    return s


rows, out = [], {}
for judge in JUDGES:
    for m in M:
        for setting in ("cb", "bm25", "fc"):
            for split in ("pooled", "dev"):
                c = score(verdicts(judge, m, setting), scope(split, setting))
                out[f"{judge}|{m}|{setting}|{split}"] = c
                rows.append((judge, m, setting, split, c))

f = lambda s: f"{s['plain']:.1f} / {s['cw']:.1f} (n={s['n']})"
print("| judge | model | setting | split | plain / cluster-weighted |")
print("|---|---|---|---|---|")
for judge, m, s, sp, c in rows:
    print(f"| {judge} | {m} | {s} | {sp} | {f(c)} |")

print("\nS3 pooled table (claude-sonnet-5-5 judge, plain / cluster-weighted, rounded):")
for m in M:
    cells = []
    for s in ("cb", "bm25", "fc"):
        c = out[f"claude-sonnet-5-5|{m}|{s}|pooled"]
        cells.append(f"{c['plain']:.0f}/{c['cw']:.0f}")
    print(f"  {m}: CB {cells[0]} | BM25 {cells[1]} | FC {cells[2]}")
n = {s: out[f"claude-sonnet-5-5|gpt-5.5|{s}|pooled"]["n"] for s in ("cb", "bm25", "fc")}
n54 = out["claude-sonnet-5-5|gpt-5.4-mini|fc|pooled"]["n"]
print(f"  n: closed-book {n['cb']}; BM25 {n['bm25']}; FC {n['fc']} (4.1-mini, 5.5) and {n54} (5.4 models)")

print("\nPooled full context by tier (claude-sonnet-5-5 judge, plain):")
for m in M:
    cells = []
    for t in ("32k", "64k", "128k", "256k", "512k"):
        c = score(verdicts("claude-sonnet-5-5", m, "fc"), {q for q in scope("pooled", "fc") if G[q].get("context_tier") == t})
        if c["n"]:
            cells.append(f"{t} {c['plain']:.0f} (n {c['n']})")
    print(f"  {m}: " + "; ".join(cells))
print(f"\nbundle exclusions: {len(B_FINAL)} items ({_bx})")
if _opt("--json"):
    json.dump(out, open(_opt("--json"), "w"), indent=1)
