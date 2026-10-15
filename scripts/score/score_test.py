#!/usr/bin/env python3
"""test_public result tables from the released predictions and verdicts.

Usage: python3 scripts/score/score_test.py [--bundle-exclusions FILE]
"""
import csv
import sys
from pathlib import Path

import verdict_sets
from verdict_sets import G, MODELS, R, SETTINGS, TEST, acc, ids, items, verdicts, wilson

if "--before" in sys.argv:
    BX = R / "data/exclusions/bundle_gold_page.txt"
elif "--bundle-exclusions" in sys.argv:
    BX = Path(sys.argv[sys.argv.index("--bundle-exclusions") + 1])
    BX = BX if BX.is_absolute() else R / BX
else:
    BX = R / "data/exclusions/bundle.txt"
verdict_sets.BEX = BEX = ids(BX)

print(f"bundle exclusions: {BX.parent.name}/{BX.name} ({len(BEX)} items, {len(BEX & set(TEST))} in test_public scope)")
verdict_sets.FIRST_CALL_RULE = True
V = {s: {m: verdicts(s, m) for m in MODELS} for s in SETTINGS}

print("\n== main (plain / cw, claude-sonnet-5-5 judge)")
for m in MODELS:
    cells = []
    for s in SETTINGS:
        qs = items(s, V[s][m])
        p, c = acc(V[s][m], qs)
        cells.append(f"{s} {p:.1f}/{c:.1f} (n={len(qs)})")
    print(f"  {m}: " + " | ".join(cells))

print("\n== paired: all settings on the same full-context sample items (plain)")
for m in MODELS:
    sample = items("fc", V["fc"][m])
    cells = [f"{s} {acc(V[s][m], sample)[0]:.0f} ({sum(V[s][m][q] for q in sample)})" for s in SETTINGS]
    print(f"  {m}: n={len(sample)} | " + " | ".join(cells))

print("\n== m1: BM25 / FC correct counts on the sample")
for m in MODELS:
    sample = items("fc", V["fc"][m])
    out = []
    for name, qs in (("all", sample),
                     ("no 512k", [q for q in sample if G[q].get("context_tier") != "512k"]),
                     ("no agg", [q for q in sample if G[q]["task_type"] != "cross_source_aggregation"])):
        b, f = sum(V["bm25"][m][q] for q in qs), sum(V["fc"][m][q] for q in qs)
        out.append(f"{name} {b}/{f} ({f - b:+d}; n={len(qs)})")
    print(f"  {m}: " + " | ".join(out))

print("\n== tier: FC plain accuracy by context tier")
for m in MODELS:
    sample = items("fc", V["fc"][m])
    out = []
    for t in ("32k", "64k", "128k", "256k", "512k"):
        qs = [q for q in sample if G[q].get("context_tier") == t]
        if qs:
            out.append(f"{t} {acc(V['fc'][m], qs)[0]:.0f} (n={len(qs)})")
    print(f"  {m}: " + "; ".join(out))

print("\n== hitmiss: non-aggregation items, BM25")
rows = [r for r in csv.DictReader(open(R / "results/retrieval/nonagg_recall_test.csv", encoding="utf-8"))
        if r["qa_id"] not in BEX]
rk = {k: 100 * sum(float(r[f"bm25_1200_r@{k}"]) for r in rows) / len(rows) for k in (5, 10, 20, 50)}
gain = sum(1 for r in rows if float(r["bm25_1200_r@20"]) >= 1 > float(r["bm25_1200_r@5"]))
print(f"  n={len(rows)}; BM25 recall@5 {rk[5]:.1f}, @10 {rk[10]:.1f}, @20 {rk[20]:.1f}, @50 {rk[50]:.1f}; "
      f"items gained from k=5 to k=20: {gain}")
hit = {r["qa_id"] for r in rows if float(r["bm25_1200_r@5"]) >= 1}
miss = {r["qa_id"] for r in rows if float(r["bm25_1200_r@5"]) == 0}
for m in MODELS:
    v = V["bm25"][m]
    h, mi = [q for q in hit if q in v], [q for q in miss if q in v]
    print(f"  {m}: hit {acc(v, h)[0]:.0f} (n={len(h)}) | miss {acc(v, mi)[0]:.0f} (n={len(mi)})")

print("\n== order: models by plain accuracy, claude-sonnet-5-5 vs gpt-4.1-mini judge")
for s in ("bm25", "dense", "fc"):
    for judge in ("claude-sonnet-5-5", "gpt-4.1-mini"):
        vals = {}
        for m in MODELS:
            v = verdicts(s, m, judge)
            vals[m] = acc(v, items(s, v))[0]
        print(f"  {s} {judge}: " + ", ".join(f"{m} {vals[m]:.1f}" for m in sorted(vals, key=lambda m: -vals[m])))

print("\n== second: main table under the gpt-4.1-mini judge (plain / cw; * = rounded value differs from claude-sonnet-5-5)")
for m in MODELS:
    cells = []
    for s_ in SETTINGS:
        v2 = verdicts(s_, m, "gpt-4.1-mini")
        qs = items(s_, v2)
        p2, c2 = acc(v2, qs)
        p1, c1 = acc(V[s_][m], items(s_, V[s_][m]))
        mark = lambda a, b: f"{round(a)}{'*' if round(a) != round(b) else ''}"
        cells.append(f"{s_} {mark(p2, p1)} / {mark(c2, c1)}")
    print(f"  {m}: " + " | ".join(cells))

print("\n== family: BM25 plain accuracy by task family")
fams = {}
for q in items("bm25", V["bm25"][MODELS[0]]):
    fams.setdefault(G[q]["task_type"], []).append(q)
for f, qs in sorted(fams.items(), key=lambda x: -len(x[1])):
    if len(qs) >= 4:
        print(f"  {f} (n={len(qs)}): " + ", ".join(f"{m} {acc(V['bm25'][m], [q for q in qs if q in V['bm25'][m]])[0]:.0f}"
                                               for m in MODELS))

print("\n== hmtab: hit / miss accuracy with Wilson 95% intervals")
for retr, col in (("bm25", "bm25_1200_r@5"), ("dense", "dense_bge_1200_r@5")):
    h_ = {r["qa_id"] for r in rows if float(r[col]) >= 1}
    m_ = {r["qa_id"] for r in rows if float(r[col]) == 0}
    print(f"  {retr}: hit n={len(h_)}, miss n={len(m_)}")
    for m in MODELS:
        v = V[retr][m]
        out = []
        for name, S in (("hit", h_), ("miss", m_)):
            qs = [q for q in S if q in v]
            k = sum(v[q] for q in qs)
            lo, hi = wilson(k, len(qs))
            out.append(f"{name} {100 * k / len(qs):.0f} [{lo:.0f}, {hi:.0f}]")
        print(f"    {m}: " + " | ".join(out))
