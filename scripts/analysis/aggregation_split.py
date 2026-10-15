"""Full context minus BM25 on aggregation and non-aggregation items.
"""
import random, sys
from collections import defaultdict
from math import comb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "judges"))
from panel_verdicts import AGG81, FULLWIN, G, MODELS, C, agg81, load
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import table_rows as q3
V, _, _ = load("full")

def binom2(k, n):
    if n == 0: return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n)
def boot(qs, unit, stat, B=10000, seed=20261013):
    g = defaultdict(list)
    for q in qs: g[unit(q)].append(q)
    keys = sorted(g); rnd = random.Random(seed)
    out = sorted(stat([q for k in (rnd.choice(keys) for _ in keys) for q in g[k]]) for _ in range(B))
    return out[int(0.025 * B)], out[int(0.975 * B) - 1]

print("== (1) FC - BM25 on the full-context sample, aggregation vs non-aggregation (points; item / cluster bootstrap; FC-only/BM25-only; exact McNemar p)")
for jn in ("claude-sonnet-5-5", "gpt-4.1-mini", "kimi"):
    print(f"-- {jn}")
    for m in MODELS:
        smp = C.items("fc", V["claude-sonnet-5-5"][("fc", m)]); fc, bm = V[jn][("fc", m)], V[jn][("bm25", m)]
        for lab, qs in (("agg", [q for q in smp if G[q]["task_type"] == "cross_source_aggregation"]),
                        ("non-agg", [q for q in smp if G[q]["task_type"] != "cross_source_aggregation"])):
            if not qs:
                print(f"  {m} {lab}: n=0"); continue
            st = lambda s: 100 * (sum(fc[q] for q in s) - sum(bm[q] for q in s)) / len(s)
            ib = boot(qs, lambda q: q, st); cb = boot(qs, lambda q: G[q]["cluster_id"], st)
            b = sum(1 for q in qs if fc[q] and not bm[q]); c = sum(1 for q in qs if bm[q] and not fc[q])
            print(f"  {m} {lab} (n={len(qs)}, clusters {len({G[q]['cluster_id'] for q in qs})}): FC {sum(fc[q] for q in qs)} BM25 {sum(bm[q] for q in qs)} "
                  f"d {st(qs):+.1f} item [{ib[0]:+.1f}, {ib[1]:+.1f}] cluster [{cb[0]:+.1f}, {cb[1]:+.1f}] | {b}/{c} p {binom2(b, b + c):.6f}")

print("\n== (2) aggregation set (81), claude-sonnet-5-5 judge: correct / n by operator and by matched-row-count bin")
op = {q: G[q]["gold_predicate"]["op"] for q in AGG81}
nrow = {q: len(q3.select(G[q]["gold_predicate"])) for q in AGG81}
BINS = [("1-10", 1, 10), ("11-50", 11, 50), ("51-223", 51, 223)]
assert all(1 <= nrow[q] <= 223 for q in AGG81)
rb = {q: next(b for b, lo, hi in BINS if lo <= nrow[q] <= hi) for q in AGG81}
print("operators:", {o: sum(1 for q in AGG81 if op[q] == o) for o in ("count", "sum", "avg", "max")},
      "| row bins:", {b: sum(1 for q in AGG81 if rb[q] == b) for b, _, _ in BINS},
      "| operator x bin:", {(o, b): n for o in ("count", "sum", "avg", "max") for b, _, _ in BINS if (n := sum(1 for q in AGG81 if op[q] == o and rb[q] == b))})
SET = [("open-region", "openregion"), ("oracle", "oracle"), ("FC", "fc"), ("BM25 top-5", "bm25"), ("dense top-5", "dense")]
for name, s in SET:
    print(f"-- {name}")
    for m in FULLWIN:
        v = agg81(V, "claude-sonnet-5-5", m, s)
        o_ = " ".join(f"{o} {sum(v[q] for q in AGG81 if op[q]==o)}/{sum(1 for q in AGG81 if op[q]==o)}" for o in ("count", "sum", "avg", "max"))
        b_ = " ".join(f"{b} {sum(v[q] for q in AGG81 if rb[q]==b)}/{sum(1 for q in AGG81 if rb[q]==b)}" for b, _, _ in BINS)
        print(f"  {m}: {o_} | {b_} | all {sum(v.values())}/81")
