"""Paired exact tests of full context against BM25 on each model's full-context sample.
"""
import random, sys
from collections import defaultdict
from math import comb
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "judges"))
from panel_verdicts import AGG81, FULLWIN, G, MODELS, C, agg81, load

V, _, _ = load("full")
JN = ["claude-sonnet-5-5", "gpt-4.1-mini", "kimi"]

def binom2(k, n):
    if n == 0: return 1.0
    t = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(t + 1)) / 2 ** n)

def boot(qs, unit, stat, B=10000, seed=20261013):
    g = defaultdict(list)
    for q in qs: g[unit(q)].append(q)
    keys = sorted(g); rnd = random.Random(seed)
    out = sorted(stat([q for k in (rnd.choice(keys) for _ in keys) for q in g[k]]) for _ in range(B))
    return out[int(0.025 * B)], out[int(0.975 * B) - 1]

print("== M1: FC - BM25 on the full-context sample (points); [item bootstrap] [cluster bootstrap]; FC-only / BM25-only correct; exact McNemar p")
for jn in JN:
    print(f"-- judge: {jn}")
    for m in MODELS:
        smp = C.items("fc", V["claude-sonnet-5-5"][("fc", m)])
        fc, bm = V[jn][("fc", m)], V[jn][("bm25", m)]
        assert all(q in bm for q in smp)
        for lab, qs in (("all", smp), ("excl512k", [q for q in smp if G[q].get("context_tier") != "512k"])):
            st = lambda s: 100 * (sum(fc[q] for q in s) - sum(bm[q] for q in s)) / len(s)
            ib = boot(qs, lambda q: q, st); cb = boot(qs, lambda q: G[q]["cluster_id"], st)
            b = sum(1 for q in qs if fc[q] and not bm[q]); c = sum(1 for q in qs if bm[q] and not fc[q])
            ncl = len({G[q]["cluster_id"] for q in qs})
            print(f"  {m} {lab} (n={len(qs)}, clusters {ncl}): FC {sum(fc[q] for q in qs)} BM25 {sum(bm[q] for q in qs)} "
                  f"d {st(qs):+.1f} item [{ib[0]:+.1f}, {ib[1]:+.1f}] cluster [{cb[0]:+.1f}, {cb[1]:+.1f}] | {b}/{c} McNemar p {binom2(b, b + c):.4f}")

PAIRU = {q: (G[q]["announcement_ids"][0], tuple(G[q]["source_ids"])) for q in AGG81}
pairs = sorted(set(PAIRU.values()))
assert len(pairs) == 9 and all(sum(1 for q in AGG81 if PAIRU[q] == p) == 9 for p in pairs)
print("\n== M2: exact sign test over 9 announcement-table pairs (pos / neg / tied pairs, p); per-pair differences in pair order")
for jn in JN:
    print(f"-- judge: {jn}")
    for a, b in (("k50page", "k50"), ("oracle", "dense"), ("oracle", "bm25")):
        for m in FULLWIN:
            if a == "k50page":
                vk = agg81(V, jn, m, "k50"); vp = dict(vk); vp.update(V[jn][("k50page", m)])
                va, vb = vp, vk
            else:
                va, vb = agg81(V, jn, m, a), agg81(V, jn, m, b)
            d = [sum(va[q] for q in AGG81 if PAIRU[q] == p) - sum(vb[q] for q in AGG81 if PAIRU[q] == p) for p in pairs]
            pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
            print(f"  {a}-{b} {m}: {pos}/{neg}/{9-pos-neg} p {binom2(pos, pos + neg):.4f} | {d} | total {sum(d):+d}")
print("\npairs:", pairs)
