"""Wilson, pair and cluster bootstrap intervals on the aggregation sets.

Usage: python3 scripts/analysis/cluster_bootstrap.py <two-stage run dir> [--judge kimi|sonnet|gpt-4.1-mini]
"""
import contextlib, io, math, random, runpy, sys
from collections import defaultdict
from pathlib import Path
SCRIPTS = Path(__file__).resolve().parents[1]
W = sys.argv[1]; JUDGE = sys.argv[sys.argv.index("--judge") + 1] if "--judge" in sys.argv else "sonnet"
sys.argv = ["analysis_R.py", W]
with contextlib.redirect_stdout(io.StringIO()):
    A = runpy.run_path(str(SCRIPTS / "two_stage/two_stage_analysis.py"))
G, cell, T81, NONOV = A["G"], A["cell"], A["T81"], A["NONOV"]
ORDER = ["gpt-4.1-mini", "gpt-5.5", "minimax-m3", "glm-5.2", "claude-sonnet-5-5", "claude-haiku-5-5", "gpt-5.4-mini", "gpt-5.4-nano"]
B, SEED = 10000, 20261011


def wilson(k, n, z=1.959963984540054):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - h) / d, 100 * (c + h) / d


def boot(items, unit, v):
    groups = defaultdict(list)
    for q in items: groups[unit(q)].append(q)
    keys = sorted(groups); rnd = random.Random(SEED)
    out = sorted((lambda s: sum(v[q] for q in s) / len(s))([q for k in (rnd.choice(keys) for _ in keys) for q in groups[k]]) for _ in range(B))
    return 100 * out[int(0.025 * B)], 100 * out[int(0.975 * B) - 1], len(keys)


pair = lambda q: (G[q]["announcement_ids"][0], tuple(G[q]["source_ids"]))
clu = lambda q: G[q]["cluster_id"]
iv = lambda t: f"[{t[0]:.0f}, {t[1]:.0f}]"
SETS = [("test81", T81, [("top5", "BM25 top-5 (re-run)"), ("top5rep", "BM25 top-5 (reported)"), ("dense", "dense top-5"), ("k50", "k=50"), ("twostage", "two-stage"),
                              ("fc", "full context (check: Table S13)"), ("openregion", "open-region (check: Table S13)"), ("oracle", "oracle (check: Table S13)")]),
        ("dev'81", NONOV, [("twostage", "two-stage"), ("top5", "BM25 top-5")])]
for name, items, conds in SETS:
    gp = defaultdict(int); gc = defaultdict(int)
    for q in items: gp[pair(q)] += 1; gc[clu(q)] += 1
    print(f"== {name}: n={len(items)}, pairs {len(gp)} (sizes {sorted(gp.values(), reverse=True)}), clusters {len(gc)} (sizes {sorted(gc.values(), reverse=True)})")
    for c, lab in conds:
        print(f"-- {lab}")
        for m in ORDER:
            k, n, v = cell(JUDGE, c, m, items)
            if n == 0: continue
            if n != len(items): print(f"   {m}: only {n} items"); continue
            w = wilson(k, n); pb = boot(items, pair, v); cb = boot(items, clu, v)
            print(f"   {m:18s} {100*k/n:5.1f} ({k}/{n})  Wilson [{w[0]:.0f}, {w[1]:.0f}]  pair {iv(pb)}  cluster {iv(cb)}")
