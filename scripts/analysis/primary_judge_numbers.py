"""Numbers under the primary judge (kimi-k2.6) quoted in the text.

Usage: python3 scripts/analysis/primary_judge_numbers.py <two-stage run dir>
"""
import contextlib, io, runpy, sys, collections, math
from pathlib import Path
S = Path(__file__).resolve().parents[1]
sys.argv = ["analysis_R.py", sys.argv[1]]
with contextlib.redirect_stdout(io.StringIO()):
    A = runpy.run_path(str(S / "two_stage/two_stage_analysis.py"))
E, cell, T81, D144, NONOV = A["E"], A["cell"], A["T81"], A["D144"], A["NONOV"]
V, _, _ = E.load("full"); G = E.G
FOUR = ["gpt-5.5", "glm-5.2", "claude-sonnet-5-5", "minimax-m3"]
FW = ["gpt-4.1-mini", "gpt-5.5", "minimax-m3", "glm-5.2", "claude-sonnet-5-5", "claude-haiku-5-5"]
ALL8 = FW + ["gpt-5.4-mini", "gpt-5.4-nano"]
JV = {"kimi": "kimi", "sonnet": "claude-sonnet-5-5"}
def a8(j, c, m, qs):
    k, n, _ = cell(j, c, m, qs); return 100 * k / n if n else None
def rng(j, c, qs, ms):
    xs = [a8(j, c, m, qs) for m in ms]; xs = [x for x in xs if x is not None]
    return f"{min(xs):.1f}-{max(xs):.1f}"
def line(label, f):
    print(f"{label}: kimi {f('kimi')} | sonnet {f('sonnet')}")
print("== Table 8 / abstract / 5.2 (test 81 aggregation items)")
line("FC, six full-window (9-100)", lambda j: rng(j, "fc", T81, FW))
line("FC, the four (63-100)", lambda j: rng(j, "fc", T81, FOUR))
line("FC, three besides gpt-5.5 (63-88)", lambda j: rng(j, "fc", T81, FOUR[1:]))
line("dense top-5 max (33)", lambda j: rng(j, "dense", T81, ALL8))
line("BM25 top-5 re-run max (10)", lambda j: rng(j, "top5", T81, ALL8))
line("k=50, the four (69-78)", lambda j: rng(j, "k50", T81, FOUR))
line("two-stage test, the four (84-86)", lambda j: rng(j, "twostage", T81, FOUR))
line("two-stage test, eight (0-86)", lambda j: rng(j, "twostage", T81, ALL8))
line("two-stage test, other four (0-25)", lambda j: rng(j, "twostage", T81, ["gpt-4.1-mini", "claude-haiku-5-5", "gpt-5.4-mini", "gpt-5.4-nano"]))
line("two-stage dev', the four (94-98)", lambda j: rng(j, "twostage", NONOV, FOUR))
line("two-stage dev', eight (0-98)", lambda j: rng(j, "twostage", NONOV, ALL8))
line("two-stage dev 144, the four (85-92)", lambda j: rng(j, "twostage", D144, FOUR))
line("open-region, the four (94-100)", lambda j: rng(j, "openregion", T81, FOUR))
line("oracle, the four (95-100)", lambda j: rng(j, "oracle", T81, FOUR))
line("open-region / oracle gpt-4.1-mini (23 / 28)", lambda j: f"{a8(j,'openregion','gpt-4.1-mini',T81):.1f} / {a8(j,'oracle','gpt-4.1-mini',T81):.1f}")
line("open-region / oracle haiku (44 / 60)", lambda j: f"{a8(j,'openregion','claude-haiku-5-5',T81):.1f} / {a8(j,'oracle','claude-haiku-5-5',T81):.1f}")
print("== r = (g - t) / (f - t), t = BM25 top-5 re-run, f = FC (5.2, S5; >= 10-point rule)")
for j in ("kimi", "sonnet"):
    out = []
    for m in FW:
        t, g, f = a8(j, "top5", m, T81), a8(j, "twostage", m, T81), a8(j, "fc", m, T81)
        out.append(f"{m} f-t {f-t:.1f}" + (f" r {(g-t)/(f-t):.2f}" if f - t >= 10 else " (not classified)"))
    print(f"  {j}: " + "; ".join(out))
print("== 5.1 / Table 7 (full-context sample; 96 or 77 items)")
for j in ("kimi", "sonnet"):
    jv = JV[j]; res = {}
    for m in ALL8:
        qs = sorted(V["claude-sonnet-5-5"][("fc", m)])
        res[m] = {s: 100 * sum(V[jv][(s, m)][q] for q in qs) / len(qs) for s in ("cb", "bm25", "dense", "fc")}
    d = {m: res[m]["fc"] - res[m]["bm25"] for m in ALL8}
    print(f"  {j}: CB max {max(r['cb'] for r in res.values()):.1f}; BM25 min {min(r['bm25'] for r in res.values()):.1f}; FC min {min(r['fc'] for r in res.values()):.1f}; "
          f"FC-BM25 the four {min(d[m] for m in FOUR):.1f}-{max(d[m] for m in FOUR):.1f}; other four max {max(d[m] for m in ALL8 if m not in FOUR):.1f}")
    gl = []
    for m in FW:
        qs = sorted(V["claude-sonnet-5-5"][("fc", m)]); ag = [q for q in qs if G[q]["task_type"] == "cross_source_aggregation"]; na = [q for q in qs if q not in ag]
        ga = 100 * (sum(V[jv][("fc", m)][q] for q in ag) - sum(V[jv][("bm25", m)][q] for q in ag)) / len(ag)
        gn = 100 * (sum(V[jv][("fc", m)][q] for q in na) - sum(V[jv][("bm25", m)][q] for q in na)) / len(na)
        gl.append((m, ga, gn))
    print("     per-item gain agg / non-agg: " + "; ".join(f"{m} {ga:+.0f} / {gn:+.0f}" for m, ga, gn in gl))
    print("     BM25-only correct items (the four): " + ", ".join(f"{m} {sum(V[jv][('bm25', m)][q] and not V[jv][('fc', m)][q] for q in sorted(V['claude-sonnet-5-5'][('fc', m)]))}" for m in FOUR))
print("== S25 BM25 by family (six full-window; 89-99 for the retrieval families) and S24 by tier (83% or more through 256k)")
for j in ("kimi", "sonnet"):
    jv = JV[j]; fam = collections.defaultdict(list)
    for m in FW:
        v = V[jv][("bm25", m)]
        for f in ("long_context_retrieval", "long_distance_retrieval"):
            qs = [q for q in v if G[q]["task_type"] == f]; fam[f].append(100 * sum(v[q] for q in qs) / len(qs))
    tiers = []
    for m in FW:
        qs = sorted(V["claude-sonnet-5-5"][("fc", m)])
        for tier in ("32k", "64k", "128k", "256k"):
            tq = [q for q in qs if G[q].get("context_tier") == tier]
            if tq: tiers.append(100 * sum(V[jv][("fc", m)][q] for q in tq) / len(tq))
    print(f"  {j}: retrieval families {min(min(x) for x in fam.values()):.1f}-{max(max(x) for x in fam.values()):.1f}; FC through 256k min {min(tiers):.1f}")
