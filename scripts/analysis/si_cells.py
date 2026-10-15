#!/usr/bin/env python3
"""Supplementary table cells under the first-call rule for re-called predictions.
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "judges"))
from panel_verdicts import AGG81, G, R, C, agg81, load, released


wilson = C.wilson
V, _, APPLIED = load("full")
V0, _, _ = load("none")
MM = "minimax-m3"
CH = {q for s, m, q in APPLIED if m == MM and s in ("agg512k", "bm25")}
print("re-called predictions scored by their first call:", ", ".join(f"{s} {q[-4:]}" for s, m, q in APPLIED if q in CH))
KB = C.items("bm25", V["kimi"][("bm25", MM)])
print(f"  {MM}, kimi-k2.6: BM25 first call {sum(V['kimi'][('bm25', MM)][q] for q in KB)}/{len(KB)}, "
      f"second call {sum(V0['kimi'][('bm25', MM)][q] for q in KB)}/{len(KB)}; 512k aggregation first call "
      f"{sum(agg81(V, 'kimi', MM, 'fc').values())}/81, second call {sum(agg81(V0, 'kimi', MM, 'fc').values())}/81")
RECALLED = {s for s, m, q in APPLIED if q in CH}
pc = lambda k, n: f"{100 * k / n:.0f}"


def show(name, value):
    print(f"  {name}: {value}")


# ---- tab:main, tab:secondary (plain / cluster-weighted, rounded)
def pw(VV, j, s, m):
    qs = C.items(s, VV[j][(s, m)]) if s != "fc" else C.items("fc", VV["claude-sonnet-5-5"][("fc", m)])
    v = VV[j][(s, m)]
    p, c = C.acc(v, qs)
    return f"{p:.0f} / {c:.0f}", (sum(v[q] for q in qs), len(qs), p, c)


print("\n== tab:main (claude-sonnet-5-5) and tab:secondary (gpt-4.1-mini), minimax-m3 row, plain / cluster-weighted %")
for j in ("claude-sonnet-5-5", "gpt-4.1-mini"):
    for s in ("cb", "bm25", "dense", "fc"):
        a, d = pw(V, j, s, MM)
        show(f"{j} {s}", a)
        if s in RECALLED:
            print(f"      {d[0]}/{d[1]} = {d[2]:.1f}% plain, {d[3]:.1f}% cluster-weighted")
p_a = pw(V, "claude-sonnet-5-5", "bm25", MM)[0]
s_a = pw(V, "gpt-4.1-mini", "bm25", MM)[0]
print(f"  tab:secondary asterisks (gpt-4.1-mini differs from claude-sonnet-5-5 after rounding): BM25 claude-sonnet-5-5 {p_a}, gpt-4.1-mini {s_a} -> "
      + ", ".join(f"{'plain' if i == 0 else 'cw'} {'*' if x != y else 'no *'}"
                  for i, (x, y) in enumerate(zip(p_a.split(' / '), s_a.split(' / ')))))

# ---- tab:m1
print("\n== tab:m1 (claude-sonnet-5-5), minimax-m3: BM25 / FC correct on the full-context sample")
smp = C.items("fc", V["claude-sonnet-5-5"][("fc", MM)])
subsets = {"All (96)": smp, "No 512k (77)": [q for q in smp if G[q].get("context_tier") != "512k"],
           "No agg. (78)": [q for q in smp if G[q]["task_type"] != "cross_source_aggregation"]}
for nm, qs in subsets.items():
    b, f_ = sum(V["claude-sonnet-5-5"][("bm25", MM)][q] for q in qs), sum(V["claude-sonnet-5-5"][("fc", MM)][q] for q in qs)
    show(nm, f"{b} / {f_}, {f_ - b:+d}")
print("  the BM25 re-called item in the sample: " + ", ".join(q[-4:] for q in CH if q in smp)
      + f" (tier {', '.join(G[q].get('context_tier') for q in CH if q in smp)}, family "
      + ", ".join(G[q]["task_type"] for q in CH if q in smp) + ")")

# ---- bootstraps
PAIR = {q: (G[q]["announcement_ids"][0], tuple(G[q]["source_ids"])) for q in AGG81}
CLU = {q: G[q]["cluster_id"] for q in AGG81}
assert len(set(PAIR.values())) == 9 and len(set(CLU.values())) == 7


def boot(units, stat, B=10000, seed=20261011):
    groups = defaultdict(list)
    for q in AGG81:
        groups[units[q]].append(q)
    keys = sorted(groups)
    rnd = random.Random(seed)
    out = sorted(stat([q for k in (rnd.choice(keys) for _ in keys) for q in groups[k]]) for _ in range(B))
    return out[int(0.025 * B)], out[int(0.975 * B) - 1]


acc = lambda v: (lambda s: sum(v[q] for q in s) / len(s))
iv = lambda t: f"[{100 * t[0]:.0f}, {100 * t[1]:.0f}]"
ivd = lambda t: f"[{100 * t[0]:+.0f}, {100 * t[1]:+.0f}]"

print("\n== tab:aggboot (claude-sonnet-5-5), minimax-m3 full context: acc, Wilson, pair, cluster")
v = agg81(V, "claude-sonnet-5-5", MM, "fc")
k = sum(v.values())
w = wilson(k, 81)
print(f"  {k}/81 = {pc(k, 81)} | Wilson [{w[0]:.0f}, {w[1]:.0f}] | pair {iv(boot(PAIR, acc(v)))} | cluster {iv(boot(CLU, acc(v)))}")

print("\n== paired-difference paragraph (claude-sonnet-5-5, full context, points; pair and cluster bootstrap)")
for a_, b_ in (("glm-5.2", MM), ("claude-sonnet-5-5", MM)):
    va, vb = agg81(V, "claude-sonnet-5-5", a_, "fc"), agg81(V, "claude-sonnet-5-5", b_, "fc")
    st = lambda s_: (sum(va[q] for q in s_) - sum(vb[q] for q in s_)) / len(s_)
    pb, cb = boot(PAIR, st), boot(CLU, st)
    sign = "keeps one sign under both" if (pb[0] > 0 and cb[0] > 0) or (pb[1] < 0 and cb[1] < 0) else "changes sign within at least one interval"
    print(f"  {a_} minus {b_}: {100 * st(AGG81):+.0f} {ivd(pb)} and {ivd(cb)} ({sign})")
print("  open-region minus full context, within model (both intervals above zero?):")
cells = []
for m in ("glm-5.2", MM, "claude-sonnet-5-5", "claude-haiku-5-5", "gpt-4.1-mini"):
    va, vb = agg81(V, "claude-sonnet-5-5", m, "openregion"), agg81(V, "claude-sonnet-5-5", m, "fc")
    st = lambda s_: (sum(va[q] for q in s_) - sum(vb[q] for q in s_)) / len(s_)
    pb, cb = boot(PAIR, st), boot(CLU, st)
    cells.append(f"{m} {100 * st(AGG81):+.0f} {ivd(pb)} {ivd(cb)}{'' if pb[0] > 0 and cb[0] > 0 else ' NOT both above 0'}")
print("    " + "; ".join(cells))

# ---- tab:pairs
print("\n== tab:pairs (claude-sonnet-5-5), minimax-m3 FC per announcement-table pair (9 items each)")
groups = defaultdict(list)
for q in AGG81:
    groups[PAIR[q]].append(q)
for p, qs in sorted(groups.items()):
    show(f"{p[0]} / {'+'.join(p[1])}", sum(agg81(V, "claude-sonnet-5-5", MM, "fc")[q] for q in qs))
show("All (81)", sum(agg81(V, "claude-sonnet-5-5", MM, "fc").values()))

# ---- tab:k50 (claude-sonnet-5-5; minimax rows)
print("\n== tab:k50 (claude-sonnet-5-5), minimax-m3 rows: FC, d = k50 - FC (points), pair bootstrap of d, label")
RA = released(["verdicts/{j}/k50_no_reasoning_minimax-m3"], "claude-sonnet-5-5")
fcv = agg81(V, "claude-sonnet-5-5", MM, "fc")
for row, kv in (("reasoning on", V["claude-sonnet-5-5"][("k50", MM)]), ("reasoning absent", RA)):
    kv = {q: kv[q] for q in AGG81}
    st = lambda s_: (sum(kv[q] for q in s_) - sum(fcv[q] for q in s_)) / len(s_)
    pb = boot(PAIR, st)
    d = 100 * st(AGG81)
    label = "below" if pb[1] < 0 else ("above" if pb[0] > 0 else "n.d.")
    print(f"  {row}: k50 {sum(kv.values())}/81, FC {pc(sum(fcv.values()), 81)}, d {d:+.0f}, pair {ivd(pb)}, label {label}")

# ---- tab:k50page (claude-sonnet-5-5; the 18 items)
unc = sorted(V["claude-sonnet-5-5"][("k50page", "gpt-5.5")])
print("\n== tab:k50page (claude-sonnet-5-5), minimax-m3 FC on the 18 items")
show("FC", sum(agg81(V, "claude-sonnet-5-5", MM, "fc")[q] for q in unc))
print("  re-called items among the 18: " + (", ".join(q[-4:] for q in CH if q in unc) or "none"))

# ---- tab:judgepanel (mmx column, aggregation set, full context; six judges)
print("\n== tab:judgepanel, minimax-m3 column (aggregation set, full context), per judge")
RJ = R / "results/verdicts/panel"
PANEL = {"claude-sonnet-5-5": None, "gpt-4.1-mini": None,
         "gemma4:31b": RJ / "panel_gemma4_31b_full_context_aggregation_minimax-m3.jsonl",
         "deepseek-v4.1-flash": R / "results/verdicts/panel/panel_deepseek-v4.1-flash_full_context_aggregation_minimax-m3.jsonl",
         "kimi-k2.6": RJ / "panel_kimi-k2.6_full_context_aggregation_minimax-m3.jsonl",
         "mistral-large-3": RJ / "panel_mistral-large-3_full_context_aggregation_minimax-m3.jsonl"}
agg_ch = [q for q in CH if q in AGG81]
for nm, f in PANEL.items():
    if f is None:
        a = sum(agg81(V, nm, MM, "fc").values())
    else:
        v = {json.loads(l)["qa_id"]: bool(json.loads(l)["judge_correct"]) for l in open(f)}
        assert set(v) == set(AGG81)
        a = sum(v.values()) - sum(v[q] for q in agg_ch)
    show(nm, f"{pc(a, 81)} ({a})")
print("  kimi-k2.6 in the primary-judge rescoring: "
      f"{sum(agg81(V, 'kimi', MM, 'fc').values())}/81")

# ---- tab:fcerrors, tab:tier: full-context sample only
print("\n== tab:fcerrors and tab:tier: full-context sample only; re-called aggregation items in the sample: "
      + (", ".join(q[-4:] for q in agg_ch if q in smp) or "none"))

# ---- tab:variance (run 1 = reported run, claude-sonnet-5-5)
print("\n== tab:variance (claude-sonnet-5-5), minimax-m3 run 1 (the reported run)")
T512 = [q for q in C.TEST if G[q].get("context_tier") == "512k"]
fc512 = {**V["claude-sonnet-5-5"][("fc", MM)], **V["claude-sonnet-5-5"][("agg512k", MM)]}
s19 = [q for q in smp if G[q].get("context_tier") == "512k"]
show("19 items", f"{sum(fc512[q] for q in s19)}/19")
all512 = [q for q in T512 if q in fc512]
k1 = sum(fc512[q] for q in all512)
print(f"  512k items with a reported full-context verdict in the evaluated scope: {len(all512)}")
show(f"{len(all512)} items", f"{k1}/{len(all512)} = {100 * k1 / len(all512):.1f}")
run1 = released([f"verdicts/{{j}}/{t}" for t in ("full_context_512k_rest_minimax-m3", "full_context_512k_sample_minimax-m3")]
                + ["verdicts/{j}/full_context_minimax-m3"], "claude-sonnet-5-5")
r82 = [q for q in run1 if G.get(q, {}).get("context_tier") == "512k"]
k1 = sum(run1[q] for q in r82) - sum(run1[q] for q in CH if q in r82)
print(f"  released 512k verdicts (all files of the reported run, any scope): {len(r82)} items, {k1} ({100 * k1 / len(r82):.1f})")
print("  runs 2 and 3 are separate runs; the first-call rule concerns the reported run only")
