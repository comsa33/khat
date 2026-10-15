#!/usr/bin/env python3
"""Agreement of kimi-k2.6 with the released verdicts, the 200-prediction validation sample, and self-preference.
"""
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "judges"))
from panel_verdicts import (
    AGG81, ANTH, FULLWIN, G, MODELS, HUMAN, R, SETTINGS, C, agg81, jl, kappa, load)

wilson = C.wilson
TIERS = ["32k", "64k", "128k", "256k", "512k"]
V, ITEMS, APPLIED = load("full")
V0, _, _ = load("none")
JN = ["claude-sonnet-5-5", "gpt-4.1-mini", "kimi"]
pct = lambda v, qs: 100 * sum(v[q] for q in qs) / len(qs)
cnt = lambda v, qs: sum(v[q] for q in qs)


def main_items(s, m):
    return C.items(s, V["claude-sonnet-5-5"][(s, m)])


print("re-called predictions scored by their first call (wrong under all judges):", ", ".join(f"{s} {m} {q[-4:]}" for s, m, q in APPLIED))


# ------------------------------------------------------------------ 1. accuracy, agreement, 200 sample
print("\n== claude-sonnet-5-5 / kimi accuracy per model x setting (plain %, n; first-call rule applied)")
for m in MODELS:
    cells = []
    for s in SETTINGS:
        if (s, m) in V["kimi"]:
            qs = sorted(V["kimi"][(s, m)])
            cells.append(f"{s} {pct(V['claude-sonnet-5-5'][(s, m)], qs):.1f}/{pct(V['kimi'][(s, m)], qs):.1f} (n={len(qs)})")
    print(f"  {m}: " + " | ".join(cells))
print("\n== kimi agreement with the claude-sonnet-5-5 judge per model (all settings; raw %, kappa)")
allp, allk = [], []
for m in MODELS:
    a, b = [], []
    for s in SETTINGS:
        if (s, m) in V["kimi"]:
            for q in sorted(V["kimi"][(s, m)]):
                a.append(V["claude-sonnet-5-5"][(s, m)][q]); b.append(V["kimi"][(s, m)][q])
    allp += a; allk += b
    ag = sum(x == y for x, y in zip(a, b))
    print(f"  {m}: {ag}/{len(a)} = {100*ag/len(a):.1f}%, kappa {kappa(a, b):.3f}; claude-sonnet-5-5 YES/kimi NO {sum(x and not y for x, y in zip(a, b))}, claude-sonnet-5-5 NO/kimi YES {sum(y and not x for x, y in zip(a, b))}")
ag = sum(x == y for x, y in zip(allp, allk))
print(f"  all: {ag}/{len(allp)} = {100*ag/len(allp):.1f}%, kappa {kappa(allp, allk):.3f}")

D = HUMAN / "judge_validation_200"
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8-sig")))
S = {r["sample_id"]: r for r in rd(D / "sample.csv")}
A_, B_ = ({r["sample_id"]: r["label"].strip() for r in rd(D / f"returned/annotator_{x}.csv")} for x in ("A", "B"))
ADJ = {r["sample_id"]: r["label_adjudicated"].strip() for r in rd(D / "adjudication_filled.csv")}
SMAP = {"open-region": "openregion"}
ok = tot = miss = 0
dis = []
for sid, r in S.items():
    if r["stratum"] != "main":
        continue
    ref = A_[sid] if (A_[sid] == B_[sid] and A_[sid] in ("correct", "incorrect")) else ADJ.get(sid)
    s, m, q = SMAP.get(r["setting"], r["setting"]), r["model"], r["qa_id"]
    if s == "fc" and (s, m) in V0["kimi"] and q not in V0["kimi"][(s, m)] and q in AGG81:
        s = "agg512k"
    kv = V0["kimi"].get((s, m), {})
    if q not in kv or ref not in ("correct", "incorrect"):
        miss += 1
        continue
    same = " ".join(ITEMS[(s, m)][q]["prediction"].split()) == " ".join(r["prediction"].split())
    if not same:
        miss += 1
        continue
    tot += 1
    if kv[q] == (ref == "correct"):
        ok += 1
    else:
        dis.append((sid, s, m, q[-4:], ref, kv[q]))
lo, hi = wilson(ok, tot)
print(f"\n== kimi (this run) vs reference labels, 200-prediction sample: {ok}/{tot} = {100*ok/tot:.1f}% [{lo:.1f}, {hi:.1f}]; "
      f"not comparable {miss} (outside the evaluated scope or a different prediction); disagreements {dis}")

# ------------------------------------------------------------------ 2. self-preference diff-in-diff
CL = defaultdict(list)
for q in C.TEST:
    CL[G[q]["cluster_id"]].append(q)
KEYS = sorted(CL)
CID = {q: G[q]["cluster_id"] for q in C.TEST}
AGGD = {}
for (s, m), kv in V["kimi"].items():
    a = defaultdict(lambda: [0, 0])
    for q in kv:
        a[CID[q]][0] += V["claude-sonnet-5-5"][(s, m)][q] - kv[q]
        a[CID[q]][1] += 1
    AGGD[(s, m)] = dict(a)
NS = {s: defaultdict(int) for s in SETTINGS}
for s in SETTINGS:
    for q in {q for m in MODELS if (s, m) in V["kimi"] for q in V["kimi"][(s, m)]}:
        NS[s][CID[q]] += 1


def did(mult, group):
    """mult: multiplicity of each cluster in this resample. Returns (headline, {s: contrast_s})."""
    cs, ns = {}, {}
    for s in SETTINGS:
        d = {}
        for m in MODELS:
            if (s, m) not in AGGD:
                continue
            num = den = 0
            for c, (x, n) in AGGD[(s, m)].items():
                k = mult.get(c, 0)
                num += k * x; den += k * n
            if den:
                d[m] = num / den
        an = [d[m] for m in ANTH if m in d]
        ot = [d[m] for m in group if m in d]
        if an and ot:
            cs[s] = sum(an) / len(an) - sum(ot) / len(ot)
            ns[s] = sum(mult.get(c, 0) * n for c, n in NS[s].items())
    return sum(cs[s] * ns[s] for s in cs) / sum(ns[s] for s in cs), cs


NONA = [m for m in MODELS if m not in ANTH]
NONA_FW = ["gpt-4.1-mini", "gpt-5.5", "minimax-m3", "glm-5.2"]
print("\n== self-preference diff-in-diff (claude-sonnet-5-5 minus kimi accuracy, Anthropic minus non-Anthropic; points; first-call rule applied)")
print("   item weight of a setting = its number of distinct items (cb 342, bm25/dense 332, fc 96, agg512k 63, openregion/oracle/k50 81, k50page 18)")
for label, grp_ in (("all non-Anthropic models that ran each setting", NONA), ("sensitivity: four non-Anthropic full-window models", NONA_FW)):
    h, cs = did({c: 1 for c in KEYS}, grp_)
    rnd = random.Random(20261013)
    bs = []
    for _ in range(10000):
        mult = defaultdict(int)
        for k in (rnd.choice(KEYS) for _ in KEYS):
            mult[k] += 1
        bs.append(did(mult, grp_)[0])
    bs.sort()
    print(f"  {label}: headline {100*h:+.2f} [{100*bs[249]:+.2f}, {100*bs[9749]:+.2f}] (clusters {len(KEYS)})")
    print("    per setting: " + ", ".join(f"{s} {100*v:+.2f}" for s, v in cs.items()))
print("\nper model d(m, s) = claude-sonnet-5-5 - kimi (points):")
for m in MODELS:
    print(f"  {m}: " + ", ".join(f"{s} {pct(V['claude-sonnet-5-5'][(s, m)], list(V['kimi'][(s, m)])) - pct(V['kimi'][(s, m)], list(V['kimi'][(s, m)])):+.1f}"
                                for s in SETTINGS if (s, m) in V["kimi"]))
