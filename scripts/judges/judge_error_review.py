#!/usr/bin/env python3
"""Human review of judge disagreements: over- and under-credit rates per judge and developer group.
"""
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "judges"))
from panel_verdicts import AGG81, ANTH, FULLWIN, G, MODELS, HUMAN, SETTINGS, C, agg81, load

wilson = C.wilson
L = HUMAN / "judge_error_review"
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8-sig")))
V, ITEMS, APPLIED = load("full")
J = ["claude-sonnet-5-5", "gpt-4.1-mini", "kimi"]
KEY = {r["sample_id"]: r for r in rd(L / "key.csv")}
A = {r["sample_id"]: r["label"].strip() for r in rd(L / "returned/annotator_A.csv")}
B = {r["sample_id"]: r["label"].strip() for r in rd(L / "returned/annotator_B.csv")}
assert set(A) == set(B) == set(KEY) and len(KEY) == 193
assert set(A.values()) | set(B.values()) <= {"correct", "incorrect", "unsure"}
CARRY = rd(L / "earlier_labels.csv")


def kappa_cat(x, y):
    n = len(x)
    cats = sorted(set(x) | set(y))
    po = sum(a == b for a, b in zip(x, y)) / n
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe)


def newcombe(k1, n1, k2, n2):
    """difference p1 - p2 with the Newcombe hybrid-score 95% interval (method 10), in points."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = (x / 100 for x in wilson(k1, n1))
    l2, u2 = (x / 100 for x in wilson(k2, n2))
    d = p1 - p2
    return 100 * d, 100 * (d - ((p1 - l1) ** 2 + (u2 - p2) ** 2) ** 0.5), 100 * (d + ((u1 - p1) ** 2 + (p2 - l2) ** 2) ** 0.5)


ALL = [(s, m, q) for (s, m), kv in sorted(V["kimi"].items()) for q in sorted(kv)]
vote = lambda k: tuple(V[j][k[:2]][k[2]] for j in J)
D = [k for k in ALL if len(set(vote(k))) > 1]
Dset = set(D)
assert len(ALL) == 10722 and len(D) == 141

H, HU, src = {}, {}, {}
for sid, r in KEY.items():
    k = (r["setting"], r["model"], r["qa_id"])
    lab = A[sid] if A[sid] == B[sid] and A[sid] != "unsure" else None
    (H if r["set"] == "D" else HU)[k] = None if lab is None else lab == "correct"
    src[k] = "now"
for r in CARRY:
    k = (r["setting"], r["model"], r["qa_id"])
    assert k in Dset and k not in H
    H[k] = (r["label_kept"] == "correct") if r["source"] == "A2=A3" else None
    src[k] = "earlier " + r["source"]
assert set(H) == Dset and len(HU) == 100

print("== judge error review (first-call rule applied)")
print(f"D = {len(D)}: labeled here {sum(src[k] == 'now' for k in D)}, labeled earlier and agreed {sum(src[k] == 'earlier A2=A3' for k in D)}, "
      f"labeled earlier and unresolved {sum(src[k] == 'earlier author' for k in D)}")
print("  per model: " + ", ".join(f"{m} {sum(k[1] == m for k in D)}" for m in MODELS))
print("  per setting: " + ", ".join(f"{s} {sum(k[0] == s for k in D)}" for s in SETTINGS))
print("  per model x setting: " + "; ".join(f"{m} " + " ".join(f"{s}:{n}" for s in SETTINGS
                                                                   if (n := sum(k[:2] == (s, m) for k in D)))
                                             for m in MODELS))

# ---- annotator agreement
xs = [A[s] for s in sorted(KEY)]
ys = [B[s] for s in sorted(KEY)]
print("\n== annotator agreement (193 rows: 93 D + 100 U)")
print(f"  raw {sum(a == b for a, b in zip(xs, ys))}/193; Cohen's kappa (correct / incorrect / unsure) {kappa_cat(xs, ys):.3f}")
for grp in ("D", "U"):
    ids = [s for s in sorted(KEY) if KEY[s]["set"] == grp]
    x, y = [A[s] for s in ids], [B[s] for s in ids]
    print(f"  {grp}: raw {sum(a == b for a, b in zip(x, y))}/{len(ids)}, kappa {kappa_cat(x, y):.3f}")
both = [(A[s], B[s]) for s in KEY if "unsure" not in (A[s], B[s])]
print(f"  rows where neither said unsure: {len(both)}, agree {sum(a == b for a, b in both)} "
      f"(correct-vs-incorrect disagreements {sum(a != b for a, b in both)})")
print("  pairs: " + ", ".join(f"{a}/{b} {n}" for (a, b), n in sorted(Counter(zip(xs, ys)).items())))
unres_now = [k for k in D if src[k] == "now" and H[k] is None]
print(f"  unresolved in D: {sum(H[k] is None for k in D)} ({sum(src[k] == 'earlier author' for k in D)} labeled earlier + {len(unres_now)} labeled here)")

# ---- judges against the human labels on D
print("\n== each judge against the human labels on D (resolved / bounds over all 141 with unresolved counted as agree or disagree)")
res = [k for k in D if H[k] is not None]
nu = len(D) - len(res)
for j in J:
    ag = sum(V[j][k[:2]][k[2]] == H[k] for k in res)
    over = sum(V[j][k[:2]][k[2]] and not H[k] for k in res)
    under = sum((not V[j][k[:2]][k[2]]) and H[k] for k in res)
    lo, hi = wilson(ag, len(res))
    print(f"  {j}: agrees on {ag}/{len(res)} = {100 * ag / len(res):.1f}% [{lo:.1f}, {hi:.1f}]; bounds over D "
          f"{ag}/{len(D)} = {100 * ag / len(D):.1f}% to {ag + nu}/{len(D)} = {100 * (ag + nu) / len(D):.1f}%; "
          f"over-credit {over}, under-credit {under}")
print("  rates per developer group over all reported predictions of the group (resolved D counts; unresolved range in brackets);")
print("  Newcombe 95% of Anthropic minus others, points")
NG = {g: sum(1 for k in ALL if (k[1] in ANTH) == (g == "anth")) for g in ("anth", "other")}
for j in J:
    for kind in ("over", "under"):
        cnt, rng = {}, {}
        for g in ("anth", "other"):
            ks = [k for k in D if (k[1] in ANTH) == (g == "anth")]
            if kind == "over":
                c = sum(1 for k in ks if H[k] is False and V[j][k[:2]][k[2]])
                u = sum(1 for k in ks if H[k] is None and V[j][k[:2]][k[2]])
            else:
                c = sum(1 for k in ks if H[k] is True and not V[j][k[:2]][k[2]])
                u = sum(1 for k in ks if H[k] is None and not V[j][k[:2]][k[2]])
            cnt[g], rng[g] = c, u
        d, lo, hi = newcombe(cnt["anth"], NG["anth"], cnt["other"], NG["other"])
        print(f"  {j} {kind}-credit: Anthropic {cnt['anth']}/{NG['anth']} = {100 * cnt['anth'] / NG['anth']:.2f}% "
              f"[+{rng['anth']} unresolved], others {cnt['other']}/{NG['other']} = {100 * cnt['other'] / NG['other']:.2f}% "
              f"[+{rng['other']} unresolved]; difference {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        if rng["anth"] or rng["other"]:
            ex = [newcombe(cnt["anth"] + a, NG["anth"], cnt["other"] + b, NG["other"])
                  for a in (0, rng["anth"]) for b in (0, rng["other"])]
            print(f"      with unresolved at the extremes: difference from {min(e[0] for e in ex):+.2f} to "
                  f"{max(e[0] for e in ex):+.2f}; lower bounds {min(e[1] for e in ex):+.2f} to {max(e[1] for e in ex):+.2f}")

# ---- U
print("\n== unanimous verdicts on U (100)")
ures = {k: h for k, h in HU.items() if h is not None}
uerr = sum(h != vote(k)[0] for k, h in ures.items())
ulo, uhi = wilson(uerr, len(ures))
print(f"  resolved {len(ures)}, unresolved {100 - len(ures)}; human label differs from the unanimous verdict on {uerr}/{len(ures)} "
      f"= {100 * uerr / len(ures):.1f}% Wilson [{ulo:.1f}, {uhi:.1f}]; agreement {len(ures) - uerr}/{len(ures)}")
print(f"  with unresolved at the extremes: errors {uerr} to {uerr + 100 - len(ures)} of 100")
print("  per model (resolved, errors): " + ", ".join(f"{m} {sum(k[1] == m for k in ures)}/{sum(k[1] == m and ures[k] != vote(k)[0] for k in ures)}" for m in MODELS))

print("\n== each judge against the error-review labels over all 10,722 predictions (unanimous verdicts taken as correct outside D;"
      " D unresolved range)")
NUN = len(ALL) - len(D)
for j in J:
    ag = sum(V[j][k[:2]][k[2]] == H[k] for k in res)
    a0, a1 = NUN + ag, NUN + ag + nu
    print(f"  {j}: {a0}-{a1}/{len(ALL)} = {100 * a0 / len(ALL):.2f}-{100 * a1 / len(ALL):.2f}%")

# ---- per cell error-review score
pt, lo_, hi_ = {}, {}, {}
for (s, m), kv in V["kimi"].items():
    for nm, fill in (("lo", False), ("hi", True)):
        d = {}
        for q in kv:
            k = (s, m, q)
            d[q] = (fill if H[k] is None else H[k]) if k in Dset else vote(k)[0]
        V.setdefault(nm, {})[(s, m)] = d
print("\n== per cell error-review score: correct count, unresolved range; claude-sonnet-5-5 judge for reference")
for s in SETTINGS:
    cells = []
    for m in MODELS:
        if (s, m) not in V["kimi"]:
            continue
        kv = V["kimi"][(s, m)]
        qs = C.items(s, V["claude-sonnet-5-5"][(s, m)]) if s in ("cb", "bm25", "dense", "fc") else sorted(kv)
        lo, hi = sum(V["lo"][(s, m)][q] for q in qs), sum(V["hi"][(s, m)][q] for q in qs)
        rng = f"{lo}" if lo == hi else f"{lo}-{hi}"
        cells.append(f"{m} {rng}/{len(qs)} (claude-sonnet-5-5 {sum(V['claude-sonnet-5-5'][(s, m)][q] for q in qs)})")
    print(f"  {s}: " + "; ".join(cells))
