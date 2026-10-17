"""Expert review of the 110 sampled test items (SI tab:expert): agreement of the expert's answers with the gold under the three judges, no answers on Q1-Q5 by task family, error rates and the annotator rates on the same items.

Usage: python3 scripts/data/expert_review.py
An error is a no on Q1, Q2 or Q3. Rates are weighted by the size of each stratum among the 342 evaluated test_public items,
with a stratified bootstrap interval (10,000 resamples, seed 20261013); per-family intervals are Wilson 95%.
"""
import csv
import json
import math
import random
from collections import Counter

from khat.paths import DATA, RESULTS

RETRIEVAL = ("long_context_retrieval", "long_distance_retrieval")
stratum = lambda fam: "retrieval" if fam in RETRIEVAL else fam

E = list(csv.DictReader(open(RESULTS / "human_labels/expert_review/labels.csv", encoding="utf-8")))
excluded = {l.strip() for l in open(DATA / "exclusions/eval.txt", encoding="utf-8") if l.strip() and not l.startswith("#")}
fam = {}
N = Counter()
for l in open(DATA / "qa_test_public.jsonl", encoding="utf-8"):
    q = json.loads(l)
    fam[q["qa_id"]] = q["task_type"]
    if q["qa_id"] not in excluded:
        N[stratum(q["task_type"])] += 1
TOTAL = sum(N.values())
S = [r["qa_id"] for r in E]
R = {r["qa_id"]: r for r in E}
strata = {}
for s in S:
    strata.setdefault(stratum(fam[s]), []).append(s)
sampled = [h for h, v in strata.items() if len(v) < N[h]]
J = {}
for l in open(RESULTS / "human_labels/expert_review/answer_judge_verdicts.jsonl", encoding="utf-8"):
    x = json.loads(l)
    J.setdefault(x["judge"], {})[x["qa_id"]] = x["judge_correct"]
assert all(sorted(v) == sorted(S) for v in J.values())
err = {s: "no" in (R[s]["q1_one_reading"], R[s]["q2_evidence_supports"], R[s]["q3_correct_complete"]) for s in S}

ann = {}
for l in open(RESULTS / "human_labels/test_review/annotator_labels.jsonl", encoding="utf-8"):
    x = json.loads(l)
    ann[(x["qa_id"], x["annotator"])] = x["error_label"]
adj = {json.loads(l)["qa_id"]: json.loads(l)["final_error"] == "yes"
       for l in open(RESULTS / "human_labels/test_review/adjudication.jsonl", encoding="utf-8")}


def wilson(k, n, z=1.959963984540054):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - h) / d, 100 * (c + h) / d


def wrate(f, g):
    return 100 * sum(N[h] / TOTAL * sum(f(s) for s in v) / len(v) for h, v in g.items())


def wboot(f):
    rng = random.Random(20261013); bs = []
    for _ in range(10000):
        bs.append(wrate(f, {h: (v if h not in sampled else [rng.choice(v) for _ in v]) for h, v in strata.items()}))
    bs.sort()
    return bs[249], bs[9749]


def report(name, f):
    lo, hi = wboot(f)
    print(f"{name}: {sum(f(s) for s in S)}/{len(S)}, weighted {wrate(f, strata):.1f}% [{lo:.1f}, {hi:.1f}]")


no = lambda s, c: R[s][c] == "no"
print(f"items {len(S)}; evaluated test_public items {TOTAL}; strata {dict(N)}")
for j, v in J.items():
    print(f"expert answer agrees with the gold, judge {j}: {sum(v.values())}/{len(v)}")
print(f"all judges agree with the gold: {sum(all(J[j][s] for j in J) for s in S)}/{len(S)}")
print("family: n | Q1 no | Q4 no | Q5 no | errors, rate % [Wilson 95%]")
for h, v in strata.items():
    k = sum(err[s] for s in v); lo, hi = wilson(k, len(v))
    sub = Counter(fam[s] for s in v)
    print(f"  {h}{' ' + str(dict(sub)) if len(sub) > 1 else ''}: {len(v)} | {sum(no(s, 'q1_one_reading') for s in v)} | "
          f"{sum(no(s, 'q4_natural') for s in v)} | {sum(no(s, 'q5_realistic') for s in v)} | {k}, {100 * k / len(v):.1f} [{lo:.1f}, {hi:.1f}]")
print(f"all: Q1 no {sum(no(s, 'q1_one_reading') for s in S)}, Q4 no {sum(no(s, 'q4_natural') for s in S)}, "
      f"Q5 no {sum(no(s, 'q5_realistic') for s in S)}")
print(f"Q2 yes {sum(R[s]['q2_evidence_supports'] == 'yes' for s in S)}, Q3 yes {sum(R[s]['q3_correct_complete'] == 'yes' for s in S)}, "
      f"Q3 unsure {[s for s in S if R[s]['q3_correct_complete'] == 'unsure']}")
print(f"errors that are Q1 no: {sum(err[s] and no(s, 'q1_one_reading') for s in S)} of {sum(err.values())}")
report("expert errors", lambda s: err[s])
report("annotators, adjudicated", lambda s: adj.get(s, False))
report("annotators, any of three", lambda s: any(ann[(s, a)] for a in ("author", "annotator2", "annotator3")))
print(f"expert errors no annotator flagged: {[s for s in S if err[s] and not any(ann[(s, a)] for a in ('author', 'annotator2', 'annotator3'))]}")
