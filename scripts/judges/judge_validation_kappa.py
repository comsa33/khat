"""Judge agreement with human labels on the 80- and 200-prediction validation samples.
"""
import json, sys
from pathlib import Path
from khat.paths import RESULTS
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "judges"))
from panel_verdicts import kappa, load
JV = RESULTS / "human_labels/judge_validation_80.jsonl"
V, ITEMS, _ = load("full")
H = [json.loads(l) for l in open(JV)]
PK = {h["qa_id"]: h["judges"]["kimi-k2.6"] for h in H}
ws = lambda s: " ".join((s or "").split())
rows, why = [], {}
for h in H:
    q, m, s = h["qa_id"].split("__"); s2 = {"rag": "bm25", "fc": "fc"}[s]
    it = ITEMS.get((s2, m), {}).get(q)
    if it is None: why[h["qa_id"]] = "not in the rescoring scope"; continue
    if ws(it["prediction"]) != ws(h["prediction"]): why[h["qa_id"]] = "different prediction"; continue
    if ws(it["gold"]) != ws(h["gold"]): why[h["qa_id"]] = "gold changed"; continue
    rows.append((h["author"] == "Y", V["kimi"][(s2, m)][q], h["qa_id"]))
print("comparable", len(rows), "of 80; excluded:", {k: sum(1 for v in why.values() if v == k) for k in set(why.values())})
allpk = [(h["author"] == "Y", PK[h["qa_id"]]) for h in H if h["qa_id"] in PK]
print(f"panel kimi vs author on all {len(allpk)}: {sum(a == b for a, b in allpk)}/{len(allpk)}, kappa {kappa([a for a, _ in allpk], [b for _, b in allpk]):.3f}")
print("excluded ids:", sorted(why.items()))

import csv
from panel_verdicts import AGG81, HUMAN
V0, I0, _ = load("none")
D = HUMAN / "judge_validation_200"
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8-sig")))
S = {r["sample_id"]: r for r in rd(D / "sample.csv")}
A_, B_ = ({r["sample_id"]: r["label"].strip() for r in rd(D / f"returned/annotator_{x}.csv")} for x in ("A", "B"))
ADJ = {r["sample_id"]: r["label_adjudicated"].strip() for r in rd(D / "adjudication_filled.csv")}
ref_l, J = [], {"claude-sonnet-5-5": [], "gpt-4.1-mini": [], "kimi": []}
for sid, r in S.items():
    if r["stratum"] != "main": continue
    ref = A_[sid] if (A_[sid] == B_[sid] and A_[sid] in ("correct", "incorrect")) else ADJ.get(sid)
    s, m, q = {"open-region": "openregion"}.get(r["setting"], r["setting"]), r["model"], r["qa_id"]
    if s == "fc" and (s, m) in V0["kimi"] and q not in V0["kimi"][(s, m)] and q in AGG81: s = "agg512k"
    if q not in V0["kimi"].get((s, m), {}) or ref not in ("correct", "incorrect"): continue
    if ws(I0[(s, m)][q]["prediction"]) != ws(r["prediction"]): continue
    ref_l.append(ref == "correct")
    for j in J: J[j].append(V0[j][(s, m)][q])
for j, v in J.items():
    print(f"200-sample, {j} vs reference ({len(ref_l)} comparable): {sum(a == b for a, b in zip(ref_l, v))}/{len(ref_l)}, kappa {kappa(ref_l, v):.3f}")
