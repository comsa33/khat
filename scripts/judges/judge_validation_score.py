#!/usr/bin/env python3
"""Judges against the two annotators' reference labels on the 200-prediction validation sample.

Usage: python3 scripts/judges/judge_validation_score.py --sample FILE --ann-a FILE --ann-b FILE --adjudication FILE --dataset .
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--sample", required=True)
ap.add_argument("--ann-a", required=True)
ap.add_argument("--ann-b", required=True)
ap.add_argument("--adjudication")
ap.add_argument("--adjudication-template")
ap.add_argument("--dataset", required=True)
a = ap.parse_args()

DS = Path(a.dataset).resolve()
PANEL = ["deepseek-v4.1-flash", "gemma4_31b", "kimi-k2.6", "mistral-large-3"]
PANEL_TAG = {"cb": "panel/panel_{j}_closed_book_{m}", "bm25": "panel/panel_{j}_bm25_{m}",
             "dense": "panel/panel_{j}_dense_{m}", "fc": "panel/panel_{j}_full_context_{m}",
             "k50": "panel/panel_{j}_k50_{m}", "open-region": "panel/panel_{j}_open_region_{m}",
             "oracle": "panel/panel_{j}_oracle_{m}"}


def read(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def kappa(x, y):
    n = len(x)
    if n == 0:
        return float("nan")
    po = sum(i == j for i, j in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% ({k}/{n}) [{100 * lo:.1f}, {100 * hi:.1f}]" if n else "n/a"


truthy = lambda v: str(v).strip().lower() in ("true", "1", "yes", "y", "correct")
sample = read(a.sample)
A = {r["sample_id"]: r["label"].strip().lower() for r in read(a.ann_a)}
B = {r["sample_id"]: r["label"].strip().lower() for r in read(a.ann_b)}
ok = {"correct", "incorrect", "unsure"}
bad = [(s, v) for d in (A, B) for s, v in d.items() if v not in ok]
if bad or set(A) != {r["sample_id"] for r in sample} or set(B) != set(A):
    sys.exit(f"label files do not match the sample or contain other values: {bad[:5]}")

adj = {}
if a.adjudication:
    adj = {r["sample_id"]: r["label_adjudicated"].strip().lower() for r in read(a.adjudication)}
need = [r for r in sample if A[r["sample_id"]] != B[r["sample_id"]] or "unsure" in (A[r["sample_id"]], B[r["sample_id"]])]
if a.adjudication_template:
    with open(a.adjudication_template, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "label_a", "label_b", "label_adjudicated", "note"])
        w.writeheader()
        for r in need:
            w.writerow({"sample_id": r["sample_id"], "label_a": A[r["sample_id"]], "label_b": B[r["sample_id"]],
                        "label_adjudicated": "", "note": ""})
    print(f"wrote adjudication template: {len(need)} rows")

ref = {}
for r in sample:
    s = r["sample_id"]
    if A[s] == B[s] and A[s] != "unsure":
        ref[s] = A[s] == "correct"
    elif adj.get(s) in ("correct", "incorrect"):
        ref[s] = adj[s] == "correct"
missing = [r["sample_id"] for r in sample if r["sample_id"] not in ref]

out = []
P = out.append
P("## Judge validation (200-prediction sample)")
P(f"- rows {len(sample)} (main {sum(r['stratum'] == 'main' for r in sample)}, "
  f"aux {sum(r['stratum'] != 'main' for r in sample)}); needing adjudication {len(need)}; "
  f"without a reference label {len(missing)}")

both = [r["sample_id"] for r in sample if "unsure" not in (A[r["sample_id"]], B[r["sample_id"]])]
xa = [A[s] == "correct" for s in both]
xb = [B[s] == "correct" for s in both]
agree = sum(i == j for i, j in zip(xa, xb))
P(f"- annotators: unsure A {sum(v == 'unsure' for v in A.values())}, B {sum(v == 'unsure' for v in B.values())}, "
  f"both {sum(A[s] == B[s] == 'unsure' for s in A)}; raw agreement on rows without unsure {fmt(agree, len(both))}; "
  f"Cohen kappa {kappa(xa, xb):.3f}")


def block(rows, label, weights=None, with_kappa=True):
    rows = [r for r in rows if r["sample_id"] in ref]
    res = []
    for jname, col in (("claude-sonnet-5-5", "judge_label"), ("gpt-4.1-mini", "judge_label_secondary")):
        j = [truthy(r[col]) for r in rows]
        h = [ref[r["sample_id"]] for r in rows]
        k = sum(x == y for x, y in zip(j, h))
        line = f"{jname} {fmt(k, len(rows))}" + (f", kappa {kappa(j, h):.3f}" if with_kappa else "")
        if weights:
            wsum = sum(weights[(r["model"], r["setting"])] for r in rows)
            wk = sum(weights[(r["model"], r["setting"])] * (x == y) for r, x, y in zip(rows, j, h))
            line += f"; post-stratified {100 * wk / wsum:.1f}%"
        res.append(line)
    P(f"- {label} (n={len(rows)}): " + " | ".join(res))


sys.path.insert(0, str(Path(__file__).resolve().parent))
import os
cwd = os.getcwd()
os.chdir(DS)
import judge_validation_sample as sj
sj.ROOT = DS
sj.P = DS / "results" / "predictions"
qa = {json.loads(l)["qa_id"]: json.loads(l) for l in open(DS / "data/qa_test_public.jsonl")}
excl = {l.strip() for l in open(DS / "data/exclusions/eval_test_and_dev_sample.txt") if l.strip() and l[0] != "#"}
scope = {q for q, r in qa.items() if r["task_type"] != "cross_document_legal_reasoning" and q not in excl}
fc_items = {r["qa_id"] for r in sj.load(sj.FC_ITEMS_FROM)}
agg = {json.loads(l)["qa_id"] for l in open(DS / "data/item_sets/agg81.jsonl")}
h80 = set()
for r in [json.loads(l) for l in open(DS / "results" / "human_labels" / "judge_validation_80.jsonl", encoding="utf-8")]:
    q, m, s = r["qa_id"].split("__")
    h80.add((q, m, {"rag": "bm25"}.get(s, s)))
pool = {}
for setting, by_model in sj.SETTINGS.items():
    for model, files in by_model.items():
        ids = set()
        for fn in files:
            ids |= {r["qa_id"] for r in sj.load(fn) if r.get("split", "test_public") == "test_public"}
        ids &= scope
        if setting == "fc":
            ids &= fc_items
        if setting in ("k50", "open-region", "oracle"):
            ids &= agg
        pool[(model, setting)] = len({q for q in ids if (q, model, setting) not in h80})
os.chdir(cwd)
per_cell = Counter((r["model"], r["setting"]) for r in sample if r["stratum"] == "main")
weights = {c: pool[c] / per_cell[c] for c in per_cell}

main = [r for r in sample if r["stratum"] == "main"]
block(main, "main stratum, overall", weights)
P("- by setting:")
for s in sj.SETTINGS:
    block([r for r in main if r["setting"] == s], f"  {s}", with_kappa=False)
P("- by model:")
for m in sj.CB:
    block([r for r in main if r["model"] == m], f"  {m}", with_kappa=False)
P("- by family:")
for f, sel in (("cross_source_aggregation", lambda t: t == "cross_source_aggregation"),
               ("long_context_retrieval", lambda t: t == "long_context_retrieval"),
               ("other", lambda t: t not in ("cross_source_aggregation", "long_context_retrieval"))):
    block([r for r in main if sel(r["task_type"])], f"  {f}", with_kappa=False)

anth = [r for r in main if r["model"].startswith("claude") and r["sample_id"] in ref]
pv = {}
for jn in PANEL:
    for s, tpl in PANEL_TAG.items():
        for m in ("claude-sonnet-5-5", "claude-haiku-5-5"):
            fp = DS / "results" / "verdicts" / (tpl.format(j=jn, m=m) + ".jsonl")
            for l in open(fp):
                d = json.loads(l)
                pv[(jn, s, m, d["qa_id"])] = bool(d["judge_correct"])
P(f"- six judges on the Anthropic predictions of the main stratum (n={len(anth)}):")
for jn, get in [("claude-sonnet-5-5", lambda r: truthy(r["judge_label"])), ("gpt-4.1-mini", lambda r: truthy(r["judge_label_secondary"]))] + \
        [(jn, (lambda jn: lambda r: pv[(jn, r["setting"], r["model"], r["qa_id"])])(jn)) for jn in PANEL]:
    j = [get(r) for r in anth]
    h = [ref[r["sample_id"]] for r in anth]
    P(f"  - {jn}: {fmt(sum(x == y for x, y in zip(j, h)), len(anth))}, kappa {kappa(j, h):.3f}")

aux = [r for r in sample if r["stratum"] != "main"]
if aux:
    block(aux, "auxiliary string-mismatch stratum (reported separately)")

print("\n".join(out))
if missing:
    print(f"\nrows without a reference label (adjudicate first): {missing[:20]}{' ...' if len(missing) > 20 else ''}")
