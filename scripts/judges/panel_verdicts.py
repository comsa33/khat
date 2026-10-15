"""Verdicts of the three judges (kimi-k2.6, claude-sonnet-5-5, gpt-4.1-mini) per setting, model and item.
"""
import json
import sys
from pathlib import Path

from khat.paths import ROOT as R, RESULTS
O = RESULTS / "judge_panel"
HUMAN = RESULTS / "human_labels"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "score"))
import verdict_sets as C

MODELS = C.MODELS
ANTH = ["claude-sonnet-5-5", "claude-haiku-5-5"]
FULLWIN = ["gpt-4.1-mini", "gpt-5.5", "minimax-m3", "glm-5.2", "claude-sonnet-5-5", "claude-haiku-5-5"]
SETTINGS = ["cb", "bm25", "dense", "fc", "agg512k", "openregion", "oracle", "k50", "k50page"]
KIMI_FILE = {"cb": "closed_book", "fc": "full_context", "agg512k": "full_context_aggregation", "k50page": "k50_page", "openregion": "open_region"}
JUDGES = {"claude-sonnet-5-5": "claude-sonnet-5-5", "gpt-4.1-mini": "gpt-4.1-mini"}
G = C.G
AGG81 = [json.loads(l)["qa_id"] for l in open(R / "data/item_sets/agg81.jsonl")]
P2 = "verdicts/{j}/"
FC512 = {"gpt-5.5": [P2 + "full_context_512k_rest_gpt-5.5", P2 + "full_context_512k_sample_gpt-5.5"],
         "gpt-4.1-mini": [P2 + "full_context_512k_rest_gpt-4.1-mini", P2 + "full_context_512k_sample_gpt-4.1-mini"],
         "minimax-m3": [P2 + "full_context_512k_rest_minimax-m3", P2 + "full_context_512k_sample_minimax-m3"],
         "glm-5.2": [P2 + "full_context_512k_rest_glm-5.2", P2 + "full_context_512k_sample_glm-5.2"]}
for m in ANTH:
    FC512[m] = ["verdicts/{j}/full_context_aggregation_" + m]
OTHER = {"openregion": "verdicts/{j}/open_region_{m}", "oracle": "verdicts/{j}/oracle_{m}",
         "k50": "verdicts/{j}/k50_{m}", "k50page": "verdicts/{j}/k50_page_{m}"}

import csv
_SHORT = {"closed_book": "cb", "full_context_aggregation": "agg512k"}
FIRST_CALL, FIRST_CALL_CB = {}, {}
for _r in csv.DictReader(open(R / "data/first_call_items.csv", encoding="utf-8")):
    _s = _SHORT.get(_r["setting"], _r["setting"])
    (FIRST_CALL_CB if _s == "cb" else FIRST_CALL).setdefault((_s, _r["model"]), []).append(_r["qa_id"][-4:])


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def released(tags, judge, m=None):
    v = {}
    for t in tags:
        v.update({r["qa_id"]: bool(r["judge_correct"]) for r in jl(R / "results" / (t.format(j=judge, m=m) + ".jsonl"))})
    return v


def load(rule="full"):
    """rule: 'none' (re-call verdicts) | 'bm25_512k' (the 8 BM25 and 512k items scored by their first call) | 'full' (also the 9 closed-book items)."""
    K, items = {}, {}
    for s in SETTINGS:
        for m in MODELS:
            f = RESULTS / "verdicts/kimi-k2.6" / f"{KIMI_FILE.get(s, s)}_{m}.jsonl"
            if not f.exists():
                continue
            rows = jl(f)
            K[(s, m)] = {r["qa_id"]: bool(r["judge_correct"]) for r in rows}
            assert all(r["judge_correct"] is not None for r in rows)
            items[(s, m)] = {json.loads(l)["qa_id"]: json.loads(l) for l in open(RESULTS / "judge_inputs/kimi-k2.6" / f"{KIMI_FILE.get(s, s)}_{m}.jsonl")}
            assert set(items[(s, m)]) == set(K[(s, m)]), (s, m)
    V = {"kimi": K, "claude-sonnet-5-5": {}, "gpt-4.1-mini": {}}
    for jn, j in JUDGES.items():
        for (s, m), kv in K.items():
            if s in ("cb", "bm25", "dense", "fc"):
                v = C.verdicts(s, m, j)
            elif s == "agg512k":
                v = released(FC512[m], j)
            else:
                v = released([OTHER[s]], j, m)
            miss = [q for q in kv if q not in v]
            assert not miss, (jn, s, m, len(miss))
            V[jn][(s, m)] = {q: v[q] for q in kv}
    ov = {}
    if rule in ("bm25_512k", "full"):
        ov.update(FIRST_CALL)
    if rule == "full":
        ov.update(FIRST_CALL_CB)
    applied = []
    for (s, m), qs in ov.items():
        for q in ("krhlrb_v05_" + x for x in qs):
            if q not in K[(s, m)]:
                assert q not in C.TEST or q in C.BEX, (s, m, q)
                continue
            for jn in V:
                V[jn][(s, m)][q] = False
            applied.append((s, m, q))
    return V, items, applied


def agg81(V, jn, m, s):
    """aggregation-set verdicts (81) for a setting; full context = sample aggregation items + agg512k."""
    if s == "fc":
        v = {q: x for q, x in V[jn][("fc", m)].items() if q in AGG81}
        v.update(V[jn][("agg512k", m)])
    else:
        v = V[jn][(s, m)]
    assert all(q in v for q in AGG81), (jn, m, s)
    return {q: v[q] for q in AGG81}


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0
