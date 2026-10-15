"""Scoring helpers (plain, cluster-weighted, Wilson) on the test scope without the legal family and the five excluded items, and the verdicts of the re-runs and the Claude runs.
"""
import json, math
from pathlib import Path
from khat.paths import ROOT as V, RESULTS
REL = RESULTS / "verdicts"
LEGAL = "cross_document_legal_reasoning"
def L(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
G = {}
for sp in ("dev", "test_public"):
    for d in L(V / f"data/qa_{sp}.jsonl"): G[d["qa_id"]] = d
EXCL = set(json.load(open(V / "data/errata/test_review_gold_fixes.json"))["exclude_from_evaluation"])
def inscope(q):
    g = G.get(q); return bool(g) and g.get("task_type") != LEGAL and q not in EXCL
def wilson(c, n):
    if n <= 0: return (0.0, 0.0)
    z = 1.96; p = c / n; d = 1 + z*z/n; centre = (p + z*z/(2*n)) / d
    half = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / d
    return (max(0.0, centre-half), min(1.0, centre+half))
def score(verdicts, ids=None):
    """verdicts: {qa_id: bool}. ids: restrict to these qa_ids (else all in scope). Returns dict."""
    q = [k for k in verdicts if inscope(k) and (ids is None or k in ids)]
    n = len(q); hits = sum(bool(verdicts[k]) for k in q)
    wn = sum(float(G[k].get("cluster_weight", 1.0)) for k in q)
    wh = sum(float(G[k].get("cluster_weight", 1.0)) for k in q if verdicts[k])
    lo, hi = wilson(wh, wn); plo, phi = wilson(hits, n)
    return {"n": n, "hits": hits, "plain": hits/n if n else 0, "cw": wh/wn if wn else 0, "cw_ci": (lo, hi), "plain_ci": (plo, phi)}
def rel(judge, tag):
    p = REL / judge / f"{tag}.jsonl"
    return {r["qa_id"]: bool(r["judge_correct"]) for r in L(p)} if p.exists() else None
def fmt(s, ci=False):
    t = f"{s['plain']*100:.0f}% / {s['cw']*100:.0f}%"
    if ci: t += f" [{s['cw_ci'][0]*100:.0f}, {s['cw_ci'][1]*100:.0f}]"
    return t + f" (n={s['n']})"
_REG = (("cb", "closed_book"), ("bm25", "bm25"), ("dense", "dense"), ("fc", "full_context"), ("agg512k", "full_context_aggregation"),
        ("e3region", "open_region"), ("e3oracle", "oracle"))
_E1 = ("gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5")
RERUNS = {f"{w}_{r}": f"{f}_{m}" for w, m in (("sonnet55", "claude-sonnet-5-5"), ("haiku55", "claude-haiku-5-5")) for r, f in _REG}
RERUNS.update({f"{r}_{m}": f"{f}_{m}" for m in ("gpt-5.5", "gpt-4.1-mini", "glm-5.2", "minimax-m3") for r, f in _REG[5:]})
RERUNS.update({f"e1_{m}_test": f"closed_book_no_row_ids_{m}_test_public" for m in _E1 + ("glm-5.2", "minimax-m3")})
RERUNS.update({f"e1_{m}_dev": f"closed_book_no_row_ids_{m}_dev" for m in _E1})
RERUNS.update({"second_call_test": "closed_book_second_call_gpt-5.5_test_public", "second_call_dev": "closed_book_second_call_gpt-5.5_dev"})
RERUNS.update({f"e2_glm-5.2_{r}": f"{'closed_book' if r == 'cb' else r}_glm-5.2" for r in ("cb", "bm25", "dense")})
def rerun_verdicts():
    """{"claude"|"gpt": {tag: {"split|qa_id": verdict}}} of the re-runs and the Claude runs, from the released verdict files."""
    out = {}
    for j, judge in (("claude", "claude-sonnet-5-5"), ("gpt", "gpt-4.1-mini")):
        out[j] = {}
        for tag, name in RERUNS.items():
            sp = "dev" if name.endswith("_dev") else "test_public"
            out[j][tag] = {f"{sp}|{r['qa_id']}": bool(r["judge_correct"]) for r in L(REL / judge / f"{name}.jsonl")}
    return out
