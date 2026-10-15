"""Strict string match against the judge on the 512k full-context items of four models.

Usage: python3 scripts/analysis/strict_vs_judge.py results/verdicts/gpt-4.1-mini_original_gold
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "score"))
from score_answers import metrics
from khat.paths import DATA, RESULTS
vd, B = sys.argv[1], RESULTS / "predictions"
XI = set(open(os.environ["EXCL_IDS"]).read().split()) if os.environ.get("EXCL_IDS") else set()
gold = {}
for l in open(DATA / "qa_test_public.jsonl", encoding="utf-8"):
    if l.strip():
        d = json.loads(l)
        if d.get("task_type") != "cross_document_legal_reasoning" and d["qa_id"] not in XI:
            gold[d["qa_id"]] = d
RUNS = [("gpt-5.5", "full_context_512k_sample_gpt-5.5", B / "full_context_512k_sample_gpt-5.5.jsonl"),
        ("glm-5.2", "full_context_512k_sample_glm-5.2", B / "full_context_512k_sample_glm-5.2.jsonl"),
        ("minimax-m3", "full_context_512k_sample_minimax-m3", B / "full_context_512k_sample_minimax-m3.jsonl"),
        ("gpt-4.1-mini", "full_context_512k_sample_gpt-4.1-mini", B / "full_context_512k_sample_gpt-4.1-mini.jsonl")]
def strict(item, pred):
    m = metrics(item, pred, 0.7); mt = item.get("evaluation", {}).get("metric", "")
    return m["num"] if mt == "exact_numbers" else m["em"] if mt == "exact_match" else m["contains"]
for name, tag, pf in RUNS:
    preds = {json.loads(l)["qa_id"]: json.loads(l).get("prediction", "") for l in pf.open(encoding="utf-8") if l.strip()}
    jv = {json.loads(l)["qa_id"]: bool(json.loads(l)["judge_correct"]) for l in (Path(vd) / f"{tag}.jsonl").open(encoding="utf-8") if l.strip()}
    ids = [q for q in jv if q in gold and gold[q]["context_tier"] == "512k"]
    js = [q for q in ids if jv[q] and not strict(gold[q], preds.get(q, ""))]
    sj = [q for q in ids if not jv[q] and strict(gold[q], preds.get(q, ""))]
    c1813 = jv.get("krhlrb_v05_1813", "absent")
    print(f"{name:13} n={len(ids)} judge_ok&strict_fail={len(js)} {sorted(q[-4:] for q in js)} | strict_ok&judge_fail={len(sj)} {sorted(q[-4:] for q in sj)} | 1813 judge={c1813} strict={strict(gold['krhlrb_v05_1813'], preds.get('krhlrb_v05_1813','')) if 'krhlrb_v05_1813' in preds else 'n/a'}")
