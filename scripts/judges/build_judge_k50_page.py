"""Judge requests (claude-sonnet-5-5, gpt-4.1-mini) and panel items for the top-50 plus page predictions.

Usage: python3 scripts/judges/build_judge_k50_page.py ROUND MODEL[,MODEL...]   (reads local/runs/k50page/preds/)
"""
import json, sys, hashlib
from pathlib import Path
from khat.paths import ROOT as V, LOCAL
HERE = LOCAL / "runs" / "k50page"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_common import judge_prompt, judge_gold, gold_fixes, CLAUDE_PARAMS
L = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
fix = gold_fixes(); TEST = {d["qa_id"]: d for d in L(V / "data/qa_test_public.jsonl")}
AGG = sorted(json.loads(l)["qa_id"] for l in open(HERE / "prompts_k50page_18.jsonl"))
MODELS = sys.argv[2].split(",")
RND = sys.argv[1]
J = HERE / "judge"; J.mkdir(exist_ok=True); (J / "panel_items").mkdir(exist_ok=True)
idx, cl, gp = [], [], []
for m in MODELS:
    preds = {r["qa_id"]: r["prediction"] for r in L(HERE / "preds" / f"{m}.jsonl")}
    assert sorted(preds) == sorted(AGG), m
    panel = []
    for q in AGG:
        tag = f"k50page_{m}"; cid = "j" + hashlib.sha1(f"{tag}|{q}".encode()).hexdigest()[:20]
        gold = judge_gold(TEST[q], fix, "multi"); p = judge_prompt(TEST[q].get("question", ""), gold, preds[q])
        idx.append({"custom_id": cid, "tag": tag, "model": m, "qa_id": q})
        cl.append({"custom_id": cid, "params": {**CLAUDE_PARAMS, "messages": [{"role": "user", "content": p}]}})
        gp.append({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions",
                   "body": {"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": p}], "max_tokens": 4, "temperature": 0.0}})
        panel.append({"qa_id": q, "question": TEST[q].get("question", ""), "gold": gold, "prediction": preds[q]})
    if m.startswith("claude"):
        with open(J / "panel_items" / f"{'sonnet55' if 'sonnet' in m else 'haiku55'}_k50page.jsonl", "w", encoding="utf-8") as f:
            for r in panel: f.write(json.dumps(r, ensure_ascii=False) + "\n")
for name, rows in ((f"index_{RND}.jsonl", idx), (f"claude_requests_{RND}.jsonl", cl), (f"gpt_requests_{RND}.jsonl", gp)):
    with open(J / name, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("requests", len(idx), "panel files", len(list((J / "panel_items").glob("*.jsonl"))))
