"""Judge requests (claude-sonnet-5-5, gpt-4.1-mini) and kimi-k2.6 items for the same-day re-run predictions.

Usage: python3 scripts/judges/build_judge_same_day.py <run dir with manifest.json, preds/>   (writes <dir>/judge/)
"""
import csv, hashlib, json, sys
from pathlib import Path
W = Path(sys.argv[1]); J = W / "judge"; J.mkdir(exist_ok=True)
MAN = json.load(open(W / "manifest.json"))
from khat.paths import ROOT as DS
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_common import CLAUDE_PARAMS, judge_prompt
MODELS = ["gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5", "minimax-m3", "glm-5.2", "claude-sonnet-5-5", "claude-haiku-5-5"]
G = {json.loads(l)["qa_id"]: json.loads(l) for l in open(DS / "data/qa_dev.jsonl", encoding="utf-8")}
fx = json.load(open(DS / "data/gold_fixes.json", encoding="utf-8"))
dev_any = {r["qa_id"] for r in csv.DictReader(open(DS / "data/errata/dev_review.csv", encoding="utf-8")) if r["status"] == "gold_accept_any"}
idx, cl, oa, km, missing = [], [], [], [], []
for rule, qs in MAN["items"].items():
    for m in MODELS:
        f = W / "preds" / f"{rule}_{m}.jsonl"
        P = {json.loads(l)["qa_id"]: json.loads(l) for l in open(f, encoding="utf-8")} if f.exists() else {}
        for q in qs:
            assert q not in fx["replace"] and q not in fx["accept_any"] and q not in dev_any, q
            if q not in P: missing.append((rule, m, q)); continue
            pred = P[q]["prediction"] or ""; gold = str(G[q]["answer"])
            cid = f"B_{len(idx):05d}"; pr = judge_prompt(G[q]["question"], gold, pred)
            idx.append({"custom_id": cid, "qa_id": q, "condition": rule, "model": m, "split": "dev", "status": P[q]["status"]})
            cl.append({"custom_id": cid, "params": {**CLAUDE_PARAMS, "messages": [{"role": "user", "content": pr}]}})
            oa.append({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4.1-mini",
                       "messages": [{"role": "user", "content": pr}], "max_tokens": 4, "temperature": 0.0}})
            km.append({"qa_id": f"{q}__{rule}_{m}", "question": G[q]["question"], "gold": gold, "prediction": pred, "src_tag": f"planB_{rule}_{m}"})
if missing: sys.exit(f"missing predictions: {len(missing)} (first {missing[:3]})")
for name, rows in (("index.jsonl", idx), ("claude_requests.jsonl", cl), ("openai_batch_input.jsonl", oa), ("items_kimi.jsonl", km)):
    with open(J / name, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows), hashlib.sha256((J / name).read_bytes()).hexdigest())
assert len(idx) == 192, len(idx)
