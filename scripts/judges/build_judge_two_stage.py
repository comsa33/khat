"""Judge requests (claude-sonnet-5-5, gpt-4.1-mini) and kimi-k2.6 items for the two-stage run predictions.

Usage: python3 scripts/judges/build_judge_two_stage.py <run set dir with manifest.json, preds/, requests/skip_*>   (writes <dir>/judge/)
"""
import csv, hashlib, json, sys
from pathlib import Path
W = Path(sys.argv[1]); J = W / "judge"; J.mkdir(exist_ok=True)
MAN = json.load(open(W / "manifest.json"))
from khat.paths import ROOT as DS, DATA
sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_common import CLAUDE_PARAMS, judge_prompt, multi_answer
MODELS = ["gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5", "minimax-m3", "glm-5.2", "claude-sonnet-5-5", "claude-haiku-5-5"]
ids = lambda f: [l.strip() for l in open(DATA / "item_sets" / f) if l.strip()]
T81, D144 = ids("items_test81.txt"), ids("items_dev144.txt")
COND = dict(MAN["items"])
SKIP = {f.name[len("skip_"):-len(".json")] for f in (W / "requests").glob("skip_*.json")}
if SKIP:
    assert SKIP == {"openai_twostage_gpt-5.5__dev", "openai_devtop5_gpt-5.5"}, SKIP
TESTONLY = {"gpt-5.5"} if SKIP else set()
G = {}
for f in ("qa_test_public.jsonl", "qa_dev.jsonl"):
    for l in open(DS / "data" / f, encoding="utf-8"):
        r = json.loads(l); G[r["qa_id"]] = r
fx = json.load(open(DS / "data/gold_fixes.json", encoding="utf-8"))
dev_any = {r["qa_id"]: [x.strip() for x in r["new_answer_or_accepted"].split(" | ")]
           for r in csv.DictReader(open(DS / "data/errata/dev_review.csv", encoding="utf-8")) if r["status"] == "gold_accept_any"}
def gold(q):
    if q in fx["replace"]: return fx["replace"][q]
    if q in fx["accept_any"]: return multi_answer(fx["accept_any"][q])
    if q in dev_any: return multi_answer(dev_any[q])
    return str(G[q]["answer"])
idx, cl, oa, km, missing = [], [], [], [], []
for c, qs in COND.items():
    for m in MODELS:
        f = W / "preds" / f"{c}_{m}.jsonl"
        P = {json.loads(l)["qa_id"]: json.loads(l) for l in open(f, encoding="utf-8")} if f.exists() else {}
        for q in qs:
            if m in TESTONLY and q not in T81: continue
            if q not in P: missing.append((c, m, q)); continue
            pred = P[q]["prediction"] or ""
            cid = f"R_{len(idx):05d}"; pr = judge_prompt(G[q]["question"], gold(q), pred)
            idx.append({"custom_id": cid, "qa_id": q, "condition": c, "model": m, "split": "test" if q in T81 else "dev", "status": P[q]["status"]})
            cl.append({"custom_id": cid, "params": {**CLAUDE_PARAMS, "messages": [{"role": "user", "content": pr}]}})
            oa.append({"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4.1-mini",
                       "messages": [{"role": "user", "content": pr}], "max_tokens": 4, "temperature": 0.0}})
            km.append({"qa_id": f"{q}__{c}_{m}", "question": G[q]["question"], "gold": gold(q), "prediction": pred, "src_tag": f"planR_{c}_{m}"})
if missing:
    sys.exit(f"missing predictions: {len(missing)} (first {missing[:3]})")
for name, rows in (("index.jsonl", idx), ("claude_requests.jsonl", cl), ("openai_batch_input.jsonl", oa), ("items_kimi.jsonl", km)):
    with open(J / name, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows), hashlib.sha256((J / name).read_bytes()).hexdigest())
if MAN["set"] == "main":
    assert len(idx) == (3600 - 288 if SKIP else 3600), len(idx)
else:
    assert len(idx) == sum(len(v) for v in COND.values()) * len(MODELS), len(idx)
