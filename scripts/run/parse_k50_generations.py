"""Answers and generation report of the dense top-50 run: applies the retries of empty answers, counts stop reasons, empty and at-cap answers and token medians.

Usage: python3 scripts/run/parse_k50_generations.py   (reads results/runs/k50/generations.jsonl; writes local/runs/k50/preds/ and gen_report.json)
"""
import json, statistics, collections
from pathlib import Path
from khat.paths import RESULTS, LOCAL
HERE = RESULTS / "runs/k50"
OUT = LOCAL / "runs/k50"
L = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
CAP = {"gpt-5.5": 16000, "gpt-4.1-mini": 4000, "claude-sonnet-5-5": 16384, "claude-haiku-5-5": 16384, "glm-5.2": 16384, "minimax-m3": 32768, "minimax-m3-reasoning": 32768}
out = {}; rep = {}
GEN = collections.defaultdict(dict)
for r in L(HERE / "generations.jsonl"):
    m, q = r.pop("model"), r.pop("qa_id"); GEN[m][q] = r
src = {m: (lambda m=m: GEN[m]) for m in GEN}
(OUT / "preds").mkdir(parents=True, exist_ok=True)
for m, fn in src.items():
    try: d = fn()
    except FileNotFoundError: print("not ready", m); continue
    if not d: print("not ready", m); continue
    retry = HERE / f"retry_{m}.json"
    rsub = json.load(open(retry)) if retry.exists() else {}
    for q, t in rsub.items():
        assert not d[q]["text"], ("retry on non-empty", m, q); d[q].update({"text": t, "retried": True})
    ins = [v["in"] for v in d.values() if v["in"]]; outs = [v["out"] for v in d.values() if v["out"] is not None]
    rep[m] = {"n": len(d), "stop": dict(collections.Counter(str(v["stop"]) for v in d.values())),
              "empty": sorted(q for q, v in d.items() if not v["text"]),
              "at_cap": sorted(q for q, v in d.items() if v["out"] is not None and v["out"] >= CAP[m]),
              "retried": sorted(rsub), "median_in": statistics.median(ins) if ins else None,
              "max_out": max(outs) if outs else None, "median_out": statistics.median(outs) if outs else None,
              "sum_in": sum(ins), "sum_out": sum(outs)}
    if m.startswith("claude"): rep[m]["nonempty_thinking_blocks"] = sum(1 for v in d.values() if v.get("thinking_block"))
    with open(OUT / "preds" / f"{m}.jsonl", "w", encoding="utf-8") as f:
        for q in sorted(d): f.write(json.dumps({"qa_id": q, "prediction": d[q]["text"]}, ensure_ascii=False) + "\n")
json.dump(rep, open(OUT / "gen_report.json", "w"), indent=1)
for m, r in rep.items(): print(m, {k: (v if not isinstance(v, list) else len(v)) for k, v in r.items()})
