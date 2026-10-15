#!/usr/bin/env python3
"""Judge prompt (question, gold, prediction; YES/NO) and a gpt-4.1-mini judge run through the OpenAI Batch API.

Usage: python3 scripts/judges/llm_judge.py submit --pred FILE | status --tag T | fetch --tag T
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from khat.paths import ROOT, LOCAL
B = LOCAL / "runs" / "judge"
GOLD = {sp: ROOT / "data" / f"qa_{sp}.jsonl" for sp in ("dev", "test_public")}

JUDGE_INSTR = (
    "당신은 한국어 QA 채점자입니다. [질문]에 대한 [정답]과 [모델답변]을 비교하세요.\n"
    "모델답변이 정답과 의미상 동일하거나 정답의 핵심 사실을 올바르게 담고 있으면 정답입니다. "
    "표현·형식·어순·부연설명 차이는 무시하고 핵심 사실의 일치만 보세요. "
    "모델이 '자료로 확정 불가' 등으로 답을 거부했는데 정답이 실제 값이면 오답입니다.\n"
    "판정을 'YES'(정답) 또는 'NO'(오답) 한 단어로만 출력하세요."
)


def load(p: Path) -> list:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def client():
    import openai
    return openai.OpenAI()


def gold_index() -> dict:
    g = {}
    for sp, f in GOLD.items():
        for d in load(f):
            d["_split"] = sp
            g[d["qa_id"]] = d
    return g


def track_path(tag: str) -> Path:
    return B / f"judge_track_{tag}.json"


def judge_prompt(q: str, gold: str, pred: str) -> str:
    return f"{JUDGE_INSTR}\n\n[질문] {q}\n[정답] {gold}\n[모델답변] {pred or '(빈 답)'}\n\n판정:"


def submit(args) -> int:
    tag = args.tag or Path(args.pred).stem
    preds = {r["qa_id"]: r.get("prediction", "") for r in load(ROOT / args.pred if not Path(args.pred).is_absolute() else Path(args.pred))}
    gi = gold_index()
    reqs = []
    for qid, pred in preds.items():
        d = gi.get(qid)
        if not d:
            continue
        body = {"model": args.judge_model,
                "messages": [{"role": "user", "content": judge_prompt(d.get("question", ""), str(d.get("answer", "")), pred)}],
                "max_tokens": 4, "temperature": 0.0}
        reqs.append({"custom_id": f"judge__{qid}", "method": "POST", "url": "/v1/chat/completions", "body": body})
    if not reqs:
        raise SystemExit("no judgeable predictions (public splits only)")
    B.mkdir(parents=True, exist_ok=True)
    inp = B / f"judge_input_{tag}.jsonl"
    inp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in reqs) + "\n", encoding="utf-8")
    cl = client()
    up = cl.files.create(file=inp.open("rb"), purpose="batch")
    bt = cl.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions", completion_window="24h",
                           metadata={"tag": tag, "judge_model": args.judge_model})
    track_path(tag).write_text(json.dumps({"tag": tag, "judge_model": args.judge_model,
                                           "batch_id": bt.id, "n": len(reqs)}, ensure_ascii=False, indent=2), encoding="utf-8")
    inp.unlink(missing_ok=True)
    print(f"[ok] judge submitted: {len(reqs)} items -> batch {bt.id} ({bt.status}); tag={tag}")
    return 0


def status(args) -> int:
    tr = json.loads(track_path(args.tag).read_text(encoding="utf-8"))
    bt = client().batches.retrieve(tr["batch_id"])
    rc = bt.request_counts
    print(f"  {tr['batch_id']}: {bt.status} ({rc.completed}/{rc.total}, {rc.failed} failed)")
    return 0


def fetch(args) -> int:
    tr = json.loads(track_path(args.tag).read_text(encoding="utf-8"))
    cl = client()
    bt = cl.batches.retrieve(tr["batch_id"])
    if bt.status != "completed":
        print(f"  not ready: {bt.status}")
        return 0
    verdict = {}
    for line in cl.files.content(bt.output_file_id).text.splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        qid = o["custom_id"].split("__", 1)[1]
        body = (o.get("response") or {}).get("body") or {}
        txt = (body.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        verdict[qid] = "YES" in txt.upper()
    out = B / f"{args.tag}.judged.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for qid, ok in verdict.items():
            f.write(json.dumps({"qa_id": qid, "judge_correct": ok}, ensure_ascii=False) + "\n")

    gi = gold_index()
    agg = defaultdict(lambda: [0, 0])
    for qid, ok in verdict.items():
        d = gi.get(qid, {})
        for key in ("ALL", f"tier:{d.get('context_tier')}", f"task:{d.get('task_type')}"):
            agg[key][0] += int(ok); agg[key][1] += 1
    print(f"[ok] judged {len(verdict)} -> {out.name} (judge={tr['judge_model']})")
    for grp in ("ALL", "tier:", "task:"):
        for k in sorted(x for x in agg if x == grp or x.startswith(grp)):
            c, n = agg[k]
            print(f"  {k:34s} judge-acc {c/n:5.1%} ({c}/{n})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    s = sub.add_parser("submit"); s.add_argument("--pred", required=True)
    s.add_argument("--judge-model", default="gpt-4.1-mini"); s.add_argument("--tag", default=None)
    s.set_defaults(fn=submit)
    for nm in ("status", "fetch"):
        q = sub.add_parser(nm); q.add_argument("--tag", required=True)
        q.set_defaults(fn={"status": status, "fetch": fetch}[nm])
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
