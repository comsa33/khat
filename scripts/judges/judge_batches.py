#!/usr/bin/env python3
"""Build, submit, fetch and write LLM-judge batches (Anthropic and OpenAI).

Usage: python3 scripts/judges/judge_batches.py build|submit-claude|fetch-claude|submit-openai|fetch-openai|write ...
"""
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

cmd = sys.argv[1]


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def wjl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def claude_sources(W):
    """Every raw_claude_<ID>.jsonl in WORKDIR (batch downloads, fetched batches, or a direct call), with its
    creation time from raw_claude_<ID>.meta.json when present (file mtime otherwise), oldest first."""
    out = []
    for f in W.glob("raw_claude_*.jsonl"):
        sid = f.name[len("raw_claude_"):-len(".jsonl")]
        mp = W / f"raw_claude_{sid}.meta.json"
        meta = json.load(open(mp)) if mp.exists() else {}
        rows = jl(f)
        out.append({"id": meta.get("id", sid), "file": f.name, "created_at": str(meta.get("created_at") or
                    time.strftime("%Y-%m-%d %H:%M:%S+00:00", time.gmtime(f.stat().st_mtime))),
                    "n": len(rows), "results": dict(Counter(r["result"]["type"] for r in rows)), "rows": rows})
    return sorted(out, key=lambda s: s["created_at"])


def claude_results(W):
    """Succeeded results only, merged by custom_id; when several sources have one, the most recently created
    source wins. Canceled and errored results are dropped."""
    merged = {}
    for s in claude_sources(W):
        for r in s["rows"]:
            if r["result"]["type"] == "succeeded":
                merged[r["custom_id"]] = dict(r, _batch=s["id"])
    return merged


if cmd == "build":
    DS, ERR, W = Path(sys.argv[2]).resolve(), Path(sys.argv[3]), Path(sys.argv[4])
    W.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from judge_common import CLAUDE_PARAMS, judge_prompt, multi_answer
    gold = {}
    for r in csv.DictReader(open(ERR, encoding="utf-8")):
        if r["status"] == "gold_accept_any":
            gold[r["qa_id"]] = multi_answer([x.strip() for x in r["new_answer_or_accepted"].split(" | ")])
        elif r["status"] == "gold_replaced":
            gold[r["qa_id"]] = r["new_answer_or_accepted"].strip()
    qa = {r["qa_id"]: r for r in jl(DS / "data/qa_dev.jsonl")}
    P = DS / "results/predictions"
    files = {("gpt-4.1-mini", "cb"): "closed_book_gpt-4.1-mini", ("gpt-5.4-mini", "cb"): "closed_book_gpt-5.4-mini",
             ("gpt-5.4-nano", "cb"): "closed_book_gpt-5.4-nano", ("gpt-5.5", "cb"): "closed_book_gpt-5.5",
             ("gpt-4.1-mini", "bm25"): "bm25_gpt-4.1-mini", ("gpt-5.4-mini", "bm25"): "bm25_gpt-5.4-mini",
             ("gpt-5.4-nano", "bm25"): "bm25_gpt-5.4-nano", ("gpt-5.5", "bm25"): "bm25_gpt-5.5",
             ("gpt-4.1-mini", "fc"): "full_context_gpt-4.1-mini", ("gpt-5.4-mini", "fc"): "full_context_gpt-5.4-mini",
             ("gpt-5.4-nano", "fc"): "full_context_gpt-5.4-nano", ("gpt-5.5", "fc"): "full_context_gpt-5.5"}
    retry = {r["qa_id"]: r for r in jl(DS / "results/predictions/closed_book_second_call_gpt-5.5_dev.jsonl")}
    prompts, index = [], []
    for (m, s), fn in files.items():
        for r in jl(P / f"{fn}.jsonl"):
            if r["split"] != "dev" or r["qa_id"] not in gold:
                continue
            src = fn
            if (m, s) == ("gpt-5.5", "cb") and r["qa_id"] in retry:
                r, src = retry[r["qa_id"]], "closed_book_second_call_gpt-5.5_dev"
            cid = f"v097_{len(prompts):04d}"
            prompts.append({"custom_id": cid, "prompt": judge_prompt(qa[r["qa_id"]]["question"], gold[r["qa_id"]],
                                                                      r["prediction"])})
            index.append({"custom_id": cid, "qa_id": r["qa_id"], "model": m, "setting": s, "file": src,
                          "old_primary": r["judge_label"], "old_secondary": r["judge_label_secondary"]})
    wjl(W / "prompts.jsonl", prompts)
    wjl(W / "index.jsonl", index)
    wjl(W / "claude_requests.jsonl", [{"custom_id": p["custom_id"], "params": {
        **CLAUDE_PARAMS, "messages": [{"role": "user", "content": p["prompt"]}]}} for p in prompts])
    wjl(W / "openai_batch_input.jsonl", [{"custom_id": p["custom_id"], "method": "POST", "url": "/v1/chat/completions",
                                         "body": {"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": p["prompt"]}],
                                                  "max_tokens": 4, "temperature": 0.0}} for p in prompts])
    print("requests", len(prompts), Counter((i["model"], i["setting"]) for i in index),
          "items", len({i["qa_id"] for i in index}), "prompt chars", sum(len(p["prompt"]) for p in prompts))

elif cmd == "submit-claude":
    import anthropic
    W = Path(sys.argv[2])
    missing = "--missing" in sys.argv
    track = W / ("batch_claude_missing.json" if missing else "batch_claude.json")
    assert not track.exists(), "already submitted: " + str(track)
    reqs = jl(W / "claude_requests.jsonl")
    if missing:
        have = set(claude_results(W))
        reqs = [r for r in reqs if r["custom_id"] not in have]
        if not reqs:
            sys.exit("nothing missing: every request has a succeeded result")
    c = anthropic.Anthropic()
    b = c.messages.batches.create(requests=reqs)
    json.dump({"id": b.id, "n": len(reqs), "custom_ids": [r["custom_id"] for r in reqs] if missing else "ALL",
               "base_url": str(c.base_url), "submitted": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, open(track, "w"))
    print("submitted", b.id, len(reqs), "endpoint", c.base_url)

elif cmd == "fetch-claude":
    import anthropic
    W = Path(sys.argv[2])
    if "--batch" in sys.argv:
        bid = sys.argv[sys.argv.index("--batch") + 1]
    else:
        bid = json.load(open(W / ("batch_claude_missing.json" if "--missing" in sys.argv else "batch_claude.json")))["id"]
    c = anthropic.Anthropic()
    b = c.messages.batches.retrieve(bid)
    print(f"endpoint {c.base_url} | batch {b.id} | status {b.processing_status} | created {b.created_at} | "
          f"ended {b.ended_at} | counts {b.request_counts}")
    if b.id != bid:
        sys.exit(f"retrieved id {b.id} differs from requested {bid}")
    if b.processing_status != "ended":
        sys.exit("not ended; nothing saved")
    rows = [json.loads(r.model_dump_json()) for r in c.messages.batches.results(bid)]
    wjl(W / f"raw_claude_{bid}.jsonl", rows)
    summ = Counter(r["result"]["type"] for r in rows)
    json.dump({"id": bid, "created_at": str(b.created_at), "ended_at": str(b.ended_at), "n": len(rows),
               "results": dict(summ), "not_succeeded": [r["custom_id"] for r in rows if r["result"]["type"] != "succeeded"],
               "endpoint": str(c.base_url), "fetched": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
              open(W / f"raw_claude_{bid}.meta.json", "w"), indent=1)
    print(f"saved raw_claude_{bid}.jsonl:", dict(summ))

elif cmd == "submit-openai":
    import openai
    W = Path(sys.argv[2])
    track = W / "batch_openai.json"
    assert not track.exists(), "already submitted"
    c = openai.OpenAI()
    up = c.files.create(file=open(W / "openai_batch_input.jsonl", "rb"), purpose="batch")
    b = c.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions", completion_window="24h")
    json.dump({"id": b.id, "file": up.id, "submitted": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, open(track, "w"))
    print("submitted", b.id)

elif cmd == "fetch-openai":
    import openai
    W = Path(sys.argv[2])
    t = json.load(open(W / "batch_openai.json"))
    c = openai.OpenAI()
    b = c.batches.retrieve(t["id"])
    if b.status != "completed":
        sys.exit(f"not completed: {b.status} {b.request_counts}")
    (W / "raw_openai.jsonl").write_bytes(c.files.content(b.output_file_id).read())
    if b.error_file_id:
        (W / "raw_openai_errors.jsonl").write_bytes(c.files.content(b.error_file_id).read())
    print("saved", b.request_counts)

elif cmd == "write":
    W, OUT = Path(sys.argv[2]), Path(sys.argv[3])
    idx = {i["custom_id"]: i for i in jl(W / "index.jsonl")}
    res, log = {"claude-sonnet-5-5": {}, "gpt-4.1-mini": {}}, {"failures": {}, "usage": {}}
    fails, ui, uo = [], 0, 0
    merged = claude_results(W)
    for cid in idx:
        if cid not in merged:
            fails.append(cid)
            continue
        rr = merged[cid]["result"]
        if rr["type"] != "succeeded":
            fails.append(cid)
            continue
        msg = rr["message"]
        txt = "".join(b.get("text", "") for b in msg["content"] if b.get("type") == "text")
        try:
            res["claude-sonnet-5-5"][cid] = json.loads(txt)["verdict"] == "YES"
        except Exception:
            fails.append(cid)
        ui += msg["usage"]["input_tokens"]
        uo += msg["usage"]["output_tokens"]
    srcs = claude_sources(W)
    log["claude_batches"] = [{"id": s["id"], "file": s["file"], "created_at": s["created_at"], "n": s["n"],
                              "results": s["results"],
                              "used": sum(1 for r in merged.values() if r["_batch"] == s["id"])} for s in srcs]
    dup = Counter(r["custom_id"] for s in srcs for r in s["rows"] if r["result"]["type"] == "succeeded")
    log["claude_custom_ids_with_several_succeeded_results"] = sorted(c for c, n in dup.items() if n > 1)
    log["failures"]["claude-sonnet-5-5"], log["usage"]["claude-sonnet-5-5"] = fails, {"input": ui, "output": uo}
    fails, ui, uo = [], 0, 0
    for r in jl(W / "raw_openai.jsonl"):
        cid = r["custom_id"]
        body = (r.get("response") or {}).get("body") or {}
        if r.get("error") or not body.get("choices"):
            fails.append(cid)
            continue
        res["gpt-4.1-mini"][cid] = "YES" in body["choices"][0]["message"]["content"].upper()
        ui += body["usage"]["prompt_tokens"]
        uo += body["usage"]["completion_tokens"]
    log["failures"]["gpt-4.1-mini"], log["usage"]["gpt-4.1-mini"] = fails, {"input": ui, "output": uo}
    for j, d in res.items():
        by_file = {}
        for cid, v in d.items():
            by_file.setdefault(idx[cid]["file"], []).append({"qa_id": idx[cid]["qa_id"], "judge_correct": v})
        (OUT / j).mkdir(parents=True, exist_ok=True)
        for fn, rows in by_file.items():
            wjl(OUT / j / f"{fn}_dev_corrected_gold.jsonl", sorted(rows, key=lambda x: x["qa_id"]))
    flips = Counter()
    for cid, i in idx.items():
        for j, col in (("claude-sonnet-5-5", "old_primary"), ("gpt-4.1-mini", "old_secondary")):
            if cid in res[j]:
                flips[(j, f"{i[col]}->{res[j][cid]}")] += 1
    log.update({"n_requests": len(idx), "verdicts": {j: len(d) for j, d in res.items()},
                "flips": {f"{a} {b}": n for (a, b), n in flips.items()}, "written": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    json.dump(log, open(W / "run_log.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(log, ensure_ascii=False, indent=1))
else:
    sys.exit(__doc__)
