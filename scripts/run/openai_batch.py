#!/usr/bin/env python3
"""OpenAI Batch API runs: build the request file of a prompt file (submit --dry-run), submit it, check status, fetch predictions.

Usage: python3 scripts/run/openai_batch.py submit --model M --regimes R --prompt-file F --max-output-tokens N [--dry-run] | status --model M | fetch --model M
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_baseline import select_prompt, _openai_chat_kwargs, _REASONING_OVERRIDE

from khat.paths import LOCAL
B = LOCAL / "runs" / "openai_batch"
PROMPTS = {
    "cb": LOCAL / "runs" / "closed_book" / "prompts_main.jsonl",
    "rag": LOCAL / "runs" / "rag_bm25" / "tp386.jsonl",
    "fc": LOCAL / "runs" / "full_context" / "groupA_tp.jsonl",
}
SPLITS = ("test_public", "dev")
MAX_BYTES = 180 * 1024 * 1024


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", s)


def client():
    import openai
    return openai.OpenAI()


def load_jsonl(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def track_path(model: str, suffix: str = "") -> Path:
    return B / f"batch_track_{safe(model)}{suffix}.json"


# --------------------------------------------------------------------------- submit
def submit(args) -> int:
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    reasoning = _REASONING_OVERRIDE[args.reasoning]
    requests: list[dict] = []
    counts: dict = defaultdict(int)
    for reg in regimes:
        pf = Path(args.prompt_file) if getattr(args, "prompt_file", None) else PROMPTS[reg]
        if not pf.exists():
            raise SystemExit(f"missing prompt file for regime {reg}: {pf}")
        for rec in load_jsonl(pf):
            if rec.get("split") not in SPLITS:
                continue
            prompt = select_prompt(rec)
            body = _openai_chat_kwargs(args.model, prompt, args.temperature, args.max_output_tokens, reasoning)
            requests.append({
                "custom_id": f"{reg}__{rec['split']}__{rec['qa_id']}",
                "method": "POST", "url": "/v1/chat/completions", "body": body,
            })
            counts[f"{reg}/{rec['split']}"] += 1

    if args.limit:
        requests = requests[: args.limit]
    if not requests:
        raise SystemExit("no requests built (check prompt files / splits)")

    cl = None if args.dry_run else client()
    batches: list[dict] = []
    chunk: list[str] = []
    size = 0
    part = 0

    def flush():
        nonlocal chunk, size, part
        if not chunk:
            return
        B.mkdir(parents=True, exist_ok=True)
        inp = B / f"batch_input_{safe(args.model)}_{part}.jsonl"
        inp.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        mb = inp.stat().st_size / 1e6
        if args.dry_run:
            print(f"[dry-run] part {part}: {len(chunk)} requests, {mb:.1f}MB -> {inp.name} (NOT uploaded)")
            batches.append({"batch_id": None, "input_file_id": None, "part": part, "n": len(chunk), "dry": True})
        else:
            up = cl.files.create(file=inp.open("rb"), purpose="batch")
            bt = cl.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                                   completion_window="24h",
                                   metadata={"model": args.model, "part": str(part)})
            batches.append({"batch_id": bt.id, "input_file_id": up.id, "part": part, "n": len(chunk)})
            print(f"[submit] part {part}: {len(chunk)} requests, {mb:.1f}MB -> batch {bt.id} ({bt.status})")
        chunk, size, part = [], 0, part + 1

    for r in requests:
        line = json.dumps(r, ensure_ascii=False)
        b = len(line.encode("utf-8")) + 1
        if chunk and size + b > MAX_BYTES:
            flush()
        chunk.append(line)
        size += b
    flush()

    if args.dry_run:
        print(f"[dry-run] built {len(requests)} requests across {len(batches)} file(s); counts={dict(counts)}. "
              "Nothing uploaded; no batch created; no tracking file written.")
        return 0

    track_path(args.model, getattr(args, 'track_suffix', '') or '').write_text(json.dumps(
        {"model": args.model, "regimes": regimes, "max_output_tokens": args.max_output_tokens,
         "reasoning": args.reasoning, "counts": dict(counts), "batches": batches},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] submitted {len(requests)} requests in {len(batches)} batch(es); "
          f"track: {track_path(args.model, getattr(args, 'track_suffix', '') or '').name}. Check: status; then fetch when completed.")
    return 0


# --------------------------------------------------------------------------- status
def status(args) -> int:
    tp = track_path(args.model, getattr(args, 'track_suffix', '') or '')
    if not tp.exists():
        raise SystemExit(f"no tracking file: {tp} (submit first)")
    track = json.loads(tp.read_text(encoding="utf-8"))
    cl = client()
    for b in track["batches"]:
        bt = cl.batches.retrieve(b["batch_id"])
        rc = bt.request_counts
        print(f"  {b['batch_id']} part{b['part']}: {bt.status} "
              f"({rc.completed}/{rc.total} done, {rc.failed} failed)"
              f"{' output=' + bt.output_file_id if bt.output_file_id else ''}")
    return 0


# --------------------------------------------------------------------------- fetch
def fetch(args) -> int:
    tp = track_path(args.model, getattr(args, 'track_suffix', '') or '')
    if not tp.exists():
        raise SystemExit(f"no tracking file: {tp}")
    track = json.loads(tp.read_text(encoding="utf-8"))
    cl = client()
    model_label = safe(args.model)
    bucket: dict = defaultdict(list)
    errbucket: dict = defaultdict(list)
    pending = 0
    for b in track["batches"]:
        bt = cl.batches.retrieve(b["batch_id"])
        if bt.status != "completed" or not bt.output_file_id:
            print(f"  part{b['part']} {b['batch_id']}: {bt.status} — skipping (not ready)")
            pending += 1
            continue
        text = cl.files.content(bt.output_file_id).text
        (B / f"batch_raw_{model_label}_part{b['part']}.jsonl").write_text(text, encoding="utf-8")
        for line in text.splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            cid = o.get("custom_id", "")
            reg, split, qa_id = cid.split("__", 2)
            resp = o.get("response") or {}
            body = resp.get("body") or {}
            err = o.get("error") or body.get("error")
            if err or resp.get("status_code") not in (None, 200) or not body.get("choices"):
                msg = err.get("message") if isinstance(err, dict) else (str(err) if err else f"status {resp.get('status_code')}")
                errbucket[f"{reg}__{split}"].append((qa_id, msg))
                continue
            pred = (body.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            bucket[f"{reg}__{split}"].append((qa_id, pred.strip(), body.get("usage") or {}))

        if getattr(bt, "error_file_id", None):
            etext = cl.files.content(bt.error_file_id).text
            (B / f"batch_raw_{model_label}_part{b['part']}.errors.jsonl").write_text(etext, encoding="utf-8")
            for line in etext.splitlines():
                if not line.strip():
                    continue
                o = json.loads(line)
                cid = o.get("custom_id", "")
                if not cid:
                    continue
                reg, split, qa_id = cid.split("__", 2)
                err = o.get("error") or ((o.get("response") or {}).get("body") or {}).get("error")
                msg = err.get("message") if isinstance(err, dict) else (str(err) if err else "error")
                errbucket[f"{reg}__{split}"].append((qa_id, msg))

    for key in sorted(set(bucket) | set(errbucket)):
        reg, split = key.split("__", 1)
        items, errs = bucket.get(key, []), errbucket.get(key, [])
        suffix = getattr(args, "out_suffix", "") or ""
        out = B / f"{reg}_{model_label}_{split}{suffix}.jsonl"
        calls = B / f"{reg}_{model_label}_{split}{suffix}.calls.jsonl"
        with out.open("w", encoding="utf-8") as of, calls.open("w", encoding="utf-8") as cf:
            for qa_id, pred, usage in items:
                of.write(json.dumps({"qa_id": qa_id, "prediction": pred}, ensure_ascii=False) + "\n")
                det = usage.get("completion_tokens_details") or {}
                cf.write(json.dumps({
                    "qa_id": qa_id, "prediction": pred,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": det.get("reasoning_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }, ensure_ascii=False) + "\n")
            for qa_id, msg in errs:
                cf.write(json.dumps({"qa_id": qa_id, "prediction": None, "error": msg}, ensure_ascii=False) + "\n")
        note = f" + {len(errs)} errored ({errs[0][1][:50]}…)" if errs else ""
        print(f"[ok] {key}: {len(items)} preds -> {out.name}{note}")
    if pending:
        print(f"[note] {pending} batch(es) not ready yet; re-run fetch later.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    sp = sub.add_parser("submit")
    sp.add_argument("--model", required=True)
    sp.add_argument("--regimes", default="cb,rag,fc")
    sp.add_argument("--temperature", type=float, default=0.0)
    sp.add_argument("--max-output-tokens", type=int, default=256,
                    help="raise for reasoning models (gpt-5*/o) so thinking doesn't starve the answer")
    sp.add_argument("--reasoning", choices=["auto", "on", "off"], default="auto")
    sp.add_argument("--dry-run", action="store_true", help="build the input JSONL only; no upload/batch")
    sp.add_argument("--limit", type=int, default=None, help="cap requests (smoke-test the live batch cycle)")
    sp.add_argument("--prompt-file", default=None,
                    help="override the prompt file for the listed regime(s) (use with a single regime, "
                         "e.g. the HUG-injected fc subset)")
    sp.add_argument("--track-suffix", default="",
                    help="suffix for the tracking file so a partial re-run does not clobber the main "
                         "run's tracking (use the same suffix on status/fetch)")
    sp.set_defaults(fn=submit)
    for name in ("status", "fetch"):
        q = sub.add_parser(name)
        q.add_argument("--model", required=True)
        q.add_argument("--track-suffix", default="", help="match the suffix used at submit")
        if name == "fetch":
            q.add_argument("--out-suffix", default="",
                           help="suffix for output filenames so a partial re-run "
                                "writes to a separate file instead of clobbering the full results")
        q.set_defaults(fn={"status": status, "fetch": fetch}[name])
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
