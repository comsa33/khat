"""Two-stage and same-day verdicts of the three judges: parse raw judge outputs, export the per-run verdict files (results/verdicts/<judge>/{run}_{model}_{split}.jsonl), load them, list missing verdicts.

Usage: python3 scripts/two_stage/judge_parse.py export|missing <judge dir>
"""
import json, sys
from pathlib import Path
from khat.paths import RESULTS
JUDGES = ["sonnet", "gpt-4.1-mini", "kimi"]
MAX_ROUNDS = 2
_src = (Path(__file__).resolve().parent.parent / "judges/judge_batches.py").read_text(encoding="utf-8")
_g = {"__name__": "judge_batch_functions"}
exec(_src.split('if cmd == "build":')[0].replace("cmd = sys.argv[1]", "cmd = None"), _g)
claude_results, jl, wjl = _g["claude_results"], _g["jl"], _g["wjl"]


def parse_raw(J):
    J = Path(J); idx = {i["custom_id"]: i for i in jl(J / "index.jsonl")}; V = {j: {} for j in JUDGES}
    for cid, r in claude_results(J).items():
        if cid not in idx: continue
        txt = "".join(b.get("text", "") for b in r["result"]["message"]["content"] if b.get("type") == "text")
        try:
            V["sonnet"][cid] = json.loads(txt)["verdict"] == "YES"
        except Exception:
            pass
    for f in [J / "raw_openai.jsonl"] + [J / f"raw_openai_missing{k}.jsonl" for k in range(1, MAX_ROUNDS + 1)]:
        if not f.exists(): continue
        for r in jl(f):
            body = (r.get("response") or {}).get("body") or {}
            if r.get("error") or not body.get("choices") or r["custom_id"] not in idx: continue
            V["gpt-4.1-mini"][r["custom_id"]] = "YES" in body["choices"][0]["message"]["content"].upper()
    key = {f"{i['qa_id']}__{i['condition']}_{i['model']}": cid for cid, i in idx.items()}
    for f in [J / "kimi_R.jsonl"] + [J / f"kimi_R_missing{k}.jsonl" for k in range(1, MAX_ROUNDS + 1)]:
        if not f.exists(): continue
        for r in jl(f):
            if r.get("judge_correct") is not None and r["qa_id"] in key: V["kimi"][key[r["qa_id"]]] = bool(r["judge_correct"])
    V["missing"] = {j: sorted(c for c in idx if c not in V[j]) for j in JUDGES}
    return V


VERDICT_DIRS = {"sonnet": "claude-sonnet-5-5", "gpt-4.1-mini": "gpt-4.1-mini", "kimi": "kimi-k2.6"}
RUN_FILE = {"twostage": "two_stage", "bm25test": "bm25_rerun", "devtop5": "bm25_rerun",
            "bv1": "same_day_bv1", "bv2": "same_day_bv2", "bv3": "same_day_bv3"}
SPLIT = {"test": "test_public", "dev": "dev"}


def verdict_file(i):
    """Released verdict file stem of an index row: {run}_{model}_{split}."""
    return f"{RUN_FILE[i['condition']]}_{i['model']}_{SPLIT[i['split']]}"


def load(J, verdicts=RESULTS / "verdicts"):
    J = Path(J); idx = {i["custom_id"]: i for i in jl(J / "index.jsonl")}; V = {j: {} for j in JUDGES}
    cid = {(verdict_file(i), i["qa_id"]): c for c, i in idx.items()}
    for j, d in VERDICT_DIRS.items():
        for name in sorted({verdict_file(i) for i in idx.values()}):
            p = Path(verdicts) / d / f"{name}.jsonl"
            if p.exists():
                V[j].update({cid[(name, r["qa_id"])]: bool(r["judge_correct"]) for r in jl(p)})
        V[j] = {c: V[j][c] for c in sorted(V[j])}
    V["missing"] = {j: sorted(c for c in idx if c not in V[j]) for j in JUDGES}
    return V


def cmd_export(J, verdicts=RESULTS / "verdicts"):
    J = Path(J); idx = {i["custom_id"]: i for i in jl(J / "index.jsonl")}; V = parse_raw(J)
    for j, d in VERDICT_DIRS.items():
        rows = {}
        for c in sorted(V[j]):
            rows.setdefault(verdict_file(idx[c]), []).append({"qa_id": idx[c]["qa_id"], "judge_correct": V[j][c]})
        (Path(verdicts) / d).mkdir(parents=True, exist_ok=True)
        for name, rs in rows.items():
            wjl(Path(verdicts) / d / f"{name}.jsonl", rs)


REQ = {"sonnet": ("claude_requests.jsonl", "claude_missing{k}.jsonl", "raw_claude_missing{k}.jsonl", "custom_id"),
       "gpt-4.1-mini": ("openai_batch_input.jsonl", "openai_missing{k}.jsonl", "raw_openai_missing{k}.jsonl", "custom_id"),
       "kimi": ("items_kimi.jsonl", "items_kimi_missing{k}.jsonl", "kimi_R_missing{k}.jsonl", "qa_id")}


def rounds(J, j):
    return sum((Path(J) / REQ[j][1].format(k=k)).exists() for k in range(1, MAX_ROUNDS + 1))


def cmd_missing(J):
    J = Path(J); V = load(J); idx = {i["custom_id"]: i for i in jl(J / "index.jsonl")}; out = {}
    for j in JUDGES:
        miss, k = V["missing"][j], rounds(J, j)
        src, nxt, raw, kf = REQ[j]
        if k and not (J / raw.format(k=k)).exists():
            out[j] = f"round {k} pending: run {nxt.format(k=k)}"; continue
        if not miss:
            out[j] = "complete"; continue
        if k >= MAX_ROUNDS:
            out[j] = f"final: {len(miss)} missing after {MAX_ROUNDS} resubmissions -> counted incorrect for {j}"; continue
        want = set(miss) if kf == "custom_id" else {f"{idx[c]['qa_id']}__{idx[c]['condition']}_{idx[c]['model']}" for c in miss}
        rows = [r for r in jl(J / src) if r[kf] in want]
        wjl(J / nxt.format(k=k + 1), rows); out[j] = f"wrote {nxt.format(k=k + 1)} ({len(rows)} requests, same settings)"
    for j, s in out.items(): print(f"  {j}: {s}")
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["missing"]: cmd_missing(sys.argv[2])
    elif sys.argv[1:2] == ["export"]: cmd_export(sys.argv[2])
    else: sys.exit(__doc__)
