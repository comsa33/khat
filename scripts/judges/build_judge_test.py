#!/usr/bin/env python3
"""Judge prompts for every judged test run (prediction layers as in the initial gpt-4.1-mini verdicts) and the multi-answer prompts of the accept-any items.

Usage: python3 scripts/judges/build_judge_test.py --out local/runs/judge_test [--baselines DIR] [--accept-any-format or|multi]
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

from judge_common import (BASELINES, LEGAL, INITIAL_VERDICTS, ROOT, gold_fixes, judge_gold, judge_prompt,
                    load, write_jsonl)


def preds(p: Path) -> dict[str, str]:
    return {r["qa_id"]: r.get("prediction", "") for r in load(p)}


def verdicts(tag: str) -> dict[str, bool] | None:
    p = INITIAL_VERDICTS / f"{tag}.jsonl"
    return {r["qa_id"]: bool(r["judge_correct"]) for r in load(p)} if p.exists() else None


def gpt_layers(M: str, R: str) -> tuple[str, list]:
    """(release tag, [(sub-tag or None for the base layer, prediction file), ...])."""
    tag = "g55fc" if (M, R) == ("gpt-5.5", "fc") else f"{M}_{R}"
    st = f"{R}_{M}_test_public.jsonl"
    if R == "fc" and M in ("gpt-4.1-mini", "gpt-5.5"):
        segs = [(None, st + ".layer2"), (tag + "_layer3", f"fc_{M}_test_public_layer3.jsonl"),
                (tag + "_layer4", f"fc_{M}_test_public_layer4.jsonl"),
                (tag + "_layer5", f"fc_{M}_test_public_layer5.jsonl"),
                (f"fc_{M}_test_public_fixed512k", f"fc_{M}_test_public_fixed512k.jsonl")]
    elif R == "rag":
        segs = [(f"jf386_{M}_bm25", f"ragbm25fx386_{M}_test_public.jsonl")]
    else:
        segs = [(None, st + ".layer1"), (tag + "_layer4", f"{R}_{M}_test_public_layer4.jsonl")]
        if (M, R) == ("gpt-5.5", "cb"):
            segs.append((tag + "_retry", "cb_gpt-5.5_test_public_retry.jsonl"))
    return tag, segs


def all_tags() -> dict[str, list]:
    tags: dict[str, list] = {}
    for M in ("gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5"):
        for R in ("cb", "rag", "fc"):
            t, s = gpt_layers(M, R)
            tags[t] = s
        tags[f"jf386_{M}_dense"] = [(None, f"ragdensefx386_{M}_test_public.jsonl")]
    for m in ("minimax-m3", "glm-5.2"):
        for R in ("cb", "rag", "fc"):
            t = f"{R}_{m}-cloud_tp"
            if R == "rag":
                tags[t] = [(f"jf386_{m}_bm25", f"rag_bm25_{m}-cloud_tp386_fixed_tp.jsonl")]
            elif R == "fc":
                tags[t] = [(None, f"run_open/{t}.jsonl"), (f"{t}_fixed", f"run_open/{t}_fixed.jsonl")]
            else:
                tags[t] = [(None, f"run_open/{t}.jsonl")]
        tags[f"jf386_{m}_dense"] = [(None, f"rag_dense_{m}-cloud_tp386_fixed_tp.jsonl")]
    for f in ("fc_gpt-5.5_test_public_512kexp", "fc_gpt-5.5_test_public_fixed512k",
              "fc_gpt-4.1-mini_test_public_512kexp", "fc_gpt-4.1-mini_test_public_fixed512k",
              "run_open/fc_minimax-m3-cloud_tp_512kexp", "run_open/fc_minimax-m3-cloud_tp_fixed",
              "run_open/fc_glm-5.2-cloud_tp_512kexp", "run_open/fc_glm-5.2-cloud_tp_fixed"):
        tags[Path(f).name] = [(None, f + ".jsonl")]
    return tags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="work directory for prompts/index files")
    ap.add_argument("--baselines", default=str(BASELINES), help="internal prediction directory")
    ap.add_argument("--accept-any-format", default="or", choices=["or", "multi"],
                    help="gold text of accept-any items in prompts.jsonl; 'or' reproduces the first "
                         "full batch, whose accept-any verdicts were replaced by prompts_multi.jsonl")
    args = ap.parse_args()
    B, out = Path(args.baselines), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gold = {d["qa_id"]: d for d in load(ROOT / "data/qa_test_public.jsonl")}
    fix = gold_fixes()
    excl = set(fix["exclude_from_evaluation"])
    inscope = {q for q, d in gold.items() if d.get("task_type") != LEGAL and q not in excl}

    reqs, idx = [], []
    for tag, segs in all_tags().items():
        v = verdicts(tag)
        want = sorted(q for q in v if q in inscope)
        src: dict[str, str] = {}
        for sub, f in segs:
            p = preds(B / f)
            if sub is None:
                src.update(p)
            else:
                sv = verdicts(sub)
                src.update({q: p[q] for q in p if q in v and sv is not None and q in sv})
        miss = [q for q in want if q not in src]
        assert not miss, (tag, len(miss))
        for q in want:
            cid = f"r{len(reqs):05d}"
            reqs.append({"custom_id": cid, "prompt": judge_prompt(
                gold[q].get("question", ""), judge_gold(gold[q], fix, args.accept_any_format), src[q])})
            idx.append({"custom_id": cid, "set": "core_or_agg", "tag": tag, "qa_id": q,
                        "gold_fixed": q in fix["replace"] or q in fix["accept_any"]})
    for r in load(B / "panel_items_human80.jsonl"):
        cid = f"r{len(reqs):05d}"
        reqs.append({"custom_id": cid, "prompt": judge_prompt(r["question"], r["gold"], r["prediction"])})
        idx.append({"custom_id": cid, "set": "human80", "tag": "human80", "qa_id": r["qa_id"], "human": r["human"]})
    write_jsonl(out / "prompts.jsonl", reqs)
    write_jsonl(out / "index.jsonl", idx)

    mreqs, midx = [], []
    for r, i in zip(reqs, idx):
        if i["set"] == "core_or_agg" and i["qa_id"] in fix["accept_any"]:
            g = gold[i["qa_id"]]
            cid = "g" + i["custom_id"][1:]
            mreqs.append({"custom_id": cid, "prompt": judge_prompt(
                g.get("question", ""), judge_gold(g, fix, "multi"), _pred_of(r["prompt"]))})
            midx.append({**i, "custom_id": cid, "orig_custom_id": i["custom_id"]})
    write_jsonl(out / "prompts_multi.jsonl", mreqs)
    write_jsonl(out / "index_multi.jsonl", midx)

    print("requests", len(reqs), "by set", dict(collections.Counter(i["set"] for i in idx)),
          "tags", len({i["tag"] for i in idx}))
    print("corrected-gold calls", sum(i.get("gold_fixed", False) for i in idx), "multi-answer calls", len(mreqs))
    return 0


def _pred_of(prompt: str) -> str:
    """Prediction text back out of a judge prompt (inverse of judge_prompt's last field)."""
    p = prompt.split("\n[모델답변] ", 1)[1].rsplit("\n\n판정:", 1)[0]
    return "" if p == "(빈 답)" else p


if __name__ == "__main__":
    raise SystemExit(main())
