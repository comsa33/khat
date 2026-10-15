#!/usr/bin/env python3
"""Deterministic answer metrics (numbers, exact match, substring, token recall) with Wilson intervals, plain and cluster-weighted.

Usage: python3 scripts/score/score_answers.py --pred FILE [--splits dev,test_public] [--ids-file FILE] [--pred-only] [--recall-th 0.7]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from khat.paths import ROOT
GOLD = {sp: ROOT / "data" / f"qa_{sp}.jsonl" for sp in ("dev", "test_public")}
UNANS = ("제공된 자료만으로는", "확정할 수 없음", "확인필요", "답할 수 없", "알 수 없")
TOK_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def load(path: Path) -> list:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()] if path.exists() else []


def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s))


def nums(s: str) -> list:
    return re.findall(r"\d[\d,]*(?:\.\d+)?", str(s))


def tokens(s: str) -> set:
    words = TOK_RE.findall(str(s).lower())
    grams = []
    for w in words:
        if len(w) >= 2 and any("가" <= c <= "힣" for c in w):
            grams += [w[i:i + 2] for i in range(len(w) - 1)]
    return set(words + grams)


def metrics(item: dict, pred: str, recall_th: float) -> dict:
    metric = item.get("evaluation", {}).get("metric", "")
    ans = item.get("answer", "")
    if not pred or not norm(pred):
        return {k: False for k in ("num", "em", "contains", "recall", "soft")}
    if metric == "boolean_and_reason":
        ok = any(m in pred for m in UNANS)
        return {"num": ok, "em": ok, "contains": ok, "recall": ok, "soft": ok}
    npred, nans = norm(pred), norm(ans)
    num = bool(nums(ans)) and all(n.replace(",", "") in npred.replace(",", "") for n in [x.replace(",", "") for x in nums(ans)])
    em = bool(nans) and nans == npred
    contains = bool(nans) and (nans in npred or npred in nans)
    gtok = tokens(ans)
    recall = (len(gtok & tokens(pred)) / len(gtok)) if gtok else 0.0
    rec_ok = recall >= recall_th
    if metric == "exact_numbers":
        soft = num
    else:
        soft = em or contains or rec_ok
    return {"num": num, "em": em, "contains": contains, "recall": rec_ok, "soft": soft}


def wilson(c: float, n: float) -> tuple:
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = c / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0, centre - half), min(1, centre + half))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--splits", default="dev,test_public")
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--pred-only", action="store_true")
    ap.add_argument("--recall-th", type=float, default=0.7, help="token-recall threshold for a soft match")
    ap.add_argument("--by", default="ALL,split,task_type,context_tier,question_style")
    args = ap.parse_args()

    gold = {}
    for sp in (s.strip() for s in args.splits.split(",") if s.strip()):
        for it in load(GOLD[sp]):
            it["_split"] = sp
            gold[it["qa_id"]] = it
    preds = {r["qa_id"]: r.get("prediction", "") for r in load(ROOT / args.pred)}

    restrict = None
    if args.ids_file:
        ip = Path(args.ids_file) if Path(args.ids_file).is_absolute() else ROOT / args.ids_file
        restrict = {json.loads(l).get("qa_id") if l.lstrip().startswith("{") else l.strip()
                    for l in ip.open(encoding="utf-8") if l.strip()}
    if args.pred_only:
        restrict = (restrict & set(preds)) if restrict is not None else set(preds)
    if restrict is not None:
        gold = {q: it for q, it in gold.items() if q in restrict}

    METS = ["soft", "em", "contains", "recall", "num"]
    cuts = [c.strip() for c in args.by.split(",")]
    agg: dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]))
    for qid, it in gold.items():
        m = metrics(it, preds.get(qid, ""), args.recall_th)
        w = it.get("cluster_weight", 1.0)
        keys = ["ALL"]
        for c in cuts:
            if c == "ALL":
                continue
            keys.append(f"{c}:{it.get('_split') if c == 'split' else it.get(c, '?')}")
        for key in keys:
            for met in METS:
                b = agg[key][met]
                ok = int(m[met]); b[0] += ok; b[1] += 1; b[2] += w * ok; b[3] += w

    def fmt(key: str) -> str:
        b = agg[key]["soft"]
        if not b[1]:
            return key
        lo, hi = wilson(b[0], b[1])
        cols = "  ".join(f"{met}={agg[key][met][0]/agg[key][met][1]:5.1%}" for met in METS)
        return (f"{key:30s} soft {b[0]/b[1]:5.1%} [95%CI {lo:.1%}-{hi:.1%}] cw {b[2]/b[3]:5.1%}  "
                f"(n={int(b[1])})  | {cols}")

    print(f"== answer scorer — {args.pred} (recall-th={args.recall_th}) ==")
    print(f"gold scored: {len(gold)}  | metrics: soft(headline) / em / contains(legacy) / recall / num")
    print(fmt("ALL"))
    for c in cuts:
        if c == "ALL":
            continue
        keys = sorted(k for k in agg if k.startswith(f"{c if c != 'split' else 'split'}:"))
        if keys:
            print(f"-- by {c} --")
            for k in keys:
                print("  " + fmt(k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
