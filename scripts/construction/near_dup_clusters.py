#!/usr/bin/env python3
"""Near-duplicate clusters: cluster_id, cluster_size and cluster_weight of every item.

Usage: python3 scripts/construction/near_dup_clusters.py
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from khat.paths import ROOT, LOCAL
QA = LOCAL / "construction" / "qa_realistic_candidates.jsonl"

SIDO = ["서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시",
        "세종특별자치시", "경기도", "강원특별자치도", "강원도", "충청북도", "충청남도", "전북특별자치도",
        "전라북도", "전라남도", "경상북도", "경상남도", "제주특별자치도"]


def signature(item: dict) -> str:
    q = item.get("question", "")
    q = re.sub(r"「[^」]*」", "「T」", q)
    q = re.sub(r"\"[^\"]*\"", "\"A\"", q)
    q = re.sub(r"\(p\.\d+\)", "(p.N)", q)
    for s in SIDO:
        q = q.replace(s, "<시도>")
    q = re.sub(r"[가-힣]+(시|군|구)\b", "<시군구>", q)
    q = re.sub(r"20\d{2}\s*년?\s*\d{0,2}\s*월?", "<날짜>", q)
    q = re.sub(r"\d[\d,\.]*", "<수>", q)
    q = re.sub(r"\s+", " ", q).strip()
    return f"{item.get('task_type','')}|{q}"


def main() -> int:
    rows = [json.loads(l) for l in QA.open(encoding="utf-8") if l.strip()]
    groups = defaultdict(list)
    for r in rows:
        groups[signature(r)].append(r["qa_id"])
    sig_order, seen = [], set()
    for r in rows:
        s = signature(r)
        if s not in seen:
            seen.add(s); sig_order.append(s)
    cid_of = {s: f"clu_{i:04d}" for i, s in enumerate(sig_order)}
    size_of = {s: len(groups[s]) for s in groups}

    for r in rows:
        s = signature(r)
        r["cluster_id"] = cid_of[s]
        r["cluster_size"] = size_of[s]
        r["cluster_weight"] = round(1.0 / size_of[s], 6)

    QA.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    sizes = Counter(size_of[s] for s in sig_order)
    n_clusters = len(sig_order)
    singletons = sum(1 for s in sig_order if size_of[s] == 1)
    weighted_total = sum(r["cluster_weight"] for r in rows)
    big = sorted(((size_of[s], s) for s in sig_order), reverse=True)[:5]
    print(f"=== near-dup clusters: {len(rows)} QA -> {n_clusters} clusters ===")
    print(f"  singleton clusters: {singletons} ({singletons/n_clusters:.0%})")
    print(f"  effective (cluster-weighted) size: {weighted_total:.1f}  (vs raw {len(rows)})")
    print(f"  cluster-size histogram (size:count): {dict(sorted(sizes.items())[:8])}{' …' if len(sizes)>8 else ''}")
    print("  largest clusters:")
    for sz, s in big:
        print(f"    size {sz}: {s[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
