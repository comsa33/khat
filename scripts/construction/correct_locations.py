#!/usr/bin/env python3
"""Location corrections from the source sweep: one answer and three region_sigungu values.

Usage: python3 scripts/construction/correct_locations.py
"""
import json, os
from khat.paths import LOCAL
PUB = [f"{LOCAL}/construction/qa_realistic_candidates.jsonl", f"{LOCAL}/construction/qa_dev.jsonl",
       f"{LOCAL}/construction/qa_test_public.jsonl", f"{LOCAL}/construction/qa_test_part_b_questions.jsonl"]
V5 = f"{LOCAL}/construction/candidates.jsonl"
MASK = "[HELD OUT]"
ANS_FIX = {"krhlrb_v05_1734": ("제주시 / 서귀포시", ["제주시", "서귀포시"])}

def aid_has(r, s): return any(s in str(a) for a in (r.get("announcement_ids") or []))

def sigungu_fix(r):
    cur = r.get("region_sigungu")
    if aid_has(r, "sh-196920") and cur == "구로구": return "서울특별시 구로구"
    if aid_has(r, "jpdc-jpdc-2025-04-29-ildo") and cur == "제주특별자치도": return "제주시"
    if aid_has(r, "발산8단지") and cur == "서울특별시": return "서울특별시 강서구"
    return None

def patch(recs, *, with_answer):
    c = {"answer": 0, "gold_terms": 0, "region_sigungu": 0}
    for r in recs:
        qid = r.get("qa_id")
        if with_answer and qid in ANS_FIX and str(r.get("answer", "")) != MASK:
            na, ngt = ANS_FIX[qid]
            if r.get("answer") != na: r["answer"] = na; c["answer"] += 1
            ev = r.get("evaluation")
            if isinstance(ev, dict) and ev.get("gold_terms") is not None and ev["gold_terms"] != ngt:
                ev["gold_terms"] = ngt; c["gold_terms"] += 1
        ns = sigungu_fix(r)
        if ns and r.get("region_sigungu") != ns:
            r["region_sigungu"] = ns; c["region_sigungu"] += 1
    return c

tot = {"answer": 0, "gold_terms": 0, "region_sigungu": 0}
for fp in PUB:
    if not os.path.exists(fp): continue
    recs = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
    c = patch(recs, with_answer=True)
    with open(fp, "w", encoding="utf-8") as f:
        for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for k in tot: tot[k] += c[k]
    print(f"  {os.path.basename(fp):40} {c}")

recs = [json.loads(l) for l in open(V5, encoding="utf-8") if l.strip()]
n = 0
for r in recs:
    qid = r.get("qa_id")
    if qid in ANS_FIX:
        na, ngt = ANS_FIX[qid]
        if r.get("answer") != na: r["answer"] = na; n += 1
        ev = r.get("evaluation")
        if isinstance(ev, dict) and ev.get("gold_terms") is not None and ev["gold_terms"] != ngt:
            ev["gold_terms"] = ngt; n += 1
with open(V5, "w", encoding="utf-8") as f:
    for r in recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"  qa_v0.5_candidates.jsonl (answer invariant)  {n} fields")
print("\n총:", tot)
