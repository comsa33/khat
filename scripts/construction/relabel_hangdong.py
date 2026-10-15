#!/usr/bin/env python3
"""Set the district of the Hangdong (sh-196920) announcement to 구로구 in answers and region_sigungu.

Usage: python3 scripts/construction/relabel_hangdong.py
"""
import json, os

from khat.paths import LOCAL
FILES = [f"{LOCAL}/construction/qa_realistic_candidates.jsonl",
         f"{LOCAL}/construction/qa_dev.jsonl",
         f"{LOCAL}/construction/qa_test_public.jsonl",
         f"{LOCAL}/construction/qa_test_part_b_questions.jsonl"]
PART_B_ANS = f"{LOCAL}/construction/qa_test_part_b_answers.jsonl"
DISTRICT_FAMILIES = {"multi_document_comparison", "region_comparison"}
MASK = "[HELD OUT]"


def is_hd(r):
    return any("196920" in str(a) for a in (r.get("announcement_ids") or []))

def hd_index(r):
    for i, a in enumerate(r.get("announcement_ids") or []):
        if "196920" in str(a):
            return i
    return -1

canon = [json.loads(l) for l in open(FILES[0], encoding="utf-8") if l.strip()]
A_fix = {}
for r in canon:
    if r["task_type"] not in DISTRICT_FAMILIES or not is_hd(r):
        continue
    i = hd_index(r)
    parts = [p.strip() for p in str(r.get("answer", "")).split("/")]
    if i < 0 or i >= len(parts) or parts[i] != "서울특별시":
        continue
    parts[i] = "구로구"
    new_ans = " / ".join(parts)
    gt = (r.get("evaluation") or {}).get("gold_terms") or []
    new_gt = ["구로구" if x == "서울특별시" else x for x in gt]
    A_fix[r["qa_id"]] = (new_ans, new_gt)

print(f"A 답수정 대상 {len(A_fix)}개: {sorted(A_fix)}")

def patch_record(r, *, allow_answer):
    changed = []
    qid = r.get("qa_id")
    if qid in A_fix and allow_answer and str(r.get("answer", "")) != MASK:
        na, ngt = A_fix[qid]
        if r.get("answer") != na:
            r["answer"] = na; changed.append("answer")
        ev = r.get("evaluation")
        if isinstance(ev, dict) and ev.get("gold_terms") is not None and ev.get("gold_terms") != ngt:
            ev["gold_terms"] = ngt; changed.append("gold_terms")
    if is_hd(r) and r.get("region_sigungu") == "서울특별시":
        r["region_sigungu"] = "구로구"; changed.append("region_sigungu")
    return changed

tot = {"answer": 0, "gold_terms": 0, "region_sigungu": 0}
for fp in FILES:
    if not os.path.exists(fp):
        continue
    recs = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
    n = 0
    for r in recs:
        for c in patch_record(r, allow_answer=True):
            tot[c] += 1; n += 1
    with open(fp, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {os.path.basename(fp):40} {n} 필드 변경")

if os.path.exists(PART_B_ANS):
    recs = [json.loads(l) for l in open(PART_B_ANS, encoding="utf-8") if l.strip()]
    n = 0
    for r in recs:
        qid = r.get("qa_id")
        if qid in A_fix:
            na, ngt = A_fix[qid]
            if r.get("answer") not in (None, na):
                r["answer"] = na; n += 1
            ev = r.get("evaluation")
            if isinstance(ev, dict) and ev.get("gold_terms") is not None:
                ev["gold_terms"] = ngt; n += 1
    with open(PART_B_ANS, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  test part b answers {n} 변경")
else:
    print("  (test part b answers 파일 없음 — 해당 split에 A항목 없으면 무관)")

print("\n총 변경:", tot)
