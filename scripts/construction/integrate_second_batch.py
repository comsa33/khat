#!/usr/bin/env python3
"""Write the second batch of verified regenerations, one answer correction and the removals (dry run without --apply).

Usage: python3 scripts/construction/integrate_second_batch.py [--apply]
"""
import json, re, os, sys, collections

from khat.paths import LOCAL, RESULTS
TC = f"{LOCAL}/audit/triage_compare"
MASK = "[HELD OUT]"
APPLY = "--apply" in sys.argv
REAL_FILES = [f"{LOCAL}/construction/qa_realistic_candidates.jsonl", f"{LOCAL}/construction/qa_dev.jsonl",
              f"{LOCAL}/construction/qa_test_public.jsonl", f"{LOCAL}/construction/candidates.jsonl"]
PART_B_Q = f"{LOCAL}/construction/qa_test_part_b_questions.jsonl"
PART_B_A = f"{LOCAL}/construction/qa_test_part_b_answers.jsonl"


def make_eval(answer, atype):
    if atype == "number":
        nums = [n.strip(",.") for n in re.findall(r"\d[\d,\.]*", answer)]
        return {"metric": "exact_numbers", "gold_numbers": nums or [answer]}
    a = str(answer).strip()
    return {"metric": "contains_all", "gold_terms": [a if len(a) <= 40 else a[:40].rstrip()]}


G68 = json.load(open(f"{TC}/regen_results_68.json", encoding="utf-8"))
genmap = {it["qa_id"]: it for g in G68 for it in (g.get("gen") or [])}
gpt = {r["qa_id"]: r for r in map(json.loads, open(RESULTS / "triage" / "regen_verify_second_batch.jsonl", encoding="utf-8"))}
regen = {}
flag68 = []
for qid, gi in genmap.items():
    v = gpt.get(qid)
    if v and v.get("verdict") == "OK":
        regen[qid] = {"question": gi["new_question"], "answer": gi["new_answer"],
                      "answer_type": gi["answer_type"], "quote": gi.get("evidence_quote", "")}
    else:
        flag68.append(qid)

answer_fix = {"krhlrb_v05_0910": ("24.10", "number")}

actions = [json.loads(l) for l in open(RESULTS / "triage" / "noncloze_actions.jsonl", encoding="utf-8")]
remove = {r["qa_id"] for r in actions if r["verdict"] == "removed"} | set(flag68)

print(f"68 regen → OK 통합 {len(regen)} / FLAG(제거로) {len(flag68)}")
print(f"answer corrections: {list(answer_fix)} | 제거 총 {len(remove)} (목록 13 + 68-FLAG {len(flag68)})")
if not APPLY:
    print("\n[DRY-RUN] --apply 로 적용")


def process(fp, *, real):
    if not os.path.exists(fp):
        return (0, 0, 0)
    recs = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
    kept = []
    nreg = nfix = nrm = 0
    for r in recs:
        qid = r.get("qa_id")
        if qid in remove:
            nrm += 1
            continue
        if qid in regen:
            f = regen[qid]
            r["question"] = f["question"]
            if real and r.get("answer") != MASK:
                r["answer"] = f["answer"]; r["answer_type"] = f["answer_type"]
                r["evaluation"] = make_eval(f["answer"], f["answer_type"])
                r["evidence"] = [{"source_id": (r.get("source_ids") or ["?"])[0], "locator": (f["quote"] or "")[:200]}]
                r["question_style"] = "real_user"
                r["original_question"] = f["question"]
                r["rewrite_rationale"] = "regenerated from positional-cloze to natural grounded question (v0.8)"
            nreg += 1
        elif qid in answer_fix and real and r.get("answer") != MASK:
            ans, at = answer_fix[qid]
            r["answer"] = ans; r["answer_type"] = at; r["evaluation"] = make_eval(ans, at)
            nfix += 1
        kept.append(r)
    if APPLY and (nreg or nfix or nrm):
        with open(fp, "w", encoding="utf-8") as fh:
            for r in kept:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return (nreg, nfix, nrm)


print("\n파일별 (regen, corrected, remove):")
for fp in REAL_FILES:
    print(f"  {os.path.basename(fp):40} {process(fp, real=True)}")
print(f"  {os.path.basename(PART_B_Q):40} {process(PART_B_Q, real=False)}")
if os.path.exists(PART_B_A):
    print(f"  test part b answers{'':21} {process(PART_B_A, real=True)}")
