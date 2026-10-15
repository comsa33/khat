#!/usr/bin/env python3
"""Write the regenerated questions that passed verification into the construction QA files (dry run without --apply).

Usage: python3 scripts/construction/integrate_regenerated.py [--apply]
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


# ---- build final map: integrate items gpt-5.4 verified OK ----
G = json.load(open(f"{TC}/regen_results.json", encoding="utf-8"))
genmap = {}
for grp in G:
    for it in (grp.get("gen") or []):
        genmap[it["qa_id"]] = it
gpt = {r["qa_id"]: r for r in map(json.loads, open(RESULTS / "triage" / "regen_verify.jsonl", encoding="utf-8"))}

final = {}
flagged = collections.defaultdict(list)
for qid, gi in genmap.items():
    v = gpt.get(qid)
    if not v:
        flagged["no_verify"].append(qid); continue
    if v.get("verdict") == "OK":
        final[qid] = {"question": gi["new_question"], "answer": gi["new_answer"],
                      "answer_type": gi["answer_type"], "quote": gi.get("evidence_quote", "")}
    else:
        flagged[v.get("issue") or "flag"].append(qid)

not_verified = flagged.get("no_verify", [])
print(f"생성 {len(genmap)} | gpt-5.4 OK(통합대상) {len(final)} | "
      f"FLAG(보류) {sum(len(v) for k, v in flagged.items() if k != 'no_verify')}")
print(f"  flagged: { {k: len(v) for k, v in flagged.items()} }")
if not APPLY:
    print("\n[DRY-RUN] 파일 미수정. 적용하려면 --apply")


def patch_record(r, *, real):
    qid = r.get("qa_id")
    f = final.get(qid)
    if not f:
        return False
    r["question"] = f["question"]
    if real and r.get("answer") != MASK:
        r["answer"] = f["answer"]
        r["answer_type"] = f["answer_type"]
        r["evaluation"] = make_eval(f["answer"], f["answer_type"])
        sid = (r.get("source_ids") or ["?"])[0]
        r["evidence"] = [{"source_id": sid, "locator": (f["quote"] or "")[:200]}]
        r["question_style"] = "real_user"
        r["original_question"] = f["question"]
        r["rewrite_rationale"] = "regenerated from positional-cloze to natural grounded question (v0.8)"
    return True


def process(fp, *, real):
    if not os.path.exists(fp):
        return 0
    recs = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
    n = sum(1 for r in recs if patch_record(r, real=real))
    if APPLY and n:
        with open(fp, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return n


print("\n파일별 변경 항목 수:")
for fp in REAL_FILES:
    print(f"  {os.path.basename(fp):42} {process(fp, real=True)}")
print(f"  {os.path.basename(PART_B_Q):42} {process(PART_B_Q, real=False)} (question만)")
if os.path.exists(PART_B_A):
    print(f"  test part b answers{'':23} {process(PART_B_A, real=True)}")

json.dump({"flagged": dict(flagged), "not_verified": not_verified},
          open(f"{TC}/regen_flagged.json", "w"), ensure_ascii=False, indent=1)
print(f"\nflagged/미검증 목록 → regen_flagged.json")
