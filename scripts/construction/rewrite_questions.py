#!/usr/bin/env python3
"""Rule-based rewrite of the question wording (answer, evidence and ids unchanged).

Usage: python3 scripts/construction/rewrite_questions.py
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from khat.paths import ROOT, LOCAL
SRC = LOCAL / "construction" / "candidates.jsonl"
OUT = LOCAL / "construction" / "qa_realistic_candidates.jsonl"

REAL_USER = {"eligibility_reasoning", "schedule_reasoning", "correction_notice_reasoning",
             "answerability_detection", "region_comparison", "provider_comparison",
             "multi_document_comparison"}
ANALYST = {"table_numeric_reasoning", "cross_source_aggregation", "cross_document_legal_reasoning",
           "format_robustness"}

CLOZE_RE = re.compile(r"^「(?P<title>.+?)」\(p\.(?P<pno>\d+)\)에서 다음 빈칸에 들어갈 값은\? — \"(?P<stem>.*)\"\s*$")


def _hash_pct(qid: str) -> int:
    return int(hashlib.md5(qid.encode("utf-8")).hexdigest(), 16) % 100


def _trim_pre(pre: str) -> str:
    pre = pre.strip()
    if pre.count("(") > pre.count(")"):
        pre = pre.rsplit("(", 1)[0].strip()
    if pre.count("（") > pre.count("）"):
        pre = pre.rsplit("（", 1)[0].strip()
    return pre.strip(" ,.·:;")


def _trim_post(post: str) -> str:
    return post.strip().strip(" ,.·:;")[:18]


def rewrite_retrieval(item: dict) -> tuple[str, str, str]:
    """Return (question, style, rationale)."""
    q = item["question"]
    m = CLOZE_RE.match(q)
    if not m:
        return q, "diagnostic_probe", "cloze form retained (unparsed)"
    title, pno, stem = m.group("title"), m.group("pno"), m.group("stem")
    qid = item["qa_id"]
    keep_cloze = item["task_type"] == "long_distance_retrieval" or _hash_pct(qid) < 14
    if keep_cloze:
        nq = (f"[위치 탐침] 「{title}」(p.{pno}) 본문에서 다음 부분의 빈칸에 들어갈 값은 무엇인가? — \"{stem}\"")
        return nq, "diagnostic_probe", "kept as position/passkey stress probe (long-context locating)"
    parts = stem.split("____")
    pre = _trim_pre(parts[0]) if parts else ""
    post = _trim_post(parts[1]) if len(parts) > 1 else ""
    if len(pre) >= 3:
        nq = f"「{title}」(p.{pno}) 공고문에서 \"{pre}\" 다음에 제시된 값은 무엇인가?"
        rat = "cloze→natural: ask for the value following the preceding phrase (same located value)"
    elif post:
        nq = f"「{title}」(p.{pno}) 공고문에서 \"{post}\" 바로 앞에 제시된 값은 무엇인가?"
        rat = "cloze→natural: ask for the value preceding the following phrase (same located value)"
    else:
        nq = f"[위치 탐침] 「{title}」(p.{pno}) 본문에서 다음 부분의 빈칸에 들어갈 값은 무엇인가? — \"{stem}\""
        return nq, "diagnostic_probe", "kept as probe (no usable anchor for natural phrasing)"
    return nq, "professional_analyst", rat


def rewrite(item: dict) -> tuple[str, str, str]:
    tt = item["task_type"]
    q = item["question"]
    if tt in ("long_context_retrieval", "long_distance_retrieval"):
        return rewrite_retrieval(item)
    if tt == "format_robustness":
        nq = q.replace("(동일 데이터, )", "(동일 데이터)").replace("(동일 데이터,  ", "(동일 데이터, ")
        return nq, "professional_analyst", "format-robustness probe; cleaned empty slice descriptor"
    if tt == "eligibility_reasoning":
        nq = q.replace("기준은 무엇인가?", "기준은 어떻게 되는가?")
        nq = nq.replace("자격요건 표에서", "공고의 신청자격 기준에서").replace(" 표에서", " 신청자격 기준에서")
        return nq, "real_user", "applicant-voice phrasing for eligibility criteria (anchors preserved)"
    if tt == "schedule_reasoning":
        nq = q.replace("일정(일자/시간)은 무엇인가?", "일정(일자/시간)은 언제인가?")
        nq = nq.replace("일정 표에서", "청약 일정에서")
        return nq, "real_user", "applicant-voice phrasing for schedule dates (anchors preserved)"
    if tt == "answerability_detection":
        nq = q.replace("확정할 수 있는가?", "확정할 수 있나요?")
        return nq, "real_user", "applicant 'can I determine this from the given material?' question"
    if tt == "correction_notice_reasoning":
        nq = q.replace("제시하라.", "알려 주세요.").replace("정정(訂正) 공고인가?", "정정(訂正) 공고인가요?")
        return nq, "real_user", "applicant correction-notice question (citizen voice)"
    if tt in ("region_comparison", "provider_comparison", "multi_document_comparison"):
        nq = q.replace("각각 제시하라.", "각각 알려 주세요.").replace("제시하라.", "알려 주세요.")
        nq = nq.replace("같은 시·도에 속하는가?", "같은 시·도에 속하나요?")
        return nq, "real_user", "citizen comparison voice (각각 제시하라→각각 알려 주세요); anchors/answer preserved"
    if tt == "cross_source_aggregation":
        nq = q.replace("에 대하여, HUG 분양이력정보에서", "에 대해 HUG 분양이력 자료를 보면").replace("얼마인가?", "얼마인가요?")
        return nq, "real_user", "citizen/market-voice region+public-data question; predicate/answer preserved"
    if tt in ANALYST:
        return q, "professional_analyst", "already an analytic data/lookup query"
    return q, "professional_analyst", "no change needed"


def main() -> int:
    rows = [json.loads(l) for l in SRC.open(encoding="utf-8") if l.strip()]
    out, styles = [], Counter()
    seen_q = set()
    n_collision = 0
    for it in rows:
        nq, style, rat = rewrite(it)
        if nq in seen_q and it["task_type"] in ("long_context_retrieval", "long_distance_retrieval"):
            nq = f"[위치 탐침] {it['question']}"
            style, rat = "diagnostic_probe", "kept as cloze to avoid duplicate after naturalization"
            n_collision += 1
        seen_q.add(nq)
        new = dict(it)
        new["original_question"] = it["question"]
        new["question"] = nq
        new["question_style"] = style
        new["rewrite_rationale"] = rat
        out.append(new)
        styles[style] += 1
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    cloze = sum(1 for r in out if "빈칸" in r["question"] or "____" in r["question"])
    print(f"=== realism rewrite: {len(out)} QA -> {OUT.name} ===")
    for k, v in styles.most_common():
        print(f"   {k:22s} {v}  ({100*v/len(out):.1f}%)")
    print(f"   cloze-phrased: {cloze} ({100*cloze/len(out):.1f}%)")
    print(f"   real_user+professional_analyst: {100*(styles['real_user']+styles['professional_analyst'])/len(out):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
