#!/usr/bin/env python3
"""Deterministic items over the LH announcements, the HUG and MOLIT rows and the statutes (gold from the sources).

Usage: python3 scripts/construction/build_lh_items.py
"""
from __future__ import annotations

import json
from collections import Counter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import table_rows as C
import lh_announcements as Q

OUT = C.LOCAL / "construction" / "lh_items.jsonl"
FMT_DIR = C.PROC / "format_variants"

MONTH_KO = lambda ym: f"{ym[:4]}년 {int(ym[4:6])}월"
items: list[dict] = []


def emit(it: dict, ok: bool, band: str | None = None, why: str = "") -> None:
    it["_ok"] = ok
    it["_why"] = why
    it["_desired_band"] = band
    items.append(it)


def rid_sample(ids, k=20):
    return ids[:k]


# ----------------------------------------------------------------- table_numeric (MOLIT + HUG)
def gen_table_numeric():
    idxm = json.loads((Q.AUDIT / "index_molit.json").read_text(encoding="utf-8"))
    slices = sorted(idxm["district_month_slices"], key=lambda x: (-x["count"], x["district"], x["deal_ymd"]))
    ops_cycle = ["count", "avg", "max", "argmax", "median", "min", "band84", "floor20"]
    n_emit = 0
    for i, s in enumerate(slices):
        dist, ym = s["district"], s["deal_ymd"]
        base = {"source": C.MOLIT, "filter": {"_lawd_name": dist, "_deal_ymd": ym}}
        chosen = [ops_cycle[(i + j) % len(ops_cycle)] for j in range(3)]
        for op in chosen:
            it = _molit_item(dist, ym, base, op)
            if it:
                emit(it[0], it[1])
                n_emit += 1
        if n_emit >= 210:
            break
    idxh = json.loads((Q.AUDIT / "index_hug.json").read_text(encoding="utf-8"))
    hsl = sorted(idxh["region_year_slices"], key=lambda x: -x["count"])
    hcount = 0
    for s in hsl:
        if s["count"] < 3:
            continue
        region, year = s["region"], str(s["year"])
        base = {"source": C.HUG, "filter": {"_query_area_name": region, "_query_year": year}}
        for op, field, unit, label in [
            ("count", None, "건", "분양 사업장(분양보증 발급) 건수"),
            ("sum", "TOT_HOCO", "세대", "분양 사업장 총세대수(TOT_HOCO) 합계"),
            ("avg", "TOT_HOCO", "세대", "분양 사업장 평균 총세대수(TOT_HOCO)")]:
            pred = {**base, "op": op}
            if field:
                pred["field"] = field
            v, ids = C.recompute(pred)
            if v is None or (op == "count" and v == 0):
                continue
            it = {
                "split": "dev", "task_type": "table_numeric_reasoning",
                "question": f"HUG 분양이력정보에서 {year}년 {region}의 {label}는 얼마인가?",
                "answer": f"{v:,}{unit}", "answer_type": "number",
                "source_ids": [C.HUG], "required_capabilities": ["table_filtering", "numeric_aggregation"],
                "evidence": [{"source_id": C.HUG, "locator": f"HUG 분양이력정보 / predicate: {C.predicate_human(pred)}"}],
                "evaluation": {"metric": "exact_numbers", "gold_numbers": [str(v)]},
                "row_ids": rid_sample(ids), "gold_predicate": pred,
                "copyright_note": "공공데이터 실데이터 행에서 결정론 집계. 공개물은 predicate+row_id+단답만.",
            }
            emit(it, True)
            hcount += 1
        if hcount >= 48:
            break


def _molit_item(dist, ym, base, op):
    loc = lambda p: f"MOLIT 아파트 매매 실거래가 상세 / predicate: {C.predicate_human(p)}"
    d = dist
    if op == "count":
        p = {**base, "op": "count"}
        v, ids = C.recompute(p)
        q = f"MOLIT 아파트 매매 실거래가 상세에서 {d} {MONTH_KO(ym)}의 아파트 매매 거래 건수는 몇 건인가?"
        a = f"{v:,}건"
        return (_mk(q, a, "number", ["table_filtering", "counting"], p, {"gold_numbers": [str(v)]}, ids, loc(p)), v is not None)
    if op == "avg":
        p = {**base, "op": "avg", "field": "dealAmount_manwon"}
        v, ids = C.recompute(p)
        if v is None:
            return None
        q = f"MOLIT 실거래 상세에서 {d} {MONTH_KO(ym)} 아파트 매매 거래금액의 평균은 약 몇 만원인가? (정수 만원)"
        return (_mk(q, f"{v:,}만원", "number", ["table_filtering", "numeric_aggregation"], p, {"gold_numbers": [str(v)]}, ids, loc(p)), True)
    if op == "median":
        p = {**base, "op": "median", "field": "dealAmount_manwon"}
        v, ids = C.recompute(p)
        if v is None:
            return None
        q = f"MOLIT 실거래 상세에서 {d} {MONTH_KO(ym)} 아파트 매매 거래금액의 중앙값은 몇 만원인가?"
        return (_mk(q, f"{v:,}만원", "number", ["table_filtering", "numeric_aggregation"], p, {"gold_numbers": [str(v)]}, ids, loc(p)), True)
    if op == "max":
        p = {**base, "op": "max", "field": "dealAmount_manwon"}
        v, ids = C.recompute(p)
        if v is None:
            return None
        q = f"MOLIT 실거래 상세에서 {d} {MONTH_KO(ym)} 아파트 매매 최고 거래금액은 몇 만원인가?"
        return (_mk(q, f"{v:,}만원", "number", ["table_filtering", "numeric_aggregation"], p, {"gold_numbers": [str(v)]}, ids, loc(p)), True)
    if op == "min":
        p = {**base, "op": "min", "field": "dealAmount_manwon"}
        v, ids = C.recompute(p)
        if v is None:
            return None
        q = f"MOLIT 실거래 상세에서 {d} {MONTH_KO(ym)} 아파트 매매 최저 거래금액은 몇 만원인가?"
        return (_mk(q, f"{v:,}만원", "number", ["table_filtering", "numeric_aggregation"], p, {"gold_numbers": [str(v)]}, ids, loc(p)), True)
    if op == "argmax":
        p = {**base, "op": "argmax", "field": "dealAmount_manwon", "return_field": "aptNm"}
        v, ids = C.recompute(p)
        if not v:
            return None
        q = f"MOLIT 실거래 상세에서 {d} {MONTH_KO(ym)} 최고 거래금액을 기록한 아파트의 이름(aptNm)은?"
        return (_mk(q, f"{v}", "text", ["table_filtering", "argmax"], p, {"gold_terms": [v]}, ids, loc(p), "exact_match"), True)
    if op == "band84":
        p = {**base, "op": "count", "filter": {**base["filter"], "excluUseAr_m2": {"min": 84, "max": 85}}}
        v, ids = C.recompute(p)
        if not v or v <= 0:
            return None
        q = f"MOLIT 실거래 상세에서 {d} {MONTH_KO(ym)} 전용면적 84㎡대(84㎡ 이상 85㎡ 미만) 매매 거래 건수는?"
        return (_mk(q, f"{v:,}건", "number", ["table_filtering", "counting"], p, {"gold_numbers": [str(v)]}, ids, loc(p)), True)
    if op == "floor20":
        p = {**base, "op": "count", "filter": {**base["filter"], "floor": {"min": 20, "max": 1000}}}
        v, ids = C.recompute(p)
        if v is None or v <= 0:
            return None
        q = f"MOLIT 실거래 상세에서 {d} {MONTH_KO(ym)} 거래 중 20층 이상에서 거래된 건수는?"
        return (_mk(q, f"{v:,}건", "number", ["table_filtering", "counting"], p, {"gold_numbers": [str(v)]}, ids, loc(p)), True)
    return None


def _mk(q, a, atype, caps, pred, gold, ids, loc, metric="exact_numbers"):
    return {
        "split": "dev", "task_type": "table_numeric_reasoning",
        "question": q, "answer": a, "answer_type": atype,
        "source_ids": [pred["source"]], "required_capabilities": caps,
        "evidence": [{"source_id": pred["source"], "locator": loc}],
        "evaluation": {"metric": metric, **gold},
        "row_ids": rid_sample(ids), "gold_predicate": pred,
        "copyright_note": "공공데이터 실데이터 행에서 결정론 집계. 공개물은 predicate+row_id+단답만.",
    }


# ----------------------------------------------------------------- format_robustness (15 slices x 4)
def _fmt_emit(base, dist, ym, pred, rows, slice_desc):
    ids = [r["_row_id"] for r in rows]
    norm = [{"row_id": r["_row_id"], "aptNm": r["aptNm"], "excluUseAr": r["excluUseAr"],
             "floor": r["floor"], "dealAmount_manwon": r["dealAmount"].replace(",", "")} for r in rows]
    (FMT_DIR / f"{base}.txt").write_text(
        "\n".join(f"{r['row_id']} {r['aptNm']} {r['excluUseAr']}㎡ {r['floor']}층 {r['dealAmount_manwon']}만원" for r in norm),
        encoding="utf-8")
    (FMT_DIR / f"{base}.md").write_text(
        "| row_id | aptNm | excluUseAr | floor | dealAmount_manwon |\n|---|---|---|---|---|\n" +
        "\n".join(f"| {r['row_id']} | {r['aptNm']} | {r['excluUseAr']} | {r['floor']} | {r['dealAmount_manwon']} |" for r in norm),
        encoding="utf-8")
    (FMT_DIR / f"{base}.csv").write_text(
        "row_id,aptNm,excluUseAr,floor,dealAmount_manwon\n" +
        "\n".join(f"{r['row_id']},{r['aptNm']},{r['excluUseAr']},{r['floor']},{r['dealAmount_manwon']}" for r in norm),
        encoding="utf-8")
    (FMT_DIR / f"{base}.json").write_text(json.dumps(norm, ensure_ascii=False), encoding="utf-8")
    cnt, _ = C.recompute(pred)
    for fmt, ext in [("plain_text", "txt"), ("markdown", "md"), ("csv", "csv"), ("json", "json")]:
        it = {
            "split": "dev", "task_type": "format_robustness",
            "question": f"다음 MOLIT 실거래 표(동일 데이터, {slice_desc})를 {fmt} 형식으로 제공했을 때, "
                        f"{dist} {MONTH_KO(ym)}{slice_desc} 아파트 매매 거래 '행(레코드) 수'는 몇 개인가?",
            "answer": f"{cnt}", "answer_type": "number",
            "source_ids": [C.MOLIT], "required_capabilities": ["format_robustness", "counting"],
            "evidence": [{"source_id": C.MOLIT,
                          "locator": f"format_variants_v04/{base}.{ext} ({fmt}) / predicate: {C.predicate_human(pred)}"}],
            "evaluation": {"metric": "exact_numbers", "gold_numbers": [str(cnt)]},
            "row_ids": rid_sample(ids), "gold_predicate": pred,
            "copyright_note": "동일 슬라이스의 4개 직렬화 형식에 대한 동일 질의(형식 강건성). 행은 내부 format_variants_v04에만 저장.",
        }
        emit(it, cnt is not None and len(rows) == cnt)


def gen_format_robustness(target_slices=15):
    FMT_DIR.mkdir(parents=True, exist_ok=True)
    idxm = json.loads((Q.AUDIT / "index_molit.json").read_text(encoding="utf-8"))
    all_slices = sorted(idxm["district_month_slices"], key=lambda x: (x["district"], x["deal_ymd"]))
    used = 0
    used_bases = set()
    for s in all_slices:
        if used >= target_slices:
            break
        dist, ym = s["district"], s["deal_ymd"]
        pred = {"source": C.MOLIT, "filter": {"_lawd_name": dist, "_deal_ymd": ym}, "op": "count"}
        rows = C.select(pred)
        if not (5 <= len(rows) <= 60):
            continue
        base = f"{dist.replace(' ', '')}_{ym}"
        if base in used_bases:
            continue
        used_bases.add(base)
        _fmt_emit(base, dist, ym, pred, rows, "")
        used += 1
    for s in all_slices:
        if used >= target_slices:
            break
        dist, ym = s["district"], s["deal_ymd"]
        pred = {"source": C.MOLIT, "filter": {"_lawd_name": dist, "_deal_ymd": ym,
                                              "excluUseAr_m2": {"min": 84, "max": 85}}, "op": "count"}
        rows = C.select(pred)
        if not (5 <= len(rows) <= 60):
            continue
        base = f"{dist.replace(' ', '')}_{ym}_84"
        if base in used_bases:
            continue
        used_bases.add(base)
        _fmt_emit(base, dist, ym, pred, rows, " 전용면적 84㎡대(84~85㎡)")
        used += 1


# ----------------------------------------------------------------- answerability (empty slices + cross-announcement)
def gen_answerability():
    molit_absent = [("부산광역시 해운대구", "202503"), ("세종특별자치시", "202504"),
                    ("경기도 성남시 분당구", "202505"), ("인천광역시 연수구", "202506"),
                    ("광주광역시 서구", "202507"), ("대구광역시 수성구", "202508"),
                    ("울산광역시 남구", "202503"), ("경기도 수원시 영통구", "202509"),
                    ("강원특별자치도 춘천시", "202506"), ("전북특별자치도 전주시 완산구", "202504"),
                    ("제주특별자치도 제주시", "202507"), ("충청남도 천안시 서북구", "202505"),
                    ("경상남도 창원시 성산구", "202508"), ("서울특별시 마포구", "202510"),
                    ("서울특별시 강서구", "202502"), ("경기도 고양시 일산동구", "202507"),
                    ("서울특별시 영등포구", "202501"), ("서울특별시 노원구", "202503"),
                    ("부산광역시 수영구", "202504"), ("대구광역시 달서구", "202505"),
                    ("인천광역시 부평구", "202506"), ("경기도 용인시 수지구", "202508"),
                    ("대전광역시 서구", "202509"), ("충청북도 청주시 흥덕구", "202511"),
                    ("경기도 남양주시", "202507"), ("경기도 시흥시", "202506")]
    for dist, ym in molit_absent:
        pred = {"source": C.MOLIT, "filter": {"_lawd_name": dist, "_deal_ymd": ym}, "op": "count"}
        v, _ = C.recompute(pred)
        it = {
            "split": "dev", "task_type": "answerability_detection",
            "question": f"제공된 MOLIT 아파트 매매 실거래가 상세(현재 표본)만으로 {dist} {MONTH_KO(ym)}의 평균 거래금액을 확정할 수 있는가?",
            "answer": f"확정할 수 없음(unanswerable). 현재 표본에 {dist} {MONTH_KO(ym)} 거래 데이터가 없음(0건).",
            "answer_type": "boolean_with_reason",
            "source_ids": [C.MOLIT], "required_capabilities": ["answerability_detection", "coverage_check"],
            "evidence": [{"source_id": C.MOLIT, "locator": f"표본 부재: {C.predicate_human(pred)} = 0"}],
            "evaluation": {"metric": "boolean_and_reason", "gold_numbers": ["0"], "gold_terms": ["확정할 수 없"]},
            "gold_predicate": pred,
            "copyright_note": "negative control. 표본 부재(0건)를 결정론으로 검증.",
        }
        emit(it, v == 0)
    hug_absent = [("세종특별자치시", "2024"), ("서울특별시", "2022"), ("경기도", "2021"),
                  ("부산광역시", "2026"), ("대전광역시", "2022"), ("인천광역시", "2020"),
                  ("제주특별자치도", "2021"), ("광주광역시", "2026")]
    for region, year in hug_absent:
        pred = {"source": C.HUG, "filter": {"_query_area_name": region, "_query_year": year}, "op": "count"}
        v, _ = C.recompute(pred)
        it = {
            "split": "dev", "task_type": "answerability_detection",
            "question": f"제공된 HUG 분양이력정보(현재 표본)만으로 {year}년 {region}의 분양 사업장 총세대수 합을 확정할 수 있는가?",
            "answer": f"확정할 수 없음(unanswerable). 현재 표본에 {year}년 {region} 분양이력 데이터가 없음(0건).",
            "answer_type": "boolean_with_reason",
            "source_ids": [C.HUG], "required_capabilities": ["answerability_detection", "coverage_check"],
            "evidence": [{"source_id": C.HUG, "locator": f"표본 부재: {C.predicate_human(pred)} = 0"}],
            "evaluation": {"metric": "boolean_and_reason", "gold_numbers": ["0"], "gold_terms": ["확정할 수 없"]},
            "gold_predicate": pred,
            "copyright_note": "negative control. 표본 부재(0건)를 결정론으로 검증.",
        }
        emit(it, v == 0)
    anns = list(Q.announcement_ids())
    meta = Q.ann_meta()
    pairs = []
    for i, a in enumerate(anns):
        for off in (3, 5):
            b = anns[(i + off) % len(anns)]
            if a != b:
                pairs.append((a, b))
    for a, b in pairs:
        b_tok = Q.ann_sigungu(b)
        if not b_tok:
            continue
        b_specific = b_tok.split()[-1]
        a_pages = [p["page_id"] for p in Q.ann_pages(a)]
        if Q.grounded_in_pages(b_specific, a_pages):
            continue
        a_title = meta.get(a, {}).get("title_from_audit") or a
        b_title = meta.get(b, {}).get("title_from_audit") or b
        it = {
            "split": "dev", "task_type": "answerability_detection",
            "question": f"제공된 입주자모집공고가 「{a_title}」 1건뿐일 때, 「{b_title}」의 공급위치가 속한 시·군·구를 확정할 수 있는가?",
            "answer": f"확정할 수 없음(unanswerable). 제공된 공고에는 해당 공고({b_title})의 공급위치 정보가 없음.",
            "answer_type": "boolean_with_reason",
            "source_ids": [Q.LH_V04], "required_capabilities": ["answerability_detection", "coverage_check"],
            "evidence": [{"source_id": Q.LH_V04, "locator": f"{Q.page_id_for(a, 1)} 제공 시 {b} 공급위치 부재"}],
            "evaluation": {"metric": "boolean_and_reason", "gold_terms": ["확정할 수 없"]},
            "page_ids": [Q.page_id_for(a, 1)],
            "announcement_ids": [a],
            "copyright_note": "negative control(문서 간 부재). 타 공고의 지역 토큰이 제공 공고에 부재함을 결정론으로 검증.",
        }
        emit(it, not Q.grounded_in_pages(b_specific, a_pages))


# ----------------------------------------------------------------- cross_source (announcement -> MOLIT/HUG)
def gen_cross_source():
    meta = Q.ann_meta()
    for a in Q.announcement_ids():
        title = meta.get(a, {}).get("title_from_audit") or a
        pid = Q.page_id_for(a, 1)
        sido = Q.ann_sido(a)
        sigungu = Q.ann_sigungu(a)
        short = Q.ann_short_region(a)
        hug_region = Q.SIDO_TO_HUG.get(sido)
        if hug_region:
            for year in ["2023", "2024", "2025"]:
                base = {"source": C.HUG, "filter": {"_query_area_name": hug_region, "_query_year": year}}
                for op, field, unit, lab in [("count", None, "건", "분양 사업장 건수"),
                                             ("sum", "TOT_HOCO", "세대", "분양 사업장 총세대수(TOT_HOCO) 합"),
                                             ("avg", "TOT_HOCO", "세대", "분양 사업장 평균 총세대수(TOT_HOCO)")]:
                    pred = {**base, "op": op}
                    if field:
                        pred["field"] = field
                    v, ids = C.recompute(pred)
                    if v is None or (op == "count" and v == 0):
                        continue
                    _emit_cs(
                        f"「{title}」(p.1)의 공급위치가 속한 시·도에 대하여, HUG 분양이력정보에서 {year}년 {lab}은 얼마인가?",
                        f"{v:,}{unit}", [Q.LH_V04, C.HUG], pred, [short, hug_region], pid, a, ids)
        if sigungu and sigungu in {r["_lawd_name"] for r in C.molit_rows()}:
            for ym in ["202503", "202506", "202510"]:
                base = {"source": C.MOLIT, "filter": {"_lawd_name": sigungu, "_deal_ymd": ym}}
                for op, field, unit, lab in [("count", None, "건", "아파트 매매 거래 건수"),
                                             ("avg", "dealAmount_manwon", "만원", "평균 거래금액"),
                                             ("max", "dealAmount_manwon", "만원", "최고 거래금액")]:
                    pred = {**base, "op": op}
                    if field:
                        pred["field"] = field
                    v, ids = C.recompute(pred)
                    if v is None or (op == "count" and v == 0):
                        continue
                    _emit_cs(
                        f"「{title}」(p.1)의 공급위치가 속한 시·군·구의 MOLIT 실거래에서 {MONTH_KO(ym)} {lab}은 얼마인가?",
                        f"{v:,}{unit}", [Q.LH_V04, C.MOLIT], pred, [short], pid, a, ids)


def _emit_cs(question, answer, srcs, pred, gold_terms, pid, ann_id, ids):
    it = {
        "split": "dev", "task_type": "cross_source_aggregation",
        "question": question, "answer": answer, "answer_type": "number",
        "source_ids": srcs,
        "required_capabilities": ["cross_source_aggregation", "multi_hop_reasoning", "table_filtering"],
        "evidence": [{"source_id": Q.LH_V04, "locator": f"{pid} 공급위치(지역 식별)"},
                     {"source_id": pred["source"], "locator": f"predicate: {C.predicate_human(pred)}"}],
        "evaluation": {"metric": "exact_numbers",
                       "gold_numbers": [answer.replace(",", "").replace("만원", "").replace("세대", "").replace("건", "")],
                       "gold_terms": [t for t in gold_terms if t]},
        "page_ids": [pid], "announcement_ids": [ann_id], "row_ids": rid_sample(ids), "gold_predicate": pred,
        "copyright_note": "공고 사실(공급위치) + 공공데이터 집계 결합. 공개물은 locator+predicate+단답만.",
    }
    ok = Q.grounded_in_pages(gold_terms[0], [pid]) if gold_terms else True
    emit(it, ok, band="middle")


# ----------------------------------------------------------------- LH cloze retrieval (early=long_context, late=long_distance)
def _cloze_item(ann_id, fact, task_type, band):
    val = fact["value_text"]
    page = Q.page_by_id().get(fact["page_id"])
    if not page or len(val) < 2:
        return None
    snip = fact["local_snippet"]
    i = snip.find(val)
    if i < 0:
        return None
    pre = snip[max(0, i - 20):i]
    if " " in pre:
        pre = pre[pre.index(" ") + 1:]
    post = snip[i + len(val):i + len(val) + 8]
    if " " in post:
        post = post[:post.rindex(" ")]
    stem = f"{pre}____{post}".strip()
    if not (4 <= len(stem) <= 30):
        return None
    if not Q.grounded_in_pages(val, [fact["page_id"]]):
        return None
    meta = Q.ann_meta().get(ann_id, {})
    title = meta.get("title_from_audit") or ann_id
    it = {
        "split": "dev", "task_type": task_type,
        "question": f"「{title}」(p.{fact['page_no']})에서 다음 빈칸에 들어갈 값은? — \"{stem}\"",
        "answer": val, "answer_type": "span",
        "source_ids": [Q.LH_V04],
        "required_capabilities": [task_type, "announcement_reading"],
        "evidence": [{"source_id": Q.LH_V04, "locator": f"{fact['locator']} (p.{fact['page_no']})"}],
        "evaluation": {"metric": "contains_all", "gold_terms": [val]},
        "page_ids": [fact["page_id"]], "announcement_ids": [ann_id],
        "copyright_note": "공고 단답 + 짧은 cloze 스템(공공기관 입주자모집공고). 원문 전체는 내부 전용.",
    }
    return it


def gen_lh_retrieval(early_per=11, late_per=7):
    for a in Q.announcement_ids():
        pages = Q.ann_pages(a)
        max_no = max(p["page_no"] for p in pages)
        uniq = Q.ann_unique_facts(a)
        seen_vals = set()
        n = 0
        for f in uniq:
            if n >= early_per:
                break
            if f["page_no"] > 6 or f["value_text"] in seen_vals:
                continue
            it = _cloze_item(a, f, "long_context_retrieval", "early")
            if it:
                seen_vals.add(f["value_text"])
                emit(it, True, band="early")
                n += 1
        late_start = max(8, int(max_no * 0.58))
        n = 0
        for f in uniq:
            if n >= late_per:
                break
            if f["page_no"] < late_start or f["value_text"] in seen_vals:
                continue
            it = _cloze_item(a, f, "long_distance_retrieval", "late")
            if it:
                seen_vals.add(f["value_text"])
                emit(it, True, band="late")
                n += 1


# ----------------------------------------------------------------- multi-document comparison (deterministic region cross-ref)
def gen_multi_compare_det():
    anns = list(Q.announcement_ids())
    meta = Q.ann_meta()
    pairs = []
    for i in range(len(anns)):
        for off in (1, 3):
            j = (i + off) % len(anns)
            if i < j:
                pairs.append((anns[i], anns[j]))
    seen = set()
    for a, b in pairs:
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key)
        ta = meta.get(a, {}).get("title_from_audit") or a
        tb = meta.get(b, {}).get("title_from_audit") or b
        pa, pb = Q.page_id_for(a, 1), Q.page_id_for(b, 1)
        sido_a, sido_b = Q.ann_sido(a), Q.ann_sido(b)
        sgg_a, sgg_b = Q.ann_short_region(a), Q.ann_short_region(b)
        if not (sido_a and sido_b and sgg_a and sgg_b):
            continue
        same = "동일하다" if sido_a == sido_b else "동일하지 않다"
        it1 = {
            "split": "dev", "task_type": "multi_document_comparison",
            "question": f"「{ta}」와 「{tb}」 두 입주자모집공고의 공급위치가 같은 시·도에 속하는가? 각 공고의 시·도를 함께 제시하라.",
            "answer": f"{same}. ({sido_a} / {sido_b})", "answer_type": "string",
            "source_ids": [Q.LH_V04],
            "required_capabilities": ["multi_document", "comparison", "long_context_retrieval"],
            "evidence": [{"source_id": Q.LH_V04, "locator": f"{pa} 공급위치"},
                         {"source_id": Q.LH_V04, "locator": f"{pb} 공급위치"}],
            "evaluation": {"metric": "contains_all", "gold_terms": [sido_a, sido_b]},
            "page_ids": [pa, pb], "announcement_ids": [a, b],
            "copyright_note": "두 공고의 공급위치(지역) 대조. 공개물은 locator+단답만.",
        }
        ok1 = Q.grounded_in_pages(sido_a, [pa]) and Q.grounded_in_pages(sido_b, [pb])
        emit(it1, ok1, band="multi")
        it2 = {
            "split": "dev", "task_type": "multi_document_comparison",
            "question": f"「{ta}」와 「{tb}」의 공급위치 시·군·구(또는 시)를 각각 제시하라.",
            "answer": f"{sgg_a} / {sgg_b}", "answer_type": "string",
            "source_ids": [Q.LH_V04],
            "required_capabilities": ["multi_document", "comparison", "long_context_retrieval"],
            "evidence": [{"source_id": Q.LH_V04, "locator": f"{pa} 공급위치"},
                         {"source_id": Q.LH_V04, "locator": f"{pb} 공급위치"}],
            "evaluation": {"metric": "contains_all", "gold_terms": [sgg_a, sgg_b]},
            "page_ids": [pa, pb], "announcement_ids": [a, b],
            "copyright_note": "두 공고의 공급위치(시·군·구) 대조. 공개물은 locator+단답만.",
        }
        ok2 = Q.grounded_in_pages(sgg_a, [pa]) and Q.grounded_in_pages(sgg_b, [pb]) and sgg_a != sgg_b
        emit(it2, ok2, band="multi")


def main():
    gen_table_numeric()
    gen_format_robustness()
    gen_answerability()
    gen_cross_source()
    gen_lh_retrieval()
    gen_multi_compare_det()

    ok = [i for i in items if i["_ok"]]
    seen_q, deduped, n_dup = set(), [], 0
    for it in ok:
        if it["question"] in seen_q:
            n_dup += 1
            continue
        seen_q.add(it["question"])
        deduped.append(it)
    ok = deduped
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for it in ok:
            f.write(json.dumps({k: v for k, v in it.items() if k != "_ok" and k != "_why"}, ensure_ascii=False) + "\n")
    fam = Counter(i["task_type"] for i in ok)
    print(f"=== DET: emitted {len(ok)} (deduped {n_dup} dup-question) ===")
    for k, v in sorted(fam.items()):
        print(f"   {k:32s} {v}")
    dropped = [i for i in items if not i["_ok"]]
    if dropped:
        print(f"   dropped {len(dropped)} (recompute None / not grounded)")
    print(f"   -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
