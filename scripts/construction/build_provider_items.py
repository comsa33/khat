#!/usr/bin/env python3
"""Deterministic items over the announcements of the other providers and cross-provider comparisons.

Usage: python3 scripts/construction/build_provider_items.py
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import providers as P
import table_rows as C
import lh_announcements as Q

OUT = Q.LOCAL / "construction" / "provider_items.jsonl"
LH = "lh-sale-announcements-v04"

NUMERIC_CELL_RE = re.compile(r"^[\d][\d,.\s~∼\-/㎡%％원만천억세대명호층년월일점배]*$")
HEADER_UNIT_RE = re.compile(r"(만원|천원|억원|원|㎡|％|%|세대|명|호|층|점|년|개월|월|일|배|평|보증금|임대료)")
COND_RE = re.compile(r"(%|％|이하|이상|미만|초과|해당|본인|구성원|무주택|만원|천원|억원|개월|이내|점|배|보증금|임대료)")
SUPPLY_KW = ["특별공급", "일반공급", "신혼부부", "생애최초", "노부모부양", "다자녀", "신생아", "기관추천", "우선공급", "추첨", "임대", "보증금", "월임대료"]
ELIG_KW = ["소득", "자산", "무주택", "거주", "입주자저축", "청약통장", "주택소유", "세대구성원", "재당첨", "월평균", "가구원", "보증금", "임대료", "임대조건"]
DATE_RE = re.compile(r"20\d{2}\s*[.\-년]\s*\d{1,2}\s*[.\-월]\s*\d{1,2}")

items = []
pat = Counter()


def emit(it, ok, band="multi"):
    it["_ok"] = ok
    it["_band"] = band
    items.append(it)


def title(a):
    if P.source_of(a) == LH:
        t = Q.ann_meta().get(a, {}).get("title_from_audit", "") or a
    else:
        rep = next((r for r in P._new_reports() if r["announcement_id"] == a), {})
        t = rep.get("title", "") or a
    import html as _html
    t = _html.unescape(t)
    t = re.sub(r"getDetailView\s*\([^)]*\)|return\s+false|on\w+\s*=\s*\S+|javascript:", " ", t, flags=re.I)
    t = re.sub(r'[;"\'<>]', " ", t)
    return re.sub(r"\s+", " ", t).strip()[:60]


def meta(a, task_type, caps):
    mf = P.meta_fields(a)
    return {"split": mf["split"], "task_type": task_type, "provider": mf["provider"],
            "region_sido": mf["region_sido"], "region_sigungu": mf["region_sigungu"],
            "housing_type": mf["housing_type"], "source_ids": [P.source_of(a)],
            "required_capabilities": caps, "announcement_ids": [a]}


def unique_pairs(cells):
    by_t = defaultdict(lambda: defaultdict(list))
    for c in cells:
        by_t[c["table_id"]][(c["row_header"], c["col_header"])].append(c)
    return [cs[0] for t in by_t.values() for cs in t.values() if len(cs) == 1]


# ---------------- cell numeric / eligibility / schedule (new providers only) ----------------
def gen_cells(per_ann=70, per_table=10):
    new = [a for a in P.announcement_ids() if P.source_of(a) != LH and P.cells(a)]
    for a in new:
        t = title(a)
        cells = unique_pairs(P.cells(a))
        n = 0
        tcount = Counter()
        for c in sorted(cells, key=lambda x: (x["page_id"], x["table_id"], x["row_index"], x["col_index"])):
            if n >= per_ann:
                break
            rh, ch, ct = c["row_header"], c["col_header"], c["cell_text"]
            if not (rh and ch and ct) or rh == ct or ch == ct or len(ct) > 24 or c["confidence"] < 0.9:
                continue
            if len(rh) > 30 or len(ch) > 30 or tcount[c["table_id"]] >= per_table:
                continue
            if c.get("normalized_value") is None or not NUMERIC_CELL_RE.match(ct):
                continue
            if not (c.get("unit") or HEADER_UNIT_RE.search(ch) or HEADER_UNIT_RE.search(rh)):
                continue
            pid = c["page_id"]
            if not (P.grounded(ct, [pid]) and P.grounded(rh, [pid]) and P.grounded(ch, [pid])):
                continue
            if pat[(rh, ch)] >= 3:
                continue
            it = meta(a, "table_numeric_reasoning", ["table_cell_lookup", "table_reading"])
            it.update({"question": f"「{t}」(p.{int(c['page_id'].split('-p')[-1])})의 표에서 '{rh}' 항목의 '{ch}' 값은 무엇인가?",
                       "answer": ct, "answer_type": "string",
                       "evidence": [{"source_id": P.source_of(a), "locator": f"{c['table_id']} '{rh}'×'{ch}'"}],
                       "evaluation": {"metric": "contains_all", "gold_terms": [ct]},
                       "page_ids": [pid], "table_ids": [c["table_id"]], "cell_ids": [P.cell_id(c)],
                       "copyright_note": "공고 표 셀 단답(행·열 머리글 교차). 원문 전체는 내부 전용."})
            emit(it, True)
            tcount[c["table_id"]] += 1
            pat[(rh, ch)] += 1
            n += 1


def gen_eligibility(per_ann=42):
    new = [a for a in P.announcement_ids() if P.source_of(a) != LH and P.cells(a)]
    for a in new:
        t = title(a)
        n = 0
        for c in sorted(unique_pairs(P.cells(a)), key=lambda x: (x["page_id"], x["row_index"])):
            if n >= per_ann:
                break
            rh, ch, ct = c["row_header"], c["col_header"], c["cell_text"]
            if not (rh and ch and ct) or rh == ct or ch == ct or len(ct) > 30 or c["confidence"] < 0.9:
                continue
            if len(rh) > 28 or len(ch) > 28:
                continue
            kw = [h for h in (rh, ch) if any(k in h for k in SUPPLY_KW + ELIG_KW)]
            if not kw or not COND_RE.search(ct):
                continue
            concept = kw[0]
            crit = ch if concept == rh else rh
            pid = c["page_id"]
            if not (P.grounded(ct, [pid]) and P.grounded(concept, [pid]) and P.grounded(crit, [pid])):
                continue
            if pat[("E", concept, crit)] >= 3:
                continue
            cap = "lease_condition_reasoning" if any(k in (rh + ch) for k in ("보증금", "임대료", "임대조건")) else "eligibility_reasoning"
            it = meta(a, "eligibility_reasoning", [cap, "table_reading", "multi_hop_reasoning"])
            it.update({"question": f"「{t}」(p.{int(c['page_id'].split('-p')[-1])}) 표에서 '{concept}'의 '{crit}' 기준은 무엇인가?",
                       "answer": ct, "answer_type": "string",
                       "evidence": [{"source_id": P.source_of(a), "locator": f"{c['table_id']} '{concept}'×'{crit}'"}],
                       "evaluation": {"metric": "contains_all", "gold_terms": [ct]},
                       "page_ids": [pid], "table_ids": [c["table_id"]], "cell_ids": [P.cell_id(c)],
                       "copyright_note": "공고 자격/임대조건 표 셀 단답. 원문 전체는 내부 전용."})
            emit(it, True)
            pat[("E", concept, crit)] += 1
            n += 1


def gen_schedule(per_ann=12):
    new = [a for a in P.announcement_ids() if P.source_of(a) != LH and P.cells(a)]
    for a in new:
        t = title(a)
        n = 0
        seen = set()
        for c in P.cells(a):
            if n >= per_ann:
                break
            rh, ch, ct = c["row_header"], c["col_header"], c["cell_text"]
            if not DATE_RE.search(ct) or len(ct) > 40:
                continue
            label = next((h for h in (rh, ch) if h and h not in ct and ct not in h and 2 <= len(h) <= 28), None)
            if not label:
                continue
            pid = c["page_id"]
            if not (P.grounded(ct, [pid]) and P.grounded(label, [pid])) or (label, ct) in seen:
                continue
            seen.add((label, ct))
            it = meta(a, "schedule_reasoning", ["schedule_reasoning", "table_reading"])
            it.update({"question": f"「{t}」(p.{int(c['page_id'].split('-p')[-1])}) 일정 표에서 '{label}'의 일정(일자/시간)은 무엇인가?",
                       "answer": ct, "answer_type": "string",
                       "evidence": [{"source_id": P.source_of(a), "locator": f"{c['table_id']} '{label}' 일정"}],
                       "evaluation": {"metric": "contains_all", "gold_terms": [ct[:40]]},
                       "page_ids": [pid], "table_ids": [c["table_id"]], "cell_ids": [P.cell_id(c)],
                       "copyright_note": "공고 일정 표 셀 단답. 원문 전체는 내부 전용."})
            emit(it, True)
            n += 1


# ---------------- provider / region comparison ----------------
def _confident_region_anns():
    out = []
    for a in P.announcement_ids():
        sr = P.short_region(a)
        if sr and P.region_sido(a) and P.grounded(sr, [P.page_id_for(a, 1)]) and P.grounded(P.region_sido(a), [P.page_id_for(a, 1)]):
            out.append(a)
    return out


def gen_provider_comparison(max_items=44):
    anns = [a for a in _confident_region_anns() if P.split_of(a) == "dev"]
    by_prov = defaultdict(list)
    for a in anns:
        by_prov[P.source_of(a)].append(a)
    provs = sorted(by_prov)
    pairs = []
    for i in range(len(provs)):
        for j in range(i + 1, len(provs)):
            ai, aj = by_prov[provs[i]], by_prov[provs[j]]
            for k in range(min(6, len(ai), len(aj))):
                pairs.append((ai[k], aj[k]))
    seen = set()
    for a, b in pairs:
        if sum(1 for it in items if it.get("task_type") == "provider_comparison") >= max_items:
            break
        if (a, b) in seen:
            continue
        seen.add((a, b))
        pa, pb = P.page_id_for(a, 1), P.page_id_for(b, 1)
        sa, sb = P.region_sido(a), P.region_sido(b)
        pra, prb = P.provider_of(a), P.provider_of(b)
        it = {"split": "dev", "task_type": "provider_comparison",
              "provider": "복수(비교)", "region_sido": "복수", "region_sigungu": "복수",
              "housing_type": "복수",
              "source_ids": sorted({P.source_of(a), P.source_of(b)}),
              "required_capabilities": ["provider_comparison", "multi_document", "cross_provider"],
              "question": f"공급기관이 다른 두 입주자모집공고 「{title(a)}」({pra})와 「{title(b)}」({prb})의 공급위치 시·도를 각각 제시하라.",
              "answer": f"{pra}={sa} / {prb}={sb}", "answer_type": "string",
              "evidence": [{"source_id": P.source_of(a), "locator": f"{pa} 공급위치"},
                           {"source_id": P.source_of(b), "locator": f"{pb} 공급위치"}],
              "evaluation": {"metric": "contains_all", "gold_terms": [sa, sb]},
              "page_ids": [pa, pb], "announcement_ids": [a, b],
              "copyright_note": "서로 다른 공급기관 두 공고의 공급위치(시·도) 대조. 공개물은 locator+단답만."}
        emit(it, P.grounded(sa, [pa]) and P.grounded(sb, [pb]) and len(it["source_ids"]) == 2)


def gen_region_comparison(max_items=34):
    anns = [a for a in _confident_region_anns() if P.split_of(a) == "dev"]
    seen = set()
    cnt = 0
    for i in range(len(anns)):
        if cnt >= max_items:
            break
        a = anns[i]
        b = anns[(i + 5) % len(anns)]
        if a == b:
            continue
        ra, rb = P.short_region(a), P.short_region(b)
        if ra == rb:
            continue
        key = tuple(sorted((ra, rb)))
        if key in seen:
            continue
        seen.add(key)
        pa, pb = P.page_id_for(a, 1), P.page_id_for(b, 1)
        it = {"split": "dev", "task_type": "region_comparison",
              "provider": "복수(비교)", "region_sido": "복수", "region_sigungu": "복수", "housing_type": "복수",
              "source_ids": sorted({P.source_of(a), P.source_of(b)}),
              "required_capabilities": ["region_comparison", "multi_document"],
              "question": f"두 입주자모집공고 「{title(a)}」와 「{title(b)}」의 공급위치 시·군·구(또는 시)를 각각 제시하라.",
              "answer": f"{ra} / {rb}", "answer_type": "string",
              "evidence": [{"source_id": P.source_of(a), "locator": f"{pa} 공급위치"},
                           {"source_id": P.source_of(b), "locator": f"{pb} 공급위치"}],
              "evaluation": {"metric": "contains_all", "gold_terms": [ra, rb]},
              "page_ids": [pa, pb], "announcement_ids": [a, b],
              "copyright_note": "두 공고의 공급위치(시·군·구) 대조. 공개물은 locator+단답만."}
        emit(it, P.grounded(ra, [pa]) and P.grounded(rb, [pb]))
        cnt += 1


def _facts(ann_id):
    d = P.registry().get(ann_id, {}).get("dir")
    if not d:
        return []
    fp = d / "numeric_facts.jsonl"
    return [json.loads(l) for l in fp.open(encoding="utf-8") if l.strip()] if fp.exists() else []


def gen_retrieval(per_ann=20):
    """long_context_retrieval cloze from unique-in-announcement numeric facts (all non-LH providers)."""
    for a in [x for x in P.announcement_ids() if P.source_of(x) != LH]:
        facts = _facts(a)
        if not facts:
            continue
        vc = Counter(f["value_text"] for f in facts)
        n, seen = 0, set()
        for f in facts:
            if n >= per_ann:
                break
            val = f["value_text"]
            if vc[val] != 1 or val in seen or len(val) < 2:
                continue
            snip = f.get("local_snippet", "")
            i = snip.find(val)
            if i < 0:
                continue
            pre = snip[max(0, i - 20):i]
            if " " in pre:
                pre = pre[pre.index(" ") + 1:]
            post = snip[i + len(val):i + len(val) + 8]
            if " " in post:
                post = post[:post.rindex(" ")]
            stem = f"{pre}____{post}".strip()
            if not (4 <= len(stem) <= 30):
                continue
            pid = f["page_id"]
            if pid not in P.all_page_ids() or not P.grounded(val, [pid]):
                continue
            seen.add(val)
            n += 1
            it = meta(a, "long_context_retrieval", ["long_context_retrieval", "announcement_reading"])
            it.update({"question": f"「{title(a)}」(p.{f['page_no']})에서 다음 빈칸에 들어갈 값은? — \"{stem}\"",
                       "answer": val, "answer_type": "span",
                       "evidence": [{"source_id": P.source_of(a), "locator": f"{f.get('locator', '')} (p.{f['page_no']})"}],
                       "evaluation": {"metric": "contains_all", "gold_terms": [val]},
                       "page_ids": [pid],
                       "copyright_note": "공고 단답 + 짧은 cloze 스템(공공기관 입주자모집공고). 원문 전체는 내부 전용."})
            emit(it, True, band="early")


def gen_answerability():
    """Cross-announcement absence: only A provided -> B's 시군구 unanswerable (B token absent from A)."""
    anns = [x for x in P.announcement_ids() if P.source_of(x) != LH]
    for i, a in enumerate(anns):
        for off in (5, 11):
            b = anns[(i + off) % len(anns)]
            if a == b:
                continue
            btok = P.region_sigungu(b)
            btok = btok.split()[-1] if (btok and " " in btok) else btok
            if not btok:
                continue
            a_pages = [p["page_id"] for p in P.pages(a)]
            if P.grounded(btok, a_pages):
                continue
            pid = P.page_id_for(a, 1)
            it = meta(a, "answerability_detection", ["answerability_detection", "coverage_check"])
            it.update({"question": f"제공된 입주자모집공고가 「{title(a)}」 1건뿐일 때, 「{title(b)}」의 공급위치가 속한 시·군·구를 확정할 수 있는가?",
                       "answer": "확정할 수 없음(unanswerable). 제공된 공고에는 해당 공고의 공급위치 정보가 없음.",
                       "answer_type": "boolean_with_reason",
                       "evidence": [{"source_id": P.source_of(a), "locator": f"{pid} 제공 시 타 공고 공급위치 부재"}],
                       "evaluation": {"metric": "boolean_and_reason", "gold_terms": ["확정할 수 없"]},
                       "page_ids": [pid],
                       "copyright_note": "negative control(문서 간 부재). 타 공고 지역 토큰이 제공 공고에 부재함을 결정론으로 검증."})
            emit(it, not P.grounded(btok, a_pages), band="early")


def gen_multi_document(max_items=70):
    """Same-provider, same-split announcement pairs comparing 공급위치 시·군·구 (>=2 announcements cited)."""
    bysrc = defaultdict(list)
    for a in P.announcement_ids():
        if P.source_of(a) == LH:
            continue
        bysrc[P.source_of(a)].append(a)
    cnt, seen = 0, set()
    for src, anns in sorted(bysrc.items()):
        conf = [a for a in anns if P.short_region(a) and P.grounded(P.short_region(a), [P.page_id_for(a, 1)])]
        for i in range(len(conf)):
            for j in range(i + 1, len(conf)):
                if cnt >= max_items:
                    return
                a, b = conf[i], conf[j]
                if P.split_of(a) != P.split_of(b):
                    continue
                ra, rb = P.short_region(a), P.short_region(b)
                if ra == rb:
                    continue
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                pa, pb = P.page_id_for(a, 1), P.page_id_for(b, 1)
                it = {"split": P.split_of(a), "task_type": "multi_document_comparison",
                      "provider": P.provider_of(a), "region_sido": "복수", "region_sigungu": "복수", "housing_type": "복수",
                      "source_ids": [P.source_of(a)],
                      "required_capabilities": ["multi_document", "comparison"],
                      "question": f"같은 공급기관({P.provider_of(a)})의 두 공고 「{title(a)}」와 「{title(b)}」의 공급위치 시·군·구(또는 시)를 각각 제시하라.",
                      "answer": f"{ra} / {rb}", "answer_type": "string",
                      "evidence": [{"source_id": P.source_of(a), "locator": f"{pa} 공급위치"},
                                   {"source_id": P.source_of(a), "locator": f"{pb} 공급위치"}],
                      "evaluation": {"metric": "contains_all", "gold_terms": [ra, rb]},
                      "page_ids": [pa, pb], "announcement_ids": [a, b],
                      "copyright_note": "동일 공급기관 두 공고의 공급위치 대조. 공개물은 locator+단답만."}
                emit(it, P.grounded(ra, [pa]) and P.grounded(rb, [pb]))
                cnt += 1


def gen_cross_source_hug():
    """cross_source_aggregation: non-LH announcement 시·도 -> HUG 분양이력 aggregation (recompute-verified).

    The region hop term must be grounded in the cited announcement page; the HUG aggregate is recomputed
    by qa_v03_common.recompute, so the number is correct by construction and independently re-checkable.
    """
    region_count = Counter()
    for a in [x for x in P.announcement_ids() if P.source_of(x) != LH]:
        sido = P.region_sido(a)
        hug_region = Q.SIDO_TO_HUG.get(sido)
        if not hug_region:
            continue
        if region_count[hug_region] >= 6:
            continue
        pid = P.page_id_for(a, 1)
        short = P.short_region(a)
        if not (P.grounded(short, [pid]) or P.grounded(sido, [pid])):
            continue
        region_count[hug_region] += 1
        gterm = short if P.grounded(short, [pid]) else sido
        for year in ["2023", "2024", "2025"]:
            base = {"source": C.HUG, "filter": {"_query_area_name": hug_region, "_query_year": year}}
            for op, field, unit, lab in [("count", None, "건", "분양 사업장(분양보증) 건수"),
                                         ("sum", "TOT_HOCO", "세대", "분양 사업장 총세대수(TOT_HOCO) 합"),
                                         ("avg", "TOT_HOCO", "세대", "분양 사업장 평균 총세대수(TOT_HOCO)")]:
                pred = {**base, "op": op}
                if field:
                    pred["field"] = field
                v, ids = C.recompute(pred)
                if v is None or (op == "count" and v == 0):
                    continue
                it = meta(a, "cross_source_aggregation",
                          ["cross_source_aggregation", "multi_hop_reasoning", "table_filtering"])
                it["source_ids"] = [P.source_of(a), C.HUG]
                it.update({
                    "question": f"「{title(a)}」(p.1)의 공급위치가 속한 시·도에 대하여, HUG 분양이력정보에서 {year}년 {lab}은 얼마인가?",
                    "answer": f"{v:,}{unit}", "answer_type": "number",
                    "evidence": [{"source_id": P.source_of(a), "locator": f"{pid} 공급위치(지역 식별)"},
                                 {"source_id": C.HUG, "locator": f"predicate: {C.predicate_human(pred)}"}],
                    "evaluation": {"metric": "exact_numbers", "gold_numbers": [str(v)], "gold_terms": [gterm]},
                    "page_ids": [pid], "row_ids": ids[:20], "gold_predicate": pred,
                    "copyright_note": "공고 사실(공급위치) + 공공데이터(HUG) 집계 결합. 공개물은 locator+predicate+단답만."})
                emit(it, P.grounded(gterm, [pid]), band="middle")


def main():
    gen_cells()
    gen_eligibility()
    gen_schedule()
    gen_retrieval()
    gen_answerability()
    gen_multi_document()
    gen_cross_source_hug()
    gen_provider_comparison()
    gen_region_comparison()
    ok = [i for i in items if i["_ok"]]
    seen_q, dedup, ndup = set(), [], 0
    for it in ok:
        if it["question"] in seen_q:
            ndup += 1
            continue
        seen_q.add(it["question"])
        dedup.append(it)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for it in dedup:
            f.write(json.dumps({k: v for k, v in it.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
    fam = Counter(i["task_type"] for i in dedup)
    print(f"=== PROVIDERS QA: {len(dedup)} (deduped {ndup}, dropped {len(items)-len(ok)}) ===")
    for k, v in sorted(fam.items()):
        print(f"   {k:30s} {v}")
    print(f"   -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
