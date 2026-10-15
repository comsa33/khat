# Claude Code Prompt: Build v0.3 From Acquired Materials

You are working in:

`<repo>`

Goal: produce a v0.3 benchmark batch for the paper framing "Korean real-world long-context + RAG comparison over housing/administrative documents", using only copyright-safe public/official materials. Do not leak secrets or raw copyrighted/full-source text into public files.

## Inputs Already Available

1. HUG key:
   - `workspace_local/secrets/hug_api.key`
   - Do not print or copy the key into any tracked file.

2. HUG real rows:
   - Script: `scripts/acquire_hug_sale_history_rows.py`
   - Output: `workspace_local/processed/hug-sale-history/rows_v0.3.jsonl`
   - Current acquisition: 2023-2025, 16 regions, 624 rows, 623 valid rows.

3. MOLIT apartment trade-detail rows:
   - Script: `scripts/acquire_molit_apt_trade_detail_rows.py`
   - Output: `workspace_local/processed/molit-apt-trade-detail/rows_v0.3.jsonl`
   - Current acquisition: 2025, 6 districts, 72 calls, 20,370 rows.
   - Districts: 서울 종로구, 서초구, 강남구, 송파구; 대전 동구, 유성구.

4. LH official announcement:
   - Script: `scripts/acquire_lh_announcement_files.py`
   - Raw PDF/HWP: `workspace_local/raw/lh-sale-announcements/`
   - Audit: `workspace_local/audit/lh-daedong2-b1-public-sale-20251017.json`
   - Title: 대전대동2 1블록 공공분양 입주자모집공고

5. Existing v0.2:
   - `data/qa_v0.2_candidates.jsonl` has 70 verified seed QA.
   - Treat it as seed only, not a paper-grade experiment set.

## Non-Negotiable Constraints

1. Do not put raw source text, full context, PDF text, HWP text, raw API payloads, or secrets in tracked/public files.
2. Public QA files may contain:
   - question
   - short answer
   - source IDs
   - row/page/article locators
   - deterministic predicates
   - metadata needed for evaluation
3. Public QA files must not contain:
   - `raw_text`
   - `document_text`
   - `full_context`
   - long evidence excerpts
   - the HUG API key
4. If a task cannot be deterministically grounded, drop it.
5. Be honest in reports: with one LH announcement, call this v0.3 development/seed batch unless more announcements are added.

## Required Work, MECE

### A. Extract LH Announcement

1. Run or improve `scripts/extract_lh_announcement_text.py`.
2. Produce:
   - `workspace_local/processed/lh-sale-announcements/document_pages.jsonl`
   - `workspace_local/processed/lh-sale-announcements/numeric_facts.jsonl`
   - `workspace_local/processed/lh-sale-announcements/extraction_report_v0.3.json`
3. Check whether page text contains the key facts needed for QA:
   - 공급위치
   - 공급대상
   - 입주자모집공고일
   - 주택관리번호
   - 청약/당첨/계약 일정
   - 타입별 세대수
   - 분양가격/계약금/중도금/잔금
   - 거주지역 우선조건
   - 중복청약/부적격 조건

### B. Build Internal Evidence Index

Create or update internal index files under `workspace_local/processed/`:

1. LH page index:
   - page_no
   - locator
   - headings if detectable
   - numeric facts
   - short local snippets for verifier only

2. HUG row index:
   - `_row_id`
   - year
   - area
   - business site name
   - household counts
   - sale price field
   - announcement/open/guarantee dates

3. MOLIT row index:
   - `_row_id`
   - 법정동코드/지역명/month
   - apartment name
   - deal amount
   - exclusive-use area
   - floor
   - build year
   - legal dong name
   - registration date where present

4. Existing legal/statute index:
   - reuse `workspace_local/processed/law-*/document_pages.jsonl`
   - reuse existing numeric/table files.

### C. Upgrade QA Schema For v0.3

Add optional v0.3 fields while keeping validator compatible:

1. `bundle_id`
2. `context_tier`: one of `32k`, `64k`, `128k`, `256k`, `512k`
3. `evidence_position`: one of `early`, `middle`, `late`, `multi`
4. `row_ids`: list of source row IDs for table QA
5. `page_ids`: list of LH page IDs for announcement QA
6. `gold_predicate`: deterministic filter/aggregation expression or structured JSON object

Update:

1. `data/task_schema.json`
2. `scripts/validate_dataset.py`
3. `scripts/verify_qa.py`

### D. Generate v0.3 QA Candidates

Target `data/qa_v0.3_candidates.jsonl`.

Minimum acceptable batch if only current materials are available:

| Family | Target |
|---|---:|
| long_context_retrieval | 35 |
| table_numeric_reasoning | 45 |
| cross_document_legal_reasoning | 25 |
| cross_source_aggregation | 25 |
| long_distance_retrieval | 20 |
| answerability_detection | 20 |
| format_robustness | 15 |
| Total | 185 |

If more official announcements are added, scale to 300+.

Generation rules:

1. LH retrieval QA must cite `lh-sale-announcements` and page IDs.
2. HUG table QA must cite `_row_id` values and use deterministic `gold_predicate`.
3. Cross-document legal QA must require both the LH announcement and a statute/rule source.
4. Cross-source aggregation QA must combine at least two sources, for example LH announcement facts + HUG rows or HUG rows + legal thresholds.
5. Answerability detection must be grounded by absence checks over the selected source set, not by guesswork.
6. Avoid duplicate questions with only wording changes.
7. Do not generate questions whose answer depends on current law unless the source version/effective date is explicit.

### E. Materialize Context Bundles

Create internal bundles under:

`workspace_local/processed/bundles/`

Minimum:

1. `bundle_lh_hug_law_32k`
2. `bundle_lh_hug_law_64k`
3. `bundle_lh_hug_law_128k`
4. `bundle_lh_hug_law_256k`

Bundle rules:

1. Use only internal text from `workspace_local/processed`.
2. Include target evidence plus distractor pages/rows/statutes.
3. Control `evidence_position` by placing target evidence early/middle/late.
4. Record bundle metadata in `workspace_local/processed/bundles/manifest.jsonl`.
5. Do not put bundle text in tracked files.

### F. Verify

Verification must include:

1. `python3 scripts/validate_dataset.py`
2. deterministic QA verification:
   - exact row exists for each `row_id`
   - exact page exists for each `page_id`
   - `gold_predicate` recomputes answer for table/aggregation QA
   - answer string is grounded in cited internal source or computed from cited rows
3. no raw fields in public files
4. no secret leakage:
   - search tracked files for the HUG key prefix/value
   - search tracked files for forbidden fields

### G. Report

Write `docs/v0.3_batch_report.md` with:

1. What was acquired
2. What was extracted
3. QA count by family
4. Count by source combination
5. Count by context tier
6. Verification result
7. Known limitations
8. Next batch needs:
   - MOLIT/data.go.kr key
   - more official announcement PDFs/HWPs
   - human review sample

## Stop Conditions

Stop and report instead of fabricating if:

1. LH extraction is too noisy to locate facts reliably.
2. HUG rows contain API error rows only.
3. A proposed QA answer cannot be recomputed or found in cited evidence.
4. Any public output would require embedding long raw source text.

## Expected Final State

1. `data/qa_v0.3_candidates.jsonl` exists and validates.
2. `workspace_local/processed/bundles/manifest.jsonl` exists.
3. `docs/v0.3_batch_report.md` exists.
4. `scripts/validate_dataset.py` passes.
5. Public files contain no secrets and no raw corpus text.
