# Claude Code Prompt: Build v0.4-public Benchmark

You are working in:

`<repo>`

Goal: upgrade `v0.3-dev` into a public-ready `v0.4-public` benchmark for Korean real-world long-context + RAG evaluation.

## Read First

1. `docs/v0.4_public_plan.md`
2. `docs/v0.3_batch_report.md`
3. `docs/v0.3_acquisition_status.md`
4. `data/v0.4_announcement_targets_seed.jsonl`

## Current State

v0.3-dev already has:

- 241 verified QA
- HUG sale-history rows
- MOLIT apartment trade-detail rows
- 1 LH announcement
- 32k/64k/128k/256k bundles
- strict no-raw/no-secret release discipline

v0.3 is not public-ready because announcement diversity and public reconstruction are insufficient.

Additional v0.4 source-expansion artifacts may already exist:

- `workspace_local/audit/lh-announcements-v04/summary.json`
- `workspace_local/audit/index_lh_v04.json`
- `workspace_local/processed/lh-sale-announcements-v04/{announcement_id}/document_pages.jsonl`
- `workspace_local/processed/lh-sale-announcements-v04/{announcement_id}/numeric_facts.jsonl`

If these files exist and show 8+ cleanly extracted announcements, do not redo source acquisition unless verification fails. Start from QA generation, bundle construction, verifier extension, and public reconstruction.

## Required Work

### 1. Source Expansion

Run:

```bash
python3 scripts/acquire_lh_announcements_from_manifest.py
```

Then inspect:

```text
workspace_local/audit/lh-announcements-v04/summary.json
```

Keep only targets with official 모집공고문 PDF/HWP. Drop pamphlets/forms-only targets.

### 2. Multi-Announcement Extraction

Create a generalized extractor for all acquired PDFs.

Required outputs:

```text
workspace_local/processed/lh-sale-announcements-v04/{announcement_id}/document_pages.jsonl
workspace_local/processed/lh-sale-announcements-v04/{announcement_id}/numeric_facts.jsonl
workspace_local/processed/lh-sale-announcements-v04/{announcement_id}/extraction_report.json
workspace_local/audit/index_lh_v04.json
```

Each page row must include:

- `source_id`
- `announcement_id`
- `page_id`
- `page_no`
- `locator`
- `char_count`
- `text`

Text stays internal only.

### 3. Build v0.4 QA

Create:

```text
data/qa_v0.4_candidates.jsonl
```

Targets:

- at least 700 QA
- target 900+ if extraction is clean
- at least 8 official announcements if available
- no single announcement may contribute more than 20% of non-table QA

Family targets:

- `long_context_retrieval`: 150
- `long_distance_retrieval`: 100
- `table_numeric_reasoning`: 250
- `cross_document_legal_reasoning`: 120
- `cross_source_aggregation`: 120
- `answerability_detection`: 80
- `format_robustness`: 60
- `multi_document_comparison`: 60

Every QA must include v0.3/v0.4 fields where applicable:

- `bundle_id`
- `context_tier`
- `evidence_position`
- `row_ids`
- `page_ids`
- `announcement_ids`
- `gold_predicate`

### 4. Build v0.4 Bundles

Create:

```text
workspace_local/processed/bundles-v04/manifest.jsonl
```

Tiers:

- 32k
- 64k
- 128k
- 256k
- 512k

Template types:

- announcement-heavy
- law-heavy
- table-heavy

Evidence positions:

- early
- middle
- late
- multi

Bundle text must remain internal.

### 5. Public Reconstruction

Create:

```text
scripts/rebuild_v04_from_public_manifest.py
docs/public_reconstruction.md
docs/release_checklist.md
```

The public reconstruction docs must explain:

- official URL acquisition
- local API key requirements
- where internal raw files are stored
- how to rebuild contexts
- how to validate and verify
- what is and is not redistributed

### 6. Verification

Extend:

```text
scripts/validate_dataset.py
scripts/verify_qa.py
```

Create:

```text
scripts/check_public_release_readiness.py
```

Checks:

- schema valid
- all source IDs known
- all row IDs exist
- all page IDs exist
- all bundle IDs exist
- predicates recompute
- evidence positions match bundle manifests
- no secrets in tracked/public files
- no raw text/full context in public files
- no overlong gold terms
- no duplicate or near-duplicate questions
- no single announcement dominance

### 7. Report

Create:

```text
docs/v0.4_batch_report.md
```

Include:

- source count by provider/region/year/type
- QA count by family
- QA count by source combination
- context tier distribution
- evidence position distribution
- verification results
- public reconstruction status
- human review plan/results
- limitations

## Stop Conditions

Stop and report instead of fabricating if:

- fewer than 6 official announcement PDFs extract cleanly;
- automatic verification fails above 1%;
- the public reconstruction flow cannot be made deterministic;
- a source requires non-public permission;
- QA generation depends on long verbatim excerpts in public files.

## Final Acceptance

Before claiming completion, run:

```bash
python3 scripts/validate_dataset.py
python3 scripts/verify_qa.py
python3 scripts/check_public_release_readiness.py
```

Expected final claim should be one of:

- `v0.4-public ready`
- `v0.4-dev expanded but not public-ready`

Do not overclaim. If it is not public-ready, state exact blockers.
