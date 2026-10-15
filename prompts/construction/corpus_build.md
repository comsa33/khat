# Worker Prompt: Copyright-Safe KR-Housing-LongRAG-Bench Corpus Build

You are a dataset construction worker for `KR-Housing-LongRAG-Bench`, a Korean long-context benchmark for comparing full-context LLMs, RAG, hybrid RAG, and table/tool pipelines on housing announcements, legal rules, and public housing tabular data.

Your task is to build an internal inspection corpus and annotation-ready dataset without creating copyright risk in the public release.

## Non-Negotiable Copyright Rule

Separate internal work from public release.

- Internal work may download and inspect official source files into `workspace_local/`.
- Public release must not contain raw PDFs, HWPs, OCR text, long extracted passages, screenshots, floor plans, brochures, or private-source text unless explicit redistribution permission is documented.
- Public release may contain source URLs, source IDs, short factual gold answers, evidence locators, row IDs, page/table/section references, deterministic scripts, and license audit notes.

If a source license is unclear, mark it `excluded_pending_permission` and do not use it for QA gold labels.

## Repository Layout

Use these paths:

- `data/source_manifest.jsonl`: approved or candidate source registry
- `data/qa_seed.jsonl`: seed QA items
- `data/task_blueprints.jsonl`: task templates
- `data/excluded_sources.jsonl`: excluded source classes and reasons
- `workspace_local/raw/`: internal-only downloaded source files
- `workspace_local/processed/`: internal-only extracted text, OCR, parsed tables
- `workspace_local/audit/`: internal-only source audit notes, hashes, license screenshots or text notes
- `docs/source_selection_and_license_audit.md`: release policy

## Workstream A: Source Acquisition

For each source in `data/source_manifest.jsonl`:

1. Open the `registry_url` and `access_url`.
2. Confirm provider, title, date, update cycle, and license status.
3. Download official files only from official provider or Public Data Portal links.
4. Save files under `workspace_local/raw/{source_id}/`.
5. Record SHA-256 hash, download URL, downloaded filename, download date, and observed license marker in `workspace_local/audit/{source_id}.json`.
6. If the source has no clear permission, do not download unless needed only for manual review; mark it as excluded.

Expected audit JSON fields:

```json
{
  "source_id": "...",
  "downloaded_at": "YYYY-MM-DD",
  "files": [
    {
      "filename": "...",
      "url": "...",
      "sha256": "...",
      "bytes": 0
    }
  ],
  "license_observation": {
    "status": "usable_public_data_no_known_restriction | usable_statutory_text | excluded_pending_permission",
    "evidence_url": "...",
    "evidence_text_short": "Do not paste long terms; summarize briefly."
  },
  "release_decision": "url_and_labels_only | raw_redistribution_allowed | excluded"
}
```

## Workstream B: Text and Table Extraction

For each approved source:

1. Extract text and tables into `workspace_local/processed/{source_id}/`.
2. Preserve page numbers, section headings, table headings, row/column labels, and source file hash.
3. Do not paraphrase during extraction.
4. For HWP files, prefer a structured converter if available; otherwise record extraction failure and keep the source for manual inspection.
5. For PDFs, extract both text and tables. If OCR is needed, record OCR engine and confidence if available.
6. For CSV/API tables, produce deterministic normalized tables with stable row IDs.

Output formats:

- `document_pages.jsonl`: one record per page or section
- `tables.jsonl`: one record per table with row/column structure
- `chunks.jsonl`: retrieval chunks with source locator
- `extraction_report.json`: tool versions, failures, and confidence notes

Internal extracted records may contain raw text because they are not public release files.

## Workstream C: QA Annotation

Create candidate QA items that test the benchmark's core research question:

Can a long-context model replace or complement RAG and table tools in Korean real-world housing/administrative documents?

MECE task families:

1. Retrieval
   - Single fact from one announcement section or table.
2. Long-distance retrieval
   - Question and evidence separated by long distractor context.
3. Table reasoning
   - Filter, compare, rank, sum, average, min/max, ratio.
4. Cross-document legal reasoning
   - Combine an announcement with housing law/rule text.
5. Cross-source aggregation
   - Combine announcement metadata with HUG or MOLIT public tables.
6. Answerability detection
   - Required evidence is absent or insufficient.
7. Robustness to format
   - Same table represented as extracted text, Markdown table, CSV, or JSON.

Each QA item must include:

- `qa_id`
- `task_type`
- `question`
- `answer`
- `answer_type`
- `source_ids`
- `required_capabilities`
- `evidence` with source ID, filename/hash if applicable, page, section, table ID, row ID, and column names
- `evaluation` metric and gold terms/numbers
- `copyright_note`

Do not include long source excerpts in public QA files. Evidence locators are enough.

## Workstream D: Deterministic Verification

For every numeric/table QA:

1. Write or run deterministic code that derives the answer from normalized rows.
2. Save the code and logs under `workspace_local/audit/`.
3. Include row IDs and filter predicates in the QA evidence.
4. Mark items as `verified_by_script`.

For every textual/legal QA:

1. Require at least two independent evidence locators if the question is cross-document.
2. Use one annotator pass and one reviewer pass.
3. Mark ambiguity explicitly. If the answer depends on interpretation, reject or convert to multiple-choice with a clear rubric.

## Workstream E: Public Release Preparation

Before preparing a public release:

1. Run `python3 scripts/validate_dataset.py`.
2. Search public files for forbidden raw fields: `raw_text`, `raw_content`, `document_text`, `pdf_text`, `hwp_text`, `full_context`.
3. Confirm no raw files exist outside `workspace_local/`.
4. Confirm every source ID in QA appears in `data/source_manifest.jsonl` or `data/excluded_sources.jsonl`.
5. Confirm every source has a license decision.
6. Confirm each QA has evidence locator, not just a source URL.

## Deliverables

Return these deliverables:

1. Updated `data/source_manifest.jsonl` with confirmed source status.
2. `workspace_local/audit/*.json` for every inspected source.
3. `workspace_local/processed/*/extraction_report.json`.
4. At least 50 annotation-ready QA candidates across the MECE task families.
5. A short `docs/annotation_batch_report.md` summarizing:
   - source counts
   - approved/excluded counts
   - task-family counts
   - unresolved license issues
   - extraction failures
   - recommended next batch

## Stop Conditions

Stop and ask before continuing if:

- A source contains private personal data.
- A source has no clear public-data, KOGL, statutory, or written-permission basis.
- A raw file appears necessary for public redistribution.
- You cannot preserve evidence locators.
- The answer requires legal advice rather than benchmark-style document understanding.

