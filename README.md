# KHAT

KHAT (Korean Housing Announcements and Tables) is a Korean benchmark for long-context and retrieval-augmented question answering over public housing announcements, public housing tables, and housing statutes. Each item is a question, a short gold answer, and evidence locators into a document bundle of 32k to 512k tokens.

This repository holds the data, the scripts that build the bundles and run the evaluations, and the model answers, judge verdicts, and human labels behind the reported results.

## Contents

| Path | Contents |
|---|---|
| `data/qa_dev.jsonl` | 1,541 `dev` items |
| `data/qa_test_public.jsonl` | 347 `test_public` items |
| `data/auxiliary/legal_reasoning.jsonl` | 109 legal-reasoning items (not evaluated; see the data card) |
| `data/errata/`, `data/gold_fixes.json` | corrections to gold answers and locators; the question files keep the original values |
| `data/exclusions/` | items left out of evaluation |
| `data/item_sets/` | item samples and subsets used by the runs |
| `data/source_manifest.jsonl`, `data/bundle_sha256.csv` | source documents and the SHA-256 of every rebuilt bundle |
| `data/tables/molit_apt_trade_rows.jsonl` | MOLIT apartment trade rows used by the bundles and gold predicates, without building number, lot and road address, registration date, buyer, seller and agent fields, and other fields the scripts do not read, obtained with `scripts/build/acquire_molit_rows.py` |
| `results/predictions/` | model answers |
| `results/verdicts/<judge>/` | per-item judge verdicts |
| `results/human_labels/` | human labels of the reviews, the expert review and the judge validations |
| `results/{two_stage,same_day,exec}/` | runs of the two-stage, same-day re-run and program-execution settings |
| `scripts/` | build, run, judge, scoring, analysis and table scripts |
| `src/khat/` | shared paths |

## Setup

```bash
uv sync --frozen
uv run --extra test pytest
```

Scripts are run from the repository root, for example `uv run python scripts/tables/make_tables.py`. Files that cannot be redistributed (source documents, bundle text, run prompts) are rebuilt under `local/`. To keep them elsewhere, copy `configs/paths.example.toml` to `configs/paths.toml` or set `KHAT_LOCAL`. The retrieval-coverage analyses and the k=50 scorers rank passages with `bge-m3`; they need a local Ollama server with `bge-m3` or the embedding cache in `local/cache/`.

## Rebuilding sources and bundles

The source documents stay at their official portals. `scripts/build/` downloads them, extracts page text and the HUG table rows, and assembles the bundles with the MOLIT rows in `data/tables/`. The reported runs used bundles in which MOLIT table rows carry their row identifiers; `scripts/build/build_bundles.py --with-row-ids` rebuilds those bundles, and the default build leaves the identifiers out. `data/bundle_sha256.csv` gives the hash of every bundle of the reported runs. These hashes are reproduced with Python 3.10 and a fixed hash seed:

```bash
PYTHONHASHSEED=1 uv run --python 3.10 --extra build python scripts/build/build_bundles.py --with-row-ids
```

## Reproducing the reported results

Each command runs from the repository root as `uv run python <script> [arguments]` and prints the values of the listed results. Commands marked † also read files rebuilt under `local/` by `scripts/build/` and `scripts/run/`; the others read only files in this repository.

| Result | Command |
|---|---|
| Dataset counts, Table 3, data card | `scripts/data/datasheet_counts.py`, `scripts/data/dataset_facts.py`, `scripts/data/scope_counts.py`, `scripts/data/verify_qa.py --qa data/construction/qa_realistic_candidates.jsonl`† |
| Dev review | `scripts/data/dev_review.py` |
| Expert review | `scripts/data/expert_review.py` |
| Main scores (judge `claude-sonnet-5-5`) and retrieval recall | `scripts/score/score_test.py` |
| Table 7 under the primary judge | `scripts/judges/kimi_cells.py results/verdicts/kimi-k2.6` |
| Tables 7 and 8 and the generated SI tables | `scripts/tables/make_tables.py results/two_stage OUT_DIR --exec results/exec/main --same-day results/same_day`† |
| Two-stage run | `scripts/two_stage/two_stage_analysis.py results/two_stage`†, `scripts/two_stage/make_cells.py`, `scripts/two_stage/table_share.py results/two_stage`†, `scripts/two_stage/region_rule.py results/two_stage`†, `scripts/two_stage/region_rule_compare.py results/two_stage`† |
| Same-day re-run (bv1–bv3: versions 1–3 of the region rule) | `scripts/same_day/same_day_analysis.py results/same_day results/two_stage`† |
| Program execution | `scripts/exec/exec_analysis.py results/exec/main --two-stage results/two_stage`† |
| Cluster bootstrap | `scripts/analysis/cluster_bootstrap.py results/two_stage --judge kimi`† |
| Primary-judge values | `scripts/analysis/primary_judge_numbers.py results/two_stage`†, `scripts/analysis/primary_judge_numbers_extra.py`, `scripts/analysis/si_cells.py` |
| Judge agreement and judge errors | `scripts/judges/judge_agreement.py`, `scripts/judges/judge_error_review.py`, `scripts/judges/secondary_agreement.py`, `scripts/score/score_judge.py --verdicts-dir results/verdicts/gpt-4.1-mini_original_gold --splits test_public --exclude-family cross_document_legal_reasoning` |
| Judge validation and judge panel | `scripts/judges/judge_validation_score.py --sample results/human_labels/judge_validation_200/sample.csv --ann-a results/human_labels/judge_validation_200/returned/annotator_A.csv --ann-b results/human_labels/judge_validation_200/returned/annotator_B.csv --adjudication results/human_labels/judge_validation_200/adjudication_filled.csv --dataset .`, `scripts/judges/judge_validation_sample.py`, `scripts/judges/judge_validation_kappa.py`, `scripts/judges/judge_panel_kappa.py`, `scripts/judges/panel_claude_compare.py` |
| Corrected gold | `scripts/score/score_corrected_gold.py`, `scripts/tables/corrected_gold_cells.py`, `scripts/analysis/accept_any_drop.py`, `scripts/analysis/answer_set_bound.py` |
| Closed book | `scripts/score/score_closed_book.py`, `scripts/score/score_reruns.py`†, `scripts/analysis/closed_book_by_family.py`, `scripts/analysis/answer_classes.py`†, `scripts/analysis/answer_class_scores.py`, `scripts/analysis/knowledge_cutoff.py` |
| Aggregation items | `scripts/analysis/aggregation_split.py`†, `scripts/analysis/molit_cancelled.py`†, `scripts/data/aggregation_evidence_check.py`† |
| Full-context errors | `scripts/analysis/full_context_errors.py` |
| Retrieval coverage | `scripts/analysis/no_marker_recall.py`†, `scripts/analysis/identifier_recall.py`, `scripts/analysis/retrieval_coverage.py`†, `scripts/analysis/retrieval_coverage_extra.py`†, `scripts/analysis/retrieval_joint_evidence.py`†, `scripts/analysis/answer_repeat_pages.py`† |
| k=50 and page-level k=50 | `scripts/score/score_k50.py`†, `scripts/score/score_k50_page.py`† |
| Run-to-run variation | `scripts/analysis/run_variance.py results/verdicts/gpt-4.1-mini results/verdicts/gpt-4.1-mini_original_gold` |
| Strict match and judge | `scripts/analysis/strict_vs_judge.py results/verdicts/gpt-4.1-mini_original_gold` |
| Paired tests, unanswerable items, token counts | `scripts/analysis/paired_tests.py`, `scripts/analysis/unanswerable_classes.py`, `scripts/analysis/cost_counts.py`† |

Reported scores use the primary judge `kimi-k2.6`; verdicts of `claude-sonnet-5-5` and `gpt-4.1-mini` are given beside them. `results/verdicts/panel/panel_kimi-k2.6_*` is the panel run; reported values use `results/verdicts/kimi-k2.6/`. Where a prediction was re-run, the scores use the first call; `data/first_call_items.csv` lists the items whose first call was empty or missing, which are scored wrong. Every reported 512k result uses the 512k bundle listed in `data/bundle_sha256.csv`.

## Data card

**Purpose.** KHAT is an evaluation resource for comparing retrieval with the full context on Korean public housing documents. It is not a training corpus, and it must not be used to make real housing-eligibility or allocation decisions.

**Composition.** 1,888 items in 11 task families (`dev` 1,541, `test_public` 347), of which 1,523 `dev` and 342 `test_public` items are evaluated after the review exclusions. The 109 items of the legal-reasoning family cite 2015 and 2018 statute versions and are kept in `data/auxiliary/`, with a check of each item against current law in `data/auxiliary/legal_errata.csv`. Each item carries the question, the gold answer, evidence locators (page, table, cell and row identifiers), public metadata (provider, region, housing type, bundle, context tier, near-duplicate cluster, split) and, for answers computed from table rows, a gold predicate. Errata files list every correction; exclusion lists name every item left out of evaluation.

**Sources.** 41 announcements from 10 public housing providers in 9 regions, the HUG sale-history table, the MOLIT apartment trade table, and three housing statutes and rules. Apart from the MOLIT trade rows in `data/tables/`, the source documents are not redistributed and remain under their original terms.

**Annotation.** Four non-author native speakers of Korean, none of them domain experts, reviewed the items and model answers. They worked independently of each other and of the first author, were instructed not to use AI tools, and took part as volunteers. Each received gift vouchers as thanks. Annotator identities are replaced by letter codes. A housing-subscription consultant with eight years of practice and no employment or authorship relation to the authors reviewed 110 `test_public` items and received a gift voucher as thanks. The consultant asked not to be named.

**Model answers.** Files in `results/predictions/` are model outputs. Spans of 200 or more characters that overlap a bundle, and phone numbers absent from every bundle, are masked. They are not covered by the CC BY 4.0 license and remain subject to each model provider's terms of use.

**Known issues.** Some template-generated questions contain wording defects (for example a mismatched particle). The questions are kept as evaluated.

## License

Code: MIT (`LICENSE`). Data, annotations and verdicts: CC BY 4.0. The MOLIT trade rows in `data/tables/` follow the terms of their source (Ministry of Land, Infrastructure and Transport, through the Public Data Portal data.go.kr, with no restriction on use). Model answers: see the data card.

## Citation

See `CITATION.cff`. Version 1.0.0: https://doi.org/10.5281/zenodo.23429293

Earlier development release: KR-Housing-LongRAG-Bench (https://doi.org/10.5281/zenodo.20559127)
