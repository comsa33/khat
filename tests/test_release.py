"""Release checks: keys, bundle text, data layout and released files.

Usage: uv run pytest tests/test_release.py
"""
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = {".py", ".md", ".toml", ".json", ".jsonl", ".csv", ".tsv", ".tex", ".html", ".ipynb", ".txt", ".yaml", ".yml", ".cff", ".sh", ".cfg", ""}
KEYS = re.compile(r"\bhf_[A-Za-z0-9]{20,}|\bsk-[A-Za-z0-9_-]{20,}|\bsk-ant-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----")
MOLIT_SIX = ["krhlrb_v05_0425", "krhlrb_v05_0426", "krhlrb_v05_0428", "krhlrb_v05_0429", "krhlrb_v05_0431", "krhlrb_v05_0432"]
LEGAL = "cross_document_legal_reasoning"


def tracked():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode == 0 and out.stdout:
        return [ROOT / p for p in out.stdout.split("\0") if p]
    return [p for p in ROOT.rglob("*") if p.is_file() and not {".git", ".venv", "local", "__pycache__"} & set(p.relative_to(ROOT).parts)]


def text_files():
    return [p for p in tracked() if p.suffix in TEXT and p.name != "uv.lock"]


def jl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


def test_no_keys():
    hits = [str(p.relative_to(ROOT)) for p in text_files() if KEYS.search(p.read_text(encoding="utf-8", errors="replace"))]
    assert not hits, hits


def test_no_bundle_files():
    path = ROOT / "data" / "bundle_sha256.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    bundle = {r[k] for r in rows for k in ("sha256", "sha256_default") if r.get(k)}
    assert len(rows) == 110
    hits = [str(p.relative_to(ROOT)) for p in tracked() if hashlib.sha256(p.read_bytes()).hexdigest() in bundle]
    assert not hits, hits


def test_default_files_exclude_legal():
    dev, test = jl(ROOT / "data" / "qa_dev.jsonl"), jl(ROOT / "data" / "qa_test_public.jsonl")
    legal = jl(ROOT / "data" / "auxiliary" / "legal_reasoning.jsonl")
    assert (len(dev), len(test), len(legal)) == (1541, 347, 109)
    assert not any(r["task_type"] == LEGAL for r in dev + test)
    assert all(r["task_type"] == LEGAL for r in legal)


def test_molit_items_keep_original_answers():
    qa = {r["qa_id"]: r for r in jl(ROOT / "data" / "qa_test_public.jsonl")}
    errata = {r["qa_id"]: r for r in csv.DictReader(open(ROOT / "data" / "errata" / "cancelled_transactions.csv", encoding="utf-8"))}
    assert sorted(errata) == MOLIT_SIX
    for q in MOLIT_SIX:
        assert qa[q]["answer"] == errata[q]["old_answer"], q


def test_released_files():
    need = ["data/gold_fixes.json", "data/source_manifest.jsonl", "data/cluster_map.jsonl", "data/errata/test_review.csv",
            "data/errata/dev_review.csv", "data/auxiliary/legal_errata.csv", "data/exclusions/eval.txt", "data/exclusions/bundle.txt",
            "results/human_labels/judge_validation_80.jsonl", "results/human_labels/judge_validation_200",
            "results/human_labels/judge_error_review", "results/verdicts/kimi-k2.6", "results/verdicts/claude-sonnet-5-5",
            "results/verdicts/gpt-4.1-mini", "results/verdicts/panel", "results/predictions", "results/triage", "prompts/construction"]
    missing = [n for n in need if not (ROOT / n).exists()]
    assert not missing, missing
