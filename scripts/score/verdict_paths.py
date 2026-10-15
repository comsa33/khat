#!/usr/bin/env python3
"""Location of the verdict files read by score_judge.py.
"""
from __future__ import annotations

from pathlib import Path

from khat.paths import RESULTS, LOCAL
INTERNAL = LOCAL / "runs"
RELEASED = RESULTS / "verdicts" / "gpt-4.1-mini_original_gold"


def verdict_file(tag: str, verdicts_dir: str | Path | None = None) -> Path:
    """Path of the verdict file for `tag` (may not exist; callers check)."""
    dirs = [Path(verdicts_dir)] if verdicts_dir else [INTERNAL, RELEASED]
    for d in dirs:
        for name in (f"{tag}.judged.jsonl", f"{tag}.jsonl"):
            p = d / name
            if p.exists():
                return p
    return dirs[0] / f"{tag}.jsonl"


def human80_file(verdicts_dir: str | Path | None = None) -> Path:
    """Human Y/N labels of the n=80 judge-validation sample ({qa_id, human, ...})."""
    if verdicts_dir:
        return Path(verdicts_dir) / "human80_labels.jsonl"
    internal = INTERNAL / "panel_items_human80.jsonl"
    return internal if internal.exists() else RELEASED / "human80_labels.jsonl"
