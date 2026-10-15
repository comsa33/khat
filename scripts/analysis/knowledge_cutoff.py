#!/usr/bin/env python3
"""Closed-book accuracy before and after each model's knowledge cutoff.
"""
import calendar
import datetime as dt
import sys
from pathlib import Path
from khat.paths import DATA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "score"))
from verdict_sets import MODELS, V, wilson, jl

CUTOFF = {"gpt-5.5": "2025-12-01", "gpt-5.4-mini": "2025-08-31", "gpt-5.4-nano": "2025-08-31",
          "gpt-4.1-mini": "2024-06-01", "claude-sonnet-5-5": "2026-06", "claude-haiku-5-5": "2026-06",
          "minimax-m3": None, "glm-5.2": None}
ANN_DATE = {"lh-daedong2-b1-public-sale-20251017": "2025-10-17",
            "lh-announce-0000061016-public-sale-2025q4": "2025-11",
            "lh-siheung-hajung-a1-newlywed-hope-20260430": "2026-04-30",
            "lh-announce-0000061086-public-sale-2026q2": "2026-05",
            "cbdc-cbdc-2025-jecheon-yeongcheon-happy-notice": "2025-07-03",
            "dgdc-dgdc-2025-11-28-youth-purchase-rental-notice": "2025-11-28",
            "ih-ih-2025-cheonwon-jeonse-notice": "2025-04-30",
            "jpdc-jpdc-2025-04-29-ildo-notice": "2025-04-29",
            "jpdc-jpdc-2025-08-25-beophwan-notice": "2025-08-25"}


def span(s):
    y, m, *d = (int(x) for x in s.split("-"))
    if d:
        return dt.date(y, m, d[0]), dt.date(y, m, d[0])
    return dt.date(y, m, 1), dt.date(y, m, calendar.monthrange(y, m)[1])


def side(ann, cut):
    if cut is None:
        return "not published"
    c = span(cut)[1]
    lo, hi = span(ANN_DATE[ann])
    if hi <= c:
        return "before"
    if lo > c:
        return "after"
    return "straddles"


CLS = [r for r in jl(DATA / "analysis/answer_classes.jsonl") if r["S_prime"]]
assert len(CLS) == 82
print("S' items per announcement and position relative to each published cutoff:")
anns = sorted({a for r in CLS for a in r["announcement_ids"]}, key=lambda a: span(ANN_DATE[a])[0])
print("announcement\tdate\tn\t" + "\t".join(MODELS))
for a in anns:
    n = sum(a in r["announcement_ids"] for r in CLS)
    print(f"{a}\t{ANN_DATE[a]}\t{n}\t" + "\t".join(side(a, CUTOFF[m]) for m in MODELS))
print("\nS' accuracy by position relative to the model's own cutoff (claude-sonnet-5-5 judge, Wilson 95%):")
for m in MODELS:
    if CUTOFF[m] is None:
        print(f"  {m}: cutoff not published")
        continue
    parts = []
    for s in ("before", "after"):
        qs = [r["qa_id"] for r in CLS if all(side(a, CUTOFF[m]) == s for a in r["announcement_ids"])]
        if qs:
            c = sum(V["cb"][m][q] for q in qs)
            lo, hi = wilson(c, len(qs))
            parts.append(f"{s} {c}/{len(qs)} = {100 * c / len(qs):.1f}% [{lo:.1f}, {hi:.1f}]")
        else:
            parts.append(f"{s} 0 items")
    print(f"  {m} (cutoff {CUTOFF[m]}): " + "; ".join(parts))
