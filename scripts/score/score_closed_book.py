"""Closed-book accuracy before and after the re-run without locator identifiers, on all items and on the 18 re-run items.

Usage: python3 scripts/score/score_closed_book.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "score"))
from verdict_sets import *
P="verdicts/claude-sonnet-5-5/"
def ld(t): return {r["qa_id"]:bool(r["judge_correct"]) for r in jl(R/"results"/(t+".jsonl"))}
before={"gpt-4.1-mini":[P+"closed_book_gpt-4.1-mini"],"gpt-5.4-mini":[P+"closed_book_gpt-5.4-mini"],"gpt-5.4-nano":[P+"closed_book_gpt-5.4-nano"],
 "gpt-5.5":[P+"closed_book_gpt-5.5"],"minimax-m3":[P+"closed_book_minimax-m3"],"glm-5.2":[P+"closed_book_no_reasoning_glm-5.2"]}
fixf={"gpt-4.1-mini":P+"closed_book_no_row_ids_gpt-4.1-mini_test_public","gpt-5.4-mini":P+"closed_book_no_row_ids_gpt-5.4-mini_test_public","gpt-5.4-nano":P+"closed_book_no_row_ids_gpt-5.4-nano_test_public",
 "gpt-5.5":P+"closed_book_no_row_ids_gpt-5.5_test_public","minimax-m3":P+"closed_book_no_row_ids_minimax-m3_test_public","glm-5.2":P+"closed_book_no_row_ids_glm-5.2_test_public"}
f18=set(ld(fixf["gpt-4.1-mini"]))&set(TEST)
for m,fs in before.items():
    b={}; [b.update(ld(f)) for f in fs]
    qs=[q for q in TEST if q in b]
    pb,cb_=acc(b,qs)
    if m=="glm-5.2": a={**b, **ld(fixf[m])}
    else: a=V["cb"][m]
    pa,ca=acc(a,qs)
    print(m,"n",len(qs),"before %.1f/%.1f"%(pb,cb_),"after %.1f/%.1f"%(pa,ca),"18 before",sum(b[q] for q in f18),"after",sum(a[q] for q in f18))
