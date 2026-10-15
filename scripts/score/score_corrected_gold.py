#!/usr/bin/env python3
"""Scores on the 24 gold-corrected and 5 excluded test_public items per run, original gold vs corrected gold.

Usage: python3 scripts/score/score_corrected_gold.py
"""
import json, os, sys
from pathlib import Path
from khat.paths import ROOT
W = str(ROOT)
G=json.load(open(f'{W}/data/errata/test_review_gold_fixes.json'))
FIX=list(G['accept_any'])+list(G['replace']); EXC=list(G['exclude_from_evaluation'])
assert len(FIX)==24 and len(EXC)==5
CD=f'{W}/results/verdicts/claude-sonnet-5-5'; GD=f'{W}/results/verdicts/gpt-4.1-mini'; OD=f'{W}/results/verdicts/gpt-4.1-mini_original_gold'
tags=['closed_book_no_reasoning_glm-5.2', 'closed_book_minimax-m3', 'full_context_glm-5.2', 'full_context_minimax-m3', 'full_context_gpt-5.5', 'closed_book_gpt-4.1-mini', 'full_context_gpt-4.1-mini', 'bm25_gpt-4.1-mini', 'closed_book_gpt-5.4-mini', 'full_context_gpt-5.4-mini', 'bm25_gpt-5.4-mini', 'closed_book_gpt-5.4-nano', 'full_context_gpt-5.4-nano', 'bm25_gpt-5.4-nano', 'closed_book_gpt-5.5', 'bm25_gpt-5.5', 'dense_no_reasoning_glm-5.2', 'dense_gpt-4.1-mini', 'dense_gpt-5.4-mini', 'dense_gpt-5.4-nano', 'dense_gpt-5.5', 'dense_minimax-m3', 'bm25_no_reasoning_glm-5.2', 'bm25_minimax-m3']
assert len(tags)==24, tags
def load(p):
    d={}
    for l in open(p):
        r=json.loads(l); d[r['qa_id']]=r['judge_correct']
    return d
def mr(t):
    for p,r in (('closed_book_no_reasoning_','cb'),('bm25_no_reasoning_','bm25'),('dense_no_reasoning_','dense'),
                ('closed_book_','cb'),('full_context_','fc'),('bm25_','bm25'),('dense_','dense')):
        if t.startswith(p): return t[len(p):],r
tp={}
for l in open(f'{W}/data/qa_test_public.jsonl'):
    r=json.loads(l); tp[r['qa_id']]=r
rows=[]; per256=[]
for t in tags:
    mo,rg=mr(t); c=load(f'{CD}/{t}.jsonl'); g=load(f'{GD}/{t}.jsonl'); o=load(f'{OD}/{t}.jsonl')
    fx=[q for q in FIX if q in c]; missing_o=[q for q in fx if q not in o]
    ex=[q for q in EXC if q in o]
    rows.append(dict(model=mo,regime=rg,tag=t,n_fix=len(fx),orig=sum(o.get(q,False) for q in fx),cl=sum(c[q] for q in fx),gp=sum(g[q] for q in fx if q in g),
                     n_exc=len(ex),exc_orig=sum(o[q] for q in ex),miss_orig=missing_o, exc_in_claude=[q for q in EXC if q in c]))
    if rg=='fc':
        for q in FIX+EXC:
            if tp[q].get('context_tier')=='256k' and (q in c or q in o):
                per256.append((mo,q,tp[q]['task_type'],'corrected' if q in FIX else 'excluded',o.get(q),c.get(q),g.get(q)))
order=['gpt-4.1-mini','gpt-5.4-mini','gpt-5.4-nano','gpt-5.5','minimax-m3','glm-5.2']
rorder=['cb','bm25','dense','fc']
rows.sort(key=lambda r:(order.index(r['model']),rorder.index(r['regime'])))
out=['| Model | Regime | tag | n corrected | original gold, gpt | corrected gold, Claude | corrected gold, gpt | n excl | excl: original gold, gpt correct |','|---|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    out.append(f"| {r['model']} | {r['regime']} | {r['tag']} | {r['n_fix']} | {r['orig']} | {r['cl']} | {r['gp']} | {r['n_exc']} | {r['exc_orig']} |")
    if r['miss_orig']: out.append(f"|  | note: missing in original-gold verdicts: {r['miss_orig']} |")
tot=lambda k:sum(r[k] for r in rows)
out.append(f"| **Total** | | | {tot('n_fix')} | {tot('orig')} | {tot('cl')} | {tot('gp')} | {tot('n_exc')} | {tot('exc_orig')} |")
out.append('')
out.append('256k-tier items among the 29, fc regime (✓ correct / ✗ wrong / — not in run):')
out.append('')
out.append('| Model | qa_id | task_type | status | original gold, gpt | corrected gold, Claude | corrected gold, gpt |'); out.append('|---|---|---|---|---|---|---|')
f=lambda v:'—' if v is None else ('✓' if v else '✗')
for p in sorted(per256,key=lambda x:(order.index(x[0]),x[1])):
    out.append(f"| {p[0]} | {p[1][-4:]} | {p[2]} | {p[3]} | {f(p[4])} | {f(p[5])} | {f(p[6])} |")
print('\n'.join(out))
print('\nexcluded ids present in Claude verdict files (should be none):', {r['tag']:r['exc_in_claude'] for r in rows if r['exc_in_claude']})
print('256k ids among 29:', [q for q in FIX+EXC if tp[q].get('context_tier')=='256k'])
