"""How often the answer string of a test long-context retrieval item occurs on more than one page of its announcement (reads the extracted page text under local/).

Usage: python3 scripts/analysis/answer_repeat_pages.py
"""
import json, re, glob, collections
from khat.paths import DATA, LOCAL
pages = {}
for f in glob.glob(str(LOCAL / '*-announcements*/*/document_pages.jsonl')):
    for l in open(f):
        r = json.loads(l); pages[r['page_id']] = r.get('text') or ''
if not pages:
    raise SystemExit(f"no extracted announcement pages under {LOCAL}; rebuild them with scripts/build/")
norm = lambda s: re.sub(r'\s+', '', s)
ann = lambda pid: re.sub(r'-p\d+$', '', pid)
byann = collections.defaultdict(list)
for pid, t in pages.items(): byann[ann(pid)].append((pid, norm(t)))
ex = {l.strip() for l in open(DATA / 'exclusions/eval_test_and_dev_sample.txt') if l.strip() and l[0] != '#'}
res = collections.Counter(); multi = []
for l in open(DATA / 'qa_test_public.jsonl'):
    x = json.loads(l)
    if x['task_type'] != 'long_context_retrieval' or x['qa_id'] in ex: continue
    g = x['page_ids'][0]; a = norm(str(x['answer']))
    pat = re.compile(r'(?<![0-9])' + re.escape(a) + r'(?![0-9])')
    occ = [pid for pid, t in byann[ann(g)] if pat.search(t)]
    tot = sum(len(pat.findall(t)) for pid, t in byann[ann(g)])
    k = 'once' if tot == 1 else ('same page only' if set(occ) == {g} else ('several pages' if len(occ) > 1 else ('not found verbatim' if not occ else 'other')))
    res[k] += 1
    if k == 'several pages': multi.append(x['qa_id'])
print(sum(res.values()), dict(res)); print(multi)
