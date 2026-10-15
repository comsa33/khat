"""Counts for the cost of further judging and runs: bundle token estimates, reported predictions per setting, full-context token sums and closed-book prompt tokens.

Usage: python3 scripts/analysis/cost_counts.py
"""
import json, csv, re, collections, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "score"))
from verdict_sets import *
from khat.paths import RESULTS, LOCAL
MAN={json.loads(l)["bundle_id"]:json.loads(l) for l in open(LOCAL/"bundles/manifest.jsonl")}
print("oversized token_estimate:", MAN.get("mix_multiprovider_256k",{}).get("token_estimate"), " mix 512k:", MAN.get("mix_multiprovider_512k",{}).get("token_estimate"))
tok=lambda q: MAN[G[q]["bundle_id"]]["token_estimate"]
TIERS=("32k","64k","128k","256k","512k")
S=list(csv.DictReader(open(RESULTS/"human_labels/judge_validation_200/sample.csv")))
nonA=[r for r in S if not r["model"].startswith("claude") and r["stratum"]=="main"]
agg81={r["qa_id"] for r in jl(R/"results/verdicts/panel/panel_kimi-k2.6_full_context_aggregation_gpt-5.5.jsonl")}
cov=[r for r in nonA if r["setting"] in ("fc",) and r["qa_id"] in agg81 and r["model"] in ("gpt-5.5","glm-5.2","minimax-m3","gpt-4.1-mini")]
print("200-sample main:", len([r for r in S if r['stratum']=='main']), "non-Anthropic:", len(nonA), "with panel verdict (FC on agg81, 4 models):", len(cov), collections.Counter(r["setting"] for r in nonA))
fc_n={m:len(items("fc",V["fc"][m])) for m in MODELS}
six=["gpt-4.1-mini","gpt-5.5","minimax-m3","glm-5.2","claude-sonnet-5-5","claude-haiku-5-5"]
rep={"cb":8*342,"bm25":8*332,"dense":8*332,"fc_sample":sum(fc_n.values()),"agg512k_extra(63/model)":6*63,"open-region":6*81,"oracle":6*81,"dense_k50":6*81,"k50_page":6*18}
print("reported predictions:", rep, "total", sum(rep.values()))
done_claude={"cb":2*342,"bm25":2*332,"dense":2*332,"fc_sample":2*96,"agg512k_extra(63/model)":2*63,"open-region":2*81,"oracle":2*81,"dense_k50":2*81,"k50_page":2*18}
done_non={"agg512k_extra(63/model)":4*63,"fc_sample":4*18}
need={k:rep[k]-done_claude.get(k,0)-done_non.get(k,0) for k in rep}
print("calls for one independent judge on everything not yet panel-scored:", need, "total", sum(need.values()))
def fcsum(m):
    qs=items("fc",V["fc"][m]); return len(qs), sum(tok(q) for q in qs), {t:(sum(1 for q in qs if G[q]['context_tier']==t), sum(tok(q) for q in qs if G[q]['context_tier']==t)) for t in TIERS}
for m in ["gpt-5.5","gpt-5.4-mini","gpt-5.4-nano","minimax-m3","glm-5.2"]:
    n,t,bt=fcsum(m); print("FC sample", m, n, "items, cl100k sum", t, bt)
S332=items("bm25",V["bm25"]["gpt-5.5"])
bt=collections.Counter(); bn=collections.Counter()
for q in S332: bt[G[q]["context_tier"]]+=tok(q); bn[G[q]["context_tier"]]+=1
print("332 by tier n:", dict(bn), "cl100k sums:", dict(bt), "total", sum(bt.values()))
over=[q for q in S332 if tok(q)>512000]; print("items with bundle > 512k cl100k:", len(over), [G[q]["bundle_id"] for q in over][:3])
le256=[q for q in S332 if G[q]["context_tier"]!="512k" and tok(q)<=272000]
print("fits 272k window (non-512k, <=272k):", len(le256), "sum", sum(tok(q) for q in le256))
ANN=re.compile(r"(?<!\d)\d{10}(?!\d)"); PG=re.compile(r"\(\s*p\.\s*\d+\s*\)")
ident=[q for q in TEST if ANN.search(G[q]["question"]) or PG.search(G[q]["question"])]
d=json.load(open(RESULTS/"runs/claude/cb_prompt_tokens.json"))
cbt=[d[f"claude-sonnet-5-5|cb__{q}"] for q in TEST]
cbi=[d[f"claude-sonnet-5-5|cb__{q}"] for q in ident]
print("CB items with identifier:", len(ident), "of", len(TEST), "; Claude-tokenizer CB prompt sum all", sum(cbt), "median", statistics.median(cbt), "; identifier items sum", sum(cbi))
