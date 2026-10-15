"""Dataset facts from the released files: counts by split, family, provider, cluster, predicate, tier and scope; question rewriting; source manifest; released predictions.

Usage: python3 scripts/data/dataset_facts.py
"""
import json, csv, collections
from pathlib import Path
from khat.paths import ROOT as R
jl = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
ids = lambda p: {l.split("#")[0].strip() for l in open(p, encoding="utf-8") if l.split("#")[0].strip()}
legal = jl(R / "data/auxiliary/legal_reasoning.jsonl")
dev = jl(R / "data/qa_dev.jsonl") + [q for q in legal if q["split"] == "dev"]
test = jl(R / "data/qa_test_public.jsonl") + [q for q in legal if q["split"] == "test_public"]
allq = dev + test
C = collections.Counter
LEG = "cross_document_legal_reasoning"; AGG = "cross_source_aggregation"
print("total", len(allq), "dev", len(dev), "test", len(test))
print("announcements", len({a for q in allq for a in q.get("announcement_ids") or []}))
print("providers", len({q.get("provider") for q in allq if q.get("provider")}), sorted(C(q.get("provider") for q in allq).items(), key=lambda x: -x[1])[:12])
print("region_sido", len({q.get("region_sido") for q in allq if q.get("region_sido")}))
fam = C(q["task_type"] for q in allq); print("families", len(fam), dict(fam.most_common()))
cs = C(q["cluster_id"] for q in allq); print("clusters", len(cs), "groups(>1)", sum(1 for v in cs.values() if v > 1), "singletons", sum(1 for v in cs.values() if v == 1))
pred = [q for q in allq if q.get("gold_predicate")]
print("predicate items", len(pred), dict(C(q["gold_predicate"].get("source") for q in pred)), "by family", dict(C(q["task_type"] for q in pred)))
print("tiers", dict(C(q.get("context_tier") for q in allq)))
ex = ids(R / "data/exclusions/eval.txt"); bex = ids(R / "data/exclusions/bundle.txt")
t347 = [q for q in test if q["task_type"] != LEG]; t342 = [q for q in t347 if q["qa_id"] not in ex]
print("test non-legal", len(t347), "evaluated", len(t342), "test bundle-excluded", sum(q["qa_id"] in bex for q in t342))
f342 = C(q["task_type"] for q in t342); print("test evaluated by family", dict(f342.most_common()),
      "top3", sum(v for _, v in f342.most_common(3)))
d_scope = [q for q in dev if q["task_type"] != LEG]
print("dev non-legal", len(d_scope), "dev legal", sum(q["task_type"] == LEG for q in dev), "dev evaluated", sum(q["qa_id"] not in ex for q in d_scope))
agg_t = [q for q in t342 if q["task_type"] == AGG]; agg_d = [q for q in dev if q["task_type"] == AGG]
print("agg all", fam[AGG], "agg test", len(agg_t), "agg dev", len(agg_d), "agg dev after bundle exclusions + eval exclusions", sum(q["qa_id"] not in bex and q["qa_id"] not in ex for q in agg_d))
print("agg test ops", dict(C(q["gold_predicate"].get("op") for q in agg_t)), "announcements", len({q["announcement_ids"][0] for q in agg_t}),
      "tables", len({q["gold_predicate"].get("source") for q in agg_t}), "pairs", len({(q["announcement_ids"][0], q["gold_predicate"].get("source")) for q in agg_t}),
      "clusters", len({q["cluster_id"] for q in agg_t}), "bundles", dict(C(q.get("bundle_id") for q in agg_t)))
ans = [q for q in allq if q["task_type"] == "answerability_detection"]
print("answerability", len(ans), "with bundle", sum(bool(q.get("bundle_id")) for q in ans), "tiers", dict(C(q.get("context_tier") for q in ans if q.get("bundle_id"))))
print("format_robustness with bundle", sum(bool(q.get("bundle_id")) for q in allq if q["task_type"] == "format_robustness"))
print("test answerability evaluated", f342["answerability_detection"], "test answerability bundle-excluded", sum(q["qa_id"] in bex for q in t342 if q["task_type"] == "answerability_detection"))
print("test evaluated with a bundle and not bundle-excluded", sum(bool(q.get("bundle_id")) and q["qa_id"] not in bex for q in t342))
print("512k bundle items (all)", sum(q.get("bundle_id") == "mix_multiprovider_512k" for q in allq), "distinct 512k bundles", len({q["bundle_id"] for q in allq if q.get("context_tier") == "512k"}))
print("question rewritten (original_question != question)", sum(bool(q.get("original_question")) and q["original_question"] != q["question"] for q in allq))
reg = jl(R / "results/triage/regen_verify.jsonl") + jl(R / "results/triage/regen_verify_second_batch.jsonl")
okreg = {r["qa_id"] for r in reg if r["verdict"] == "OK"}; released = {q["qa_id"] for q in allq}
print("cloze regen verified", len(reg), "OK", len(okreg), "OK and released", len(okreg & released), "families", dict(C(q["task_type"] for q in allq if q["qa_id"] in okreg)))
tmpl = [q for q in allq if q["qa_id"] not in okreg and q.get("original_question") and q["original_question"] != q["question"]]
print("template-reworded (rewritten, not cloze)", len(tmpl), dict(C(q["task_type"] for q in tmpl)), "families", len(C(q["task_type"] for q in tmpl)))
print("neither", len(allq) - len(okreg & released) - len(tmpl))
man = jl(R / "data/source_manifest.jsonl"); print("source manifest entries", len(man), dict(C(m.get("source_type") for m in man)))
print("  with registry_url", sum(bool(m.get("registry_url")) for m in man), "with sha256 field", sum(any("sha" in k and m.get(k) for k in m) for m in man))
pman = [r for r in csv.DictReader(open(R / "results" / "MANIFEST.csv", encoding="utf-8"))
       if r["kind"] == "prediction" and r["file"].startswith("predictions/")]
for d, page in (("results/predictions", False), ("results/predictions k50_page", True)):
    rows = [r for r in pman if (r["setting"] == "k50_page") == page]
    print(d, "files", len(rows), "predictions", sum(int(r["n"]) for r in rows))
m5 = list((R / "results/predictions").rglob("*.jsonl")); nm = 0
for f in m5:
    for r in jl(f):
        nm += bool(r.get("answer_masked") or r.get("masked"))
print("masked predictions (answer_masked/masked flag) in results/predictions", nm)
print("sido over items of single-provider announcements (excl. '복수')", len({q.get("region_sido") for q in allq if q.get("region_sido") and q.get("provider") not in ("공공데이터(MOLIT/HUG)", "복수(비교)")} - {"복수"}))
print("source manifest content_sha256 non-empty", sum(bool(m.get("content_sha256")) for m in man))
om = sum(l.count("omitted") > 0 for f in (R / "results/predictions").rglob("*.jsonl") for l in open(f, encoding="utf-8"))
print("released answers with an 'omitted' mask marker", om)
import json as _j
for m in ("gpt-5.5", "gpt-5.4-mini"):
    fi = [_j.loads(l)["qa_id"] for l in open(R / f"results/judge_inputs/kimi-k2.6/full_context_{m}.jsonl")]
    fm = {q["qa_id"]: q["task_type"] for q in allq}
    print("full-context sample", m, len(fi), dict(C(fm[q] for q in fi).most_common()))
print("announcement providers (excl. public-data and comparison buckets)", len({q["provider"] for q in allq if q.get("provider") and "공공데이터" not in q["provider"] and not q["provider"].startswith("복수")}))
