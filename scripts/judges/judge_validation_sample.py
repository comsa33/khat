"""Stratified random sample of reported test_public predictions for the 200-prediction judge validation.

Usage: python3 scripts/judges/judge_validation_sample.py [--exclude-human80] [--aux-string-mismatch] [--write OUT_DIR]
"""
import argparse
import csv
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

from khat.paths import ROOT
P = ROOT / "results" / "predictions"
V2, V3, V4 = "", "", ""

CB = {
    "gpt-4.1-mini": [V2 + "closed_book_gpt-4.1-mini", V3 + "closed_book_no_row_ids_gpt-4.1-mini_test_public"],
    "gpt-5.4-mini": [V2 + "closed_book_gpt-5.4-mini", V3 + "closed_book_no_row_ids_gpt-5.4-mini_test_public"],
    "gpt-5.4-nano": [V2 + "closed_book_gpt-5.4-nano", V3 + "closed_book_no_row_ids_gpt-5.4-nano_test_public"],
    "gpt-5.5": [V2 + "closed_book_gpt-5.5", V3 + "closed_book_second_call_gpt-5.5_test_public", V3 + "closed_book_no_row_ids_gpt-5.5_test_public"],
    "minimax-m3": [V2 + "closed_book_minimax-m3", V3 + "closed_book_no_row_ids_minimax-m3_test_public"],
    "glm-5.2": [V3 + "closed_book_glm-5.2"],
    "claude-sonnet-5-5": [V3 + "closed_book_claude-sonnet-5-5"],
    "claude-haiku-5-5": [V3 + "closed_book_claude-haiku-5-5"],
}
BM25 = {
    "gpt-4.1-mini": [V2 + "bm25_gpt-4.1-mini"], "gpt-5.4-mini": [V2 + "bm25_gpt-5.4-mini"],
    "gpt-5.4-nano": [V2 + "bm25_gpt-5.4-nano"], "gpt-5.5": [V2 + "bm25_gpt-5.5"],
    "minimax-m3": [V2 + "bm25_minimax-m3"], "glm-5.2": [V3 + "bm25_glm-5.2"],
    "claude-sonnet-5-5": [V3 + "bm25_claude-sonnet-5-5"], "claude-haiku-5-5": [V3 + "bm25_claude-haiku-5-5"],
}
DENSE = {m: [V2 + f"dense_{m}"] for m in ["gpt-4.1-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5", "minimax-m3"]}
DENSE.update({"glm-5.2": [V3 + "dense_glm-5.2"], "claude-sonnet-5-5": [V3 + "dense_claude-sonnet-5-5"],
              "claude-haiku-5-5": [V3 + "dense_claude-haiku-5-5"]})
FC = {
    "gpt-4.1-mini": [V2 + "full_context_gpt-4.1-mini", V2 + "full_context_512k_sample_gpt-4.1-mini"],
    "gpt-5.4-mini": [V2 + "full_context_gpt-5.4-mini"], "gpt-5.4-nano": [V2 + "full_context_gpt-5.4-nano"],
    "gpt-5.5": [V2 + "full_context_gpt-5.5", V2 + "full_context_512k_sample_gpt-5.5"],
    "minimax-m3": [V2 + "full_context_minimax-m3", V2 + "full_context_512k_sample_minimax-m3"],
    "glm-5.2": [V2 + "full_context_glm-5.2", V2 + "full_context_512k_sample_glm-5.2"],
    "claude-sonnet-5-5": [V3 + "full_context_claude-sonnet-5-5"], "claude-haiku-5-5": [V3 + "full_context_claude-haiku-5-5"],
}
FULL = ["gpt-5.5", "glm-5.2", "minimax-m3", "claude-sonnet-5-5", "claude-haiku-5-5", "gpt-4.1-mini"]
K50 = {m: [V4 + f"k50_{m}"] for m in FULL}
OPENREGION = {m: [V3 + f"open_region_{m}"] for m in FULL}
ORACLE = {m: [V3 + f"oracle_{m}"] for m in FULL}
SETTINGS = {"cb": CB, "bm25": BM25, "dense": DENSE, "fc": FC, "k50": K50, "open-region": OPENREGION, "oracle": ORACLE}
FC_ITEMS_FROM = V3 + "full_context_claude-sonnet-5-5"


report_aux = []
aux_families = Counter()


FIXES = None


def gold_strings(q, qa):
    """Gold answers as scored: accept-any alternatives, the replaced value, or the released answer."""
    global FIXES
    if FIXES is None:
        FIXES = json.load(open(ROOT / "data" / "errata" / "test_review_gold_fixes.json", encoding="utf-8"))
    if q in FIXES["accept_any"]:
        return list(FIXES["accept_any"][q])
    if q in FIXES["replace"]:
        return [FIXES["replace"][q]]
    return [qa[q]["answer"]]


def snorm(s):
    return "".join(unicodedata.normalize("NFKC", str(s)).lower().split())


def load(name):
    with open(P / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--exclude-human80", action="store_true")
    ap.add_argument("--write", metavar="OUT_DIR")
    ap.add_argument("--aux-string-mismatch", action="store_true")
    a = ap.parse_args()

    qa = {}
    with open(ROOT / "data" / "qa_test_public.jsonl", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            qa[r["qa_id"]] = r
    excl = {l.strip() for l in open(ROOT / "data" / "exclusions" / "eval_test_and_dev_sample.txt") if l.strip() and l[0] != "#"}
    scope = {q for q, r in qa.items() if r["task_type"] != "cross_document_legal_reasoning" and q not in excl}
    fc_items = {r["qa_id"] for r in load(FC_ITEMS_FROM)}
    agg = {json.loads(l)["qa_id"] for l in open(ROOT / "data" / "item_sets" / "agg81.jsonl")}

    h80 = set()
    if a.exclude_human80:
        for r in [json.loads(l) for l in open(ROOT / "results" / "human_labels" / "judge_validation_80.jsonl", encoding="utf-8")]:
            q, m, s = r["qa_id"].split("__")
            h80.add((q, m, {"rag": "bm25"}.get(s, s)))

    rng = random.Random(a.seed)
    rows, report, allpred = [], [], []
    for setting, by_model in SETTINGS.items():
        for model, files in by_model.items():
            pred = {}
            for fn in files:
                for r in load(fn):
                    if r.get("split", "test_public") == "test_public":
                        pred[r["qa_id"]] = r
            items = scope & set(pred)
            if setting == "fc":
                items &= fc_items
            if setting in ("k50", "open-region", "oracle"):
                items &= agg
            pool = sorted(q for q in items if (q, model, setting) not in h80)
            allpred += [(model, setting, q, pred[q]) for q in pool]
            dropped = len(items) - len(pool)
            pick = rng.sample(pool, min(a.per_cell, len(pool)))
            yes_pool = sum(bool(pred[q]["judge_label"]) for q in pool)
            yes_pick = sum(bool(pred[q]["judge_label"]) for q in pick)
            report.append((model, setting, len(items), dropped, len(pick), yes_pool, yes_pick))
            for q in pick:
                rows.append(make_row(q, model, setting, pred[q], qa, "main"))

    aux = []
    if a.aux_string_mismatch:
        golds = lambda q: gold_strings(q, qa)
        taken = {(r["qa_id"], r["model"], r["setting"]) for r in rows}
        for model in CB:
            cand = sorted((s, q) for m, s, q, r in allpred if m == model and (q, m, s) not in taken
                          and not any(snorm(g) in snorm(r["prediction"]) for g in golds(q)))
            pick = rng.sample(cand, min(2, len(cand)))
            for s, q in pick:
                r = next(r for m, s2, q2, r in allpred if m == model and s2 == s and q2 == q)
                aux.append(make_row(q, model, s, r, qa, "aux_string_mismatch"))
            yes = sum(bool(r["judge_label"]) for m, s, q, r in allpred if m == model and (s, q) in set(cand))
            report_aux.append((model, len(cand), len(pick), yes))
        aux_families.update(qa[q]["task_type"] for m, s, q, r in allpred
                            if (q, m, s) not in taken and not any(snorm(g) in snorm(r["prediction"]) for g in golds(q)))

    report_rows(a, report, rows, aux)
    if a.write:
        write(a, rows + aux)


def make_row(q, model, setting, r, qa, stratum):
    x = qa[q]
    ev = x.get("evaluation") or {}
    return {
        "stratum": stratum, "qa_id": q, "model": model, "setting": setting, "task_type": x["task_type"],
        "question": x["question"], "gold_answer": " ‖ ".join(gold_strings(q, qa)),
        "gold_alternatives": len(gold_strings(q, qa)),
        "gold_terms": json.dumps(ev.get("gold_terms") or ev.get("gold_numbers") or [], ensure_ascii=False),
        "metric": ev.get("metric", ""), "prediction": r["prediction"],
        "judge_label": r["judge_label"], "judge_label_secondary": r.get("judge_label_secondary"),
    }


def report_rows(a, report, rows, aux):
    print(f"seed {a.seed}, per cell {a.per_cell}, exclude_human80={a.exclude_human80}")
    print("| model | setting | pool | dropped (human80) | sampled | claude-sonnet-5-5 YES in pool | claude-sonnet-5-5 YES in sample |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for m, s, n, d, k, yp, ys in report:
        print(f"| {m} | {s} | {n} | {d} | {k} | {yp}/{n - d} | {ys}/{k} |")
    print(f"\ncells {len(report)}, sampled {len(rows)}, distinct items {len({r['qa_id'] for r in rows})}, "
          f"claude-sonnet-5-5 YES {sum(bool(r['judge_label']) for r in rows)}")
    print("by setting:", dict(Counter(r["setting"] for r in rows)))
    print("by task family:", dict(Counter(r["task_type"] for r in rows)))
    lit = sum(1 for r in rows if r["metric"] in ("exact_numbers", "contains_all"))
    print(f"metric types: {dict(Counter(r['metric'] for r in rows))} (string-scorable {lit})")
    print("\npost-stratification: cell weight = cell pool / sampled (the pool column above); "
          "it equals the setting weight except for the two 77-item FC cells")
    pop = Counter()
    for m, s, n, d, k, yp, ys in report:
        pop[s] += n - d
    print("  setting population:", dict(pop), "total", sum(pop.values()))
    if aux:
        print("\naux_string_mismatch stratum (judge labels for reference only; not used in selection):")
        print("| model | candidates | sampled | claude-sonnet-5-5 YES among candidates |")
        print("|---|---:|---:|---:|")
        for m, n, k, y in report_aux:
            print(f"| {m} | {n} | {k} | {y}/{n} |")
        print(f"candidates by family: {dict(aux_families)}")
        print(f"sampled: {len(aux)}; by setting {dict(Counter(r['setting'] for r in aux))}; "
              f"by family {dict(Counter(r['task_type'] for r in aux))}; "
              f"claude-sonnet-5-5 YES {sum(bool(r['judge_label']) for r in aux)}/{len(aux)}")


def write(a, rows):
    out = Path(a.write)
    out.mkdir(parents=True, exist_ok=True)
    order = list(range(len(rows)))
    random.Random(a.seed + 1).shuffle(order)
    human = {"label_annotator_a": "", "note_a": "", "label_annotator_b": "", "note_b": "", "label_adjudicated": ""}
    full = [dict(sample_id=f"s{i:03d}", **rows[j], **human) for i, j in enumerate(order)]
    with open(out / "sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(full[0]))
        w.writeheader()
        w.writerows(full)
    with open(out / "annotator.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "question", "gold_answer", "prediction", "label", "memo"])
        w.writeheader()
        for r in full:
            w.writerow({"sample_id": r["sample_id"], "question": r["question"], "gold_answer": r["gold_answer"],
                        "prediction": r["prediction"], "label": "", "memo": ""})
    with open(out / "key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "stratum", "qa_id", "model", "setting"], extrasaction="ignore")
        w.writeheader()
        w.writerows(full)
    print(f"wrote {out}/sample.csv, annotator.csv, key.csv")


if __name__ == "__main__":
    main()
