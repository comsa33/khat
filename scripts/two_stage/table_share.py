"""Share of a table's passages that the two-stage prompt contains.

Usage: python3 scripts/two_stage/table_share.py <two-stage run dir>
"""
import json, re, sys, collections, statistics as st
from pathlib import Path
from khat.paths import DATA, LOCAL
W = Path(sys.argv[1])
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import retrieval as RS
ch = RS.split_chunks((LOCAL / "bundles/mix_multiprovider_512k.txt").read_text(encoding="utf-8"), 1200)
HUG = re.compile(r"^[가-힣]+ \| 20\d\d \| .+ \| \d+ \| \d+$"); MOL = re.compile(r"^MOLIT-APTTRADE-")
def tab(i):
    if ch[i][0] != "preamble-p000": return None
    ls = [l for l in ch[i][1].splitlines() if l.strip()]
    for name, rx in (("hug", HUG), ("molit", MOL)):
        if sum(bool(rx.match(l)) for l in ls) * 2 >= len(ls): return name
TOT = collections.Counter(tab(i) for i in range(len(ch)))
HDR = {"hug": {i for i in range(len(ch)) if "HUG 분양이력 자료" in ch[i][1]}, "molit": {i for i in range(len(ch)) if "행ID | 시군구" in ch[i][1]}}
Q = {}
for f in ("qa_test_public.jsonl", "qa_dev.jsonl"):
    for l in open(DATA / f, encoding="utf-8"):
        r = json.loads(l); Q[r["qa_id"]] = r["question"]
ex = [json.loads(l) for l in open(W / "extract.jsonl")]; P = {json.loads(l)["qa_id"]: len(json.loads(l)["prompt"]) for l in open(LOCAL / "runs/two_stage/prompts_twostage.jsonl")}
print("table passages in the bundle:", {k: v for k, v in TOT.items() if k}, "| header passages:", {k: sorted(v) for k, v in HDR.items()},
      "| header passages typed as table:", {k: [tab(i) for i in v] for k, v in HDR.items()})
for split in ("test", "dev"):
    rows = [r for r in ex if r["split"] == split]; share = collections.Counter(); hdr = 0
    for r in rows:
        t = "molit" if ("MOLIT" in Q[r["qa_id"]] or "실거래" in Q[r["qa_id"]]) else "hug"
        share[(t, sum(tab(i) == t for i in r["stage2_ids"]))] += 1; hdr += bool(HDR[t] & (set(r["stage2_ids"]) | set(r["stage1_ids"])))
    c = [P[r["qa_id"]] for r in rows]
    print(f"{split} {len(rows)}: (table, its passages among the 20) -> items {dict(sorted(share.items()))}; header passage of the asked table in the prompt {hdr}/{len(rows)}; "
          f"prompt chars median {st.median(c):.0f} min {min(c)} max {max(c)}; o200k tokens median {0.555*st.median(c)/1000:.1f}k max {0.555*max(c)/1000:.1f}k")
