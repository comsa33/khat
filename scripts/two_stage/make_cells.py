"""Dev items that share no table cell with a test aggregation item (writes data/item_sets/dev_cells.tsv).
"""
import json, hashlib
from pathlib import Path
from khat.paths import ROOT, DATA
HERE = Path(__file__).resolve().parent
DS = ROOT
G = {}
for f in ("qa_test_public.jsonl", "qa_dev.jsonl"):
    for l in open(DS / "data" / f, encoding="utf-8"):
        r = json.loads(l); G[r["qa_id"]] = r
ids = lambda f: [l.strip() for l in open(DATA / "item_sets" / f) if l.strip()]
cell = lambda q: "{}|{}|{}".format(G[q]["gold_predicate"]["source"], json.dumps(G[q]["gold_predicate"]["filter"], sort_keys=True, ensure_ascii=False), G[q]["gold_predicate"]["op"])
TC = {cell(q) for q in ids("items_test81.txt")}
rows = [(q, cell(q), int(cell(q) in TC)) for q in ids("items_dev144.txt")]
with open(DATA / "item_sets/dev_cells.tsv", "w", encoding="utf-8") as f:
    f.write("qa_id\tcell\tsame_cell_as_test\n")
    for r in rows: f.write("\t".join(map(str, r)) + "\n")
print(f"dev items {len(rows)}; cells {len({c for _, c, _ in rows})}; items sharing a cell with test {sum(o for *_, o in rows)}; "
      f"non-overlap items {sum(1 - o for *_, o in rows)}; test cells {len(TC)}")
print("dev_cells.tsv", hashlib.sha256((DATA / "item_sets/dev_cells.tsv").read_bytes()).hexdigest())
