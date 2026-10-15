"""MOLIT count and average items with and without cancelled transactions.
"""
import statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "judges"))
from panel_verdicts import C
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import table_rows as q3
G, TEST = C.G, C.TEST
mol = sorted(q for q in G if (G[q].get("gold_predicate") or {}).get("source", "").startswith("molit"))
print("MOLIT items:", len(mol), "| split", {s: sum(1 for q in mol if G[q]["split"] == s) for s in {G[q]["split"] for q in mol}},
      "| in the 342 evaluated:", sum(q in TEST for q in mol))
print("cdealType values over all MOLIT rows used:", sorted({str(r.get("cdealType", "")) for q in mol for r in q3.select(G[q]["gold_predicate"])}))
for q in mol:
    p = G[q]["gold_predicate"]; rows = q3.select(p); val, _ = q3.recompute(p)
    keep = [r for r in rows if str(r.get("cdealType", "")).strip() != "O"]
    if p["op"] == "count":
        ex = len(keep)
    else:
        nums = [v for v in (q3._num(p["source"], r, p["field"]) for r in keep) if v is not None]
        ex = {"avg": lambda: int(round(statistics.mean(nums), p.get("round", 0))), "max": lambda: max(nums),
              "min": lambda: min(nums), "sum": lambda: sum(nums)}[p["op"]]()
    print(f"{q} op={p['op']:<5} filter={p['filter']} | gold {G[q]['answer']} | value {val} | rows {len(rows)}, cancelled {len(rows)-len(keep)} | "
          f"excluding cancelled: {ex}{' (unchanged)' if ex == val else ''}")
