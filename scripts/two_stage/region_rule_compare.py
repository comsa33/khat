"""Per-item comparison of the region rules and periods on dev' and test.

Usage: python3 scripts/two_stage/region_rule_compare.py <two-stage run dir>
"""
import sys, re, runpy, contextlib, io
from pathlib import Path
with contextlib.redirect_stdout(io.StringIO()):
    M = runpy.run_path(str(Path(__file__).resolve().parent / "region_rule.py"), run_name="region_rule")
G, EX, v, DP, D144, T81 = M["G"], M["EX"], M["v"], M["DP"], M["D144"], M["T81"]
def per_dry(q):
    m = re.search(r"(20\d\d)년(?:\s*(\d{1,2})월)?", G[q]["question"])
    return (m.group(1) + "년" + (f" {int(m.group(2))}월" if m.group(2) else "")) if m else ""
for name, qs in (("dev' 81", DP), ("dev 144", D144), ("test 81", T81)):
    diff_r = [(q, v(q, False), v(q, True), EX[q]["region"]) for q in qs if not (v(q, False) == v(q, True) == EX[q]["region"])]
    diff_p = [(q, per_dry(q), EX[q]["period"]) for q in qs if per_dry(q) != EX[q]["period"]]
    print(f"{name}: region values differ among v1/v2/v3 on {len(diff_r)} items; dry-run vs adopted period differ on {len(diff_p)}")
    for d in diff_r[:12]: print("   region", d)
    for d in diff_p[:6]: print("   period", d)
