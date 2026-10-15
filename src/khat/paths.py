"""Repository paths. LOCAL holds files rebuilt on the user's machine (sources, bundles, table rows, run files).

LOCAL is taken from $KHAT_LOCAL, else from `local` in configs/paths.toml, else ROOT/local.
"""
import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
_CFG = ROOT / "configs" / "paths.toml"
_PATHS = tomllib.loads(_CFG.read_text(encoding="utf-8")) if _CFG.exists() else {}
LOCAL = Path(os.environ.get("KHAT_LOCAL") or _PATHS.get("local") or ROOT / "local").expanduser()
