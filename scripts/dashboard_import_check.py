"""Import every dashboard module headlessly to catch missing files / typos / bad imports
before you open the browser.   python scripts\\dashboard_import_check.py"""
from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from dashboard.state import PAGES  # noqa: E402

MODULES = [m for _, _, m in PAGES] + ["dashboard.jobs", "dashboard.session", "dashboard.sim_settings",
                                      "dashboard.visualizations.intersection", "dashboard.visualizations.charts",
                                      "dashboard.components.status"]


def main() -> int:
    bad = 0
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
            if name.startswith("dashboard.pages") and not callable(getattr(mod, "render", None)):
                raise AttributeError("page has no render()")
            print(f"[ OK ] {name}")
        except Exception as exc:
            bad += 1
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")
    print("\nAll dashboard modules import." if not bad else f"\n{bad} module(s) failed.")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())