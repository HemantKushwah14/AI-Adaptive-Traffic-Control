"""Render every dashboard page headlessly with Streamlit's AppTest and report exceptions.
    python scripts\\check_all_pages.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

from dashboard.state import PAGES, nav_label  # noqa: E402


def main() -> int:
    bad = 0
    for title, _icon, _module in PAGES:
        try:
            at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
            at.session_state["nav"] = nav_label(title)
            at.run()
            errors = [e.value for e in at.exception]
            alerts = [m.value for m in at.markdown if "This page failed to render" in m.value]
            if errors or alerts:
                bad += 1
                print(f"[FAIL] {title}")
                for e in errors:
                    print("       ", str(e)[:300])
            else:
                print(f"[ OK ] {title}  ({len(at.markdown)} markdown blocks, {len(at.button)} buttons)")
        except Exception as exc:
            bad += 1
            print(f"[FAIL] {title}: {type(exc).__name__}: {exc}")
    print("\nAll 7 pages render without exceptions." if not bad else f"\n{bad} page(s) failed.")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())