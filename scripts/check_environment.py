"""Verify the local installation. Run:  python scripts\\check_environment.py"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import SUMO_SETUP_HELP, SumoNotFoundError, ensure_traci_importable, find_sumo_binary  # noqa: E402

OK, FAIL = "[ OK ]", "[FAIL]"
failures = 0


def report(ok: bool, msg: str) -> None:
    global failures
    if not ok:
        failures += 1
    print(f"{OK if ok else FAIL} {msg}")


def main() -> int:
    v = sys.version_info
    report((v.major, v.minor) == (3, 11), f"Python {v.major}.{v.minor}.{v.micro} (3.11 required)")

    for pkg in ("numpy", "pandas", "cv2", "torch", "ultralytics", "streamlit", "plotly", "yaml", "scipy"):
        try:
            m = importlib.import_module(pkg)
            report(True, f"{pkg} {getattr(m, '__version__', '')}")
        except Exception as exc:
            report(False, f"{pkg}: {exc}")

    try:
        import torch
        print(f"       torch CUDA available: {torch.cuda.is_available()} (CPU is fine locally)")
    except Exception:
        pass

    try:
        ensure_traci_importable()
        import traci
        report(True, f"traci importable ({getattr(traci, '__file__', '')})")
    except SumoNotFoundError as exc:
        report(False, str(exc))

    for binary in ("sumo", "netconvert"):
        try:
            path = find_sumo_binary(binary)
            out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=30)
            first = (out.stdout or out.stderr).strip().splitlines()[0]
            report(True, f"{binary}: {first}")
        except SumoNotFoundError:
            report(False, f"{binary} not found")
            print(SUMO_SETUP_HELP)
            break
        except Exception as exc:
            report(False, f"{binary} failed to run: {exc}")

    print("\nAll checks passed." if failures == 0 else f"\n{failures} check(s) failed.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())