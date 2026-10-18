"""One-command verification of the whole project. It NEVER trains and NEVER overwrites a checkpoint
or a saved report (DQN-dependent steps are skipped when no checkpoint exists).

    python scripts\\run_all_checks.py             # default: everything except slow model tests
    python scripts\\run_all_checks.py --quick     # no SUMO simulations, ~1 minute
    python scripts\\run_all_checks.py --full      # + DQN session, comparison and emergency demo (needs a checkpoint)

A JSON report is written to outputs\\all_checks_report.json (every line is a measured pass/fail).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.rl_config import BEST_CKPT, LAST_CKPT  # noqa: E402
from config.settings import OUTPUTS_DIR, ensure_traci_importable, find_sumo_binary  # noqa: E402

PY = sys.executable


@dataclass
class Step:
    name: str
    args: List[str]
    modes: Tuple[str, ...] = ("quick", "default", "full")
    needs_sumo: bool = False
    needs_checkpoint: bool = False
    timeout: int = 900


def sumo_available() -> bool:
    try:
        find_sumo_binary("sumo")
        find_sumo_binary("netconvert")
        ensure_traci_importable()
        return True
    except Exception:
        return False


def build_steps(mode: str) -> List[Step]:
    pytest_filter = {"quick": ["-m", "not integration and not slow"], "default": ["-m", "not slow"], "full": []}[mode]
    steps = [
        Step("Environment check", ["scripts/check_environment.py"]),
        Step("Build SUMO network", ["-m", "simulation.build_network"], needs_sumo=True),
        Step("DQN learning-rule self-test", ["-m", "reinforcement_learning.dqn_agent", "--selftest"]),
        Step("Adaptive controller self-test", ["-m", "simulation.adaptive_controller", "--selftest"]),
        Step("Emergency controller self-test", ["-m", "simulation.emergency_controller", "--selftest"]),
        Step("pytest suite", ["-m", "pytest", "-q", *pytest_filter], timeout=1800),
        Step("Dashboard import check", ["scripts/dashboard_import_check.py"]),
        Step("Fixed-time simulation (120 s)", ["scripts/run_fixed_time.py", "--duration", "120"],
             modes=("default", "full"), needs_sumo=True),
        Step("Dashboard session smoke test (fixed-time)", ["scripts/session_smoke_test.py", "--duration", "120"],
             modes=("default", "full"), needs_sumo=True),
        Step("Dashboard session smoke test (DQN)",
             ["scripts/session_smoke_test.py", "--controller", "dqn", "--duration", "120"],
             modes=("full",), needs_sumo=True, needs_checkpoint=True),
        Step("DQN vs fixed-time, 1 seed (not saved)",
             ["scripts/run_comparison.py", "--seeds", "1", "--duration", "300", "--no-save"],
             modes=("full",), needs_sumo=True, needs_checkpoint=True),
        Step("Emergency demo (not saved)", ["scripts/run_emergency_demo.py", "--no-save"],
             modes=("full",), needs_sumo=True),
    ]
    return [s for s in steps if mode in s.modes]


def run_step(step: Step) -> Tuple[str, float, str]:
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    t0 = time.perf_counter()
    try:
        p = subprocess.run([PY, *step.args], cwd=ROOT, capture_output=True, text=True, timeout=step.timeout, env=env)
    except subprocess.TimeoutExpired:
        return "FAIL", time.perf_counter() - t0, f"timeout after {step.timeout}s"
    out = (p.stdout or "") + (p.stderr or "")
    tail = "\n".join(out.strip().splitlines()[-30:])
    return ("PASS" if p.returncode == 0 else "FAIL"), time.perf_counter() - t0, tail


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all project checks.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    a = ap.parse_args()
    mode = "quick" if a.quick else "full" if a.full else "default"
    steps = build_steps(mode)
    sumo_ok = sumo_available()
    has_ckpt = BEST_CKPT.exists() or LAST_CKPT.exists()
    print(f"Mode: {mode} | SUMO available: {sumo_ok} | DQN checkpoint present: {has_ckpt}\n")

    results = []
    for i, st in enumerate(steps, 1):
        prefix = f"[{i:2d}/{len(steps)}] {st.name:<46}"
        if st.needs_sumo and not sumo_ok:
            status, secs, detail = "SKIP", 0.0, "SUMO not available"
        elif st.needs_checkpoint and not has_ckpt:
            status, secs, detail = "SKIP", 0.0, "no DQN checkpoint (this script never trains one)"
        else:
            print(prefix + " running ...", end="\r", flush=True)
            status, secs, detail = run_step(st)
        print(f"{prefix} {status:<5} {secs:7.1f}s" + (f"  ({detail})" if status == "SKIP" else ""))
        if status == "FAIL":
            print("      --- last output lines ---")
            print("\n".join("      " + ln for ln in detail.splitlines()))
        results.append({"step": st.name, "status": status, "seconds": round(secs, 1),
                        "detail": detail if status != "PASS" else ""})

    fails = [r for r in results if r["status"] == "FAIL"]
    skips = [r for r in results if r["status"] == "SKIP"]
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report = OUTPUTS_DIR / "all_checks_report.json"
    report.write_text(json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"), "mode": mode,
                                  "sumo_available": sumo_ok, "checkpoint_present": has_ckpt, "results": results},
                                 indent=2), encoding="utf-8")
    print(f"\n{len(results) - len(fails) - len(skips)} passed, {len(fails)} failed, {len(skips)} skipped. Report: {report}")
    if skips:
        print("Skipped steps verified nothing; fix the reason above to cover them.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())