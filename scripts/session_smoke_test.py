"""Headless test of the simulation session (no Streamlit): start, pause, resume, speed change,
dispatch a simulated ambulance, run to the end, inspect the snapshot.

    python scripts\\session_smoke_test.py
    python scripts\\session_smoke_test.py --controller dqn --duration 300
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.session import SessionState, SimulationSession  # noqa: E402
from dashboard.sim_settings import default_settings  # noqa: E402


def wait_for(cond, timeout: float, what: str) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.1)
    print(f"[FAIL] timeout waiting for {what}")
    return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--controller", choices=["fixed_time", "dqn"], default="fixed_time")
    p.add_argument("--duration", type=int, default=240)
    a = p.parse_args()
    s = default_settings()
    s.update(duration=a.duration, controller=a.controller, speed=5.0)
    sess = SimulationSession()
    ok = True

    err = sess.start(s)
    if err:
        print("[FAIL] could not start:\n", err)
        return 1
    print("started; state =", sess.state.value)
    ok &= wait_for(lambda: (sess.snapshot() or {}).get("time", 0) >= 20, 60, "t >= 20 s")

    sess.pause()
    time.sleep(0.4)
    t1 = sess.snapshot()["time"]
    time.sleep(1.0)
    t2 = sess.snapshot()["time"]
    print(f"paused: t {t1:.0f} -> {t2:.0f} (must be equal)")
    ok &= abs(t2 - t1) < 1e-9 and sess.state == SessionState.PAUSED
    sess.resume()

    try:
        print("dispatched", sess.dispatch_ambulance("N", "straight"))
    except Exception as exc:
        print("[FAIL] dispatch:", exc)
        ok = False

    sess.set_speed(0.0)   # Max
    ok &= wait_for(lambda: sess.state in (SessionState.FINISHED, SessionState.ERROR), 600, "run to finish")
    snap = sess.snapshot()
    print("final state:", snap["state"], "| error:", snap["error"])
    ok &= snap["state"] == "FINISHED"
    print("KPIs:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in snap["kpi"].items()})
    print("series rows:", len(sess.series_df()), "| vehicles in last frame:", len(snap["vehicles"]))
    print("emergency events:", len(sess.emergency_events_df()))
    print("[ OK ] session smoke test passed" if ok else "[FAIL] session smoke test failed")
    sess.reset()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())