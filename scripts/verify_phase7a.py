"""Phase 7A verification: pass rate of checks + vehicle completion rate on the real SUMO simulation.
Uses the fixed-time controller (no trained model needed).

    python scripts\\verify_phase7a.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.constants import APPROACHES  # noqa: E402
from config.settings import SimulationConfig  # noqa: E402
from dashboard.session import SessionState, SimulationSession  # noqa: E402
from dashboard.sim_settings import default_settings  # noqa: E402
from dashboard.visualizations.intersection import build_intersection_figure, load_geometry  # noqa: E402
from simulation.build_network import ensure_network  # noqa: E402
from simulation.fixed_time_controller import FixedTimeController  # noqa: E402
from simulation.route_generator import generate_routes  # noqa: E402
from simulation.sumo_environment import SumoEnvironment  # noqa: E402
from simulation.traci_controller import SignalMode  # noqa: E402

DURATION = 900
DEMAND = 300.0
VIEW = 110.0
MAX_RED_S = 150.0
COMPLETION_TARGET = 0.90

results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def main() -> int:
    ensure_network()

    # ---------- network and geometry ----------
    geo = load_geometry()
    inc = [lane for lane in geo.lanes if lane.incoming]
    check("Network geometry: 24 lanes, 12 incoming, all 4 approaches",
          len(geo.lanes) == 24 and len(inc) == 12 and {lane.approach for lane in inc} == set(APPROACHES))
    check("Junction centred at origin for drawing",
          all(abs(lane.p1[0]) < 60 and abs(lane.p1[1]) < 60 for lane in inc),
          f"SUMO centre {geo.center[0]:.0f},{geo.center[1]:.0f}")

    # ---------- routes ----------
    cfg = SimulationConfig.with_uniform_demand(DEMAND, duration=DURATION, seed=42)
    rs = generate_routes(cfg, Path("outputs") / "verify_routes.rou.xml")
    check("Routes: vehicles generated for all 4 approaches", all(rs.per_approach[a] > 0 for a in APPROACHES),
          str(rs.per_approach))
    rs2 = generate_routes(cfg, Path("outputs") / "verify_routes2.rou.xml")
    check("Routes: same seed gives identical file", rs.path.read_bytes() == rs2.path.read_bytes())

    # ---------- real simulation ----------
    moving = {a: 0 for a in APPROACHES}              # vehicles seen at speed > 2 m/s on the incoming edge
    moving_ids = {a: set() for a in APPROACHES}
    exited_ids = {a: set() for a in APPROACHES}      # vehicles seen on the outgoing edge toward a side
    green_seen = {a: False for a in APPROACHES}
    red_run = {a: 0.0 for a in APPROACHES}
    max_red = {a: 0.0 for a in APPROACHES}
    safety_bad = []
    depart_t, arrive_ids = {}, set()
    drawn = {a: 0 for a in APPROACHES}
    sampled = [False]
    cx, cy = geo.center

    def on_step(env) -> None:
        sig = env.signal
        state = sig.current_state
        greens = {sig._signals[i][0] for i, ch in enumerate(state) if ch == "G"}
        if sig.mode != SignalMode.GREEN and "G" in state:
            safety_bad.append((env.sim_time, "green during transition"))
        if not (greens <= {"N", "S"} or greens <= {"E", "W"}):
            safety_bad.append((env.sim_time, f"conflict {greens}"))
        for a in APPROACHES:
            if a in greens:
                green_seen[a] = True
                red_run[a] = 0.0
            else:
                red_run[a] += env.cfg.step_length
                max_red[a] = max(max_red[a], red_run[a])
        obs = env.observer
        for vid in obs.newly_departed:
            depart_t[vid] = env.sim_time
        for rec in obs.newly_finished:
            arrive_ids.add(rec["id"])
        for vid, v in obs.vehicles.items():
            lane = v["lane"]
            if len(lane) > 3 and lane[1:3] == "2C" and lane[0] in APPROACHES and v["speed"] > 2.0:
                moving_ids[lane[0]].add(vid)
            if lane.startswith("C2") and len(lane) > 2 and lane[2] in APPROACHES:
                exited_ids[lane[2]].add(vid)
        if env.sim_time >= 300 and not sampled[0]:
            sampled[0] = True
            for v in obs.vehicles.values():
                lane = v["lane"]
                if len(lane) > 3 and lane[1:3] == "2C" and lane[0] in APPROACHES:
                    if abs(v["x"] - cx) <= VIEW and abs(v["y"] - cy) <= VIEW:
                        drawn[lane[0]] += 1

    t0 = time.time()
    with SumoEnvironment(cfg, label="verify7a") as env:
        env.reset()
        summary = env.run(FixedTimeController(), on_step=on_step)
    print(f"\nSimulated {DURATION} s in {time.time() - t0:.0f} s; entered {summary['vehicles_entered']}, "
          f"arrived {summary['vehicles_arrived']}\n")

    check("Simulation ran to the end", summary["sim_time_s"] >= DURATION - 1)
    check("Signal safety on every step", not safety_bad, f"{len(safety_bad)} violations")
    for a in APPROACHES:
        check(f"Approach {a}: vehicles enter and MOVE (speed > 2 m/s)", len(moving_ids[a]) > 0,
              f"{len(moving_ids[a])} vehicles seen moving")
    for a in APPROACHES:
        check(f"Exit {a}: vehicles leave toward this side", len(exited_ids[a]) > 0,
              f"{len(exited_ids[a])} vehicles seen")
    for a in APPROACHES:
        check(f"Approach {a}: gets a green and never waits > {MAX_RED_S:.0f} s", green_seen[a] and max_red[a] <= MAX_RED_S,
              f"longest red {max_red[a]:.0f} s")
    check("Dashboard drawing: vehicles from all 4 approaches fall inside the view at t=300 s",
          all(drawn[a] > 0 for a in APPROACHES), str(drawn))

    snap_vehicles = []
    check("Figure builds with the junction-centred coordinates",
          len(build_intersection_figure({"time": 1.0, "vehicles": snap_vehicles,
                                         "signal": {"links": [(a, i) for a in APPROACHES for i in range(3)],
                                                    "state_string": "G" * 12, "phase_names": ["a", "b", "c", "d"],
                                                    "active_phase": 0, "mode": "GREEN"},
                                         "approaches": {a: {"count": 0, "queue": 0, "green": True} for a in APPROACHES}}
                                        ).data) > 5)

    # ---------- completion rate ----------
    eligible = [v for v, t in depart_t.items() if t <= DURATION - 150]
    done = [v for v in eligible if v in arrive_ids]
    rate = len(done) / len(eligible) if eligible else 0.0
    check(f"Vehicle completion rate >= {COMPLETION_TARGET:.0%} (vehicles that departed >150 s before the end)",
          rate >= COMPLETION_TARGET, f"{len(done)}/{len(eligible)} = {rate:.1%}")

    # ---------- dashboard session ----------
    s = default_settings()
    s.update(duration=120, controller="fixed_time", speed=0.0, seed=3, step_length=1.0)
    sess = SimulationSession()
    try:
        err = sess.start(s)
        t1 = time.time()
        while sess.state == SessionState.RUNNING and time.time() - t1 < 120:
            time.sleep(0.1)
        snap = sess.snapshot()
        check("Dashboard session runs to FINISHED without error",
              err is None and snap is not None and snap["state"] == "FINISHED" and not snap["error"])
        check("Session produced time series and vehicles", len(sess.series_df()) >= 100)
    finally:
        sess.reset()

    # ---------- result ----------
    passed = sum(ok for _, ok in results)
    pct = 100.0 * passed / len(results)
    print(f"\nChecks passed: {passed}/{len(results)} = {pct:.1f}%")
    print(f"Vehicle completion rate: {rate:.1%}")
    good = pct >= 90.0 and rate >= COMPLETION_TARGET
    print("[ OK ] Phase 7A working (>= 90% of checks and completion rate)" if good else "[FAIL] below target, see FAIL lines")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())