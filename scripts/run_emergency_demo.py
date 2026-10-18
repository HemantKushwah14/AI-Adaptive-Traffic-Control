"""Phase 6 milestone: reproducible ambulance scenario, SAME seed with and without priority.

    python scripts\\run_emergency_demo.py
    python scripts\\run_emergency_demo.py --events "100:N:straight,260:E:left" --duration 600 --seed 7
    python scripts\\run_emergency_demo.py --base dqn            (needs models\\dqn\\best.pt or last.pt)
    python scripts\\run_emergency_demo.py --gui

Ambulances here are EXPLICITLY SIMULATED vehicles; the "detection" is the SUMO vehicle type
read through TraCI (source: simulation_ground_truth).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config.emergency_config import (DEFAULT_CLEARANCE_BUFFER_S, DEFAULT_DETECTION_RANGE_M,  # noqa: E402
                                     DEFAULT_MAX_HOLD_S, DEFAULT_SCENARIO)
from config.settings import SimulationConfig, SumoNotFoundError  # noqa: E402
from evaluation.emergency_report import ambulance_comparison, save_emergency_report  # noqa: E402
from reinforcement_learning.dqn_agent import CheckpointError  # noqa: E402
from simulation.adaptive_controller import AdaptiveController  # noqa: E402
from simulation.emergency_controller import EmergencyController  # noqa: E402
from simulation.fixed_time_controller import FixedTimeController  # noqa: E402
from simulation.sumo_environment import SumoEnvironment  # noqa: E402
from simulation.traci_controller import SimulationError  # noqa: E402

KPIS = ["avg_waiting_time_s", "avg_queue_length", "throughput_veh_per_hour", "avg_travel_time_s",
        "avg_stopped_vehicles", "total_delay_s"]


def parse_events(text: str) -> List[Tuple[float, str, str]]:
    out = []
    for item in text.split(","):
        parts = item.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad event '{item}': expected time:approach:movement, e.g. 100:N:straight")
        out.append((float(parts[0]), parts[1].strip().upper(), parts[2].strip().lower()))
    return out


def run_scenario(cfg: SimulationConfig, base, enabled: bool, label: str, args):
    ctrl = EmergencyController(base, enabled=enabled, detection_range_m=args.detection_range,
                               clearance_buffer_s=args.buffer, max_hold_s=args.max_hold)
    with SumoEnvironment(cfg, label=label) as env:
        env.reset(seed=cfg.seed)
        route_hash = hashlib.sha256(Path(env.route_summary.path).read_bytes()).hexdigest()[:16]
        n_amb = len(env.route_summary.ambulances)
        summary = env.run(ctrl, on_step=ctrl.observe)
    return {"summary": summary, "events": ctrl.events_dataframe(), "timeline": ctrl.timeline_dataframe(),
            "stats": ctrl.stats(), "hash": route_hash, "ambulances_scheduled": n_amb,
            "messages": list(ctrl.messages)}


def main() -> int:
    p = argparse.ArgumentParser(description="Emergency priority demo (with vs without priority).")
    p.add_argument("--base", choices=["fixed", "dqn"], default="fixed")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--events", default=",".join(f"{t:g}:{a}:{m}" for t, a, m in DEFAULT_SCENARIO))
    p.add_argument("--duration", type=int, default=600)
    p.add_argument("--demand", type=float, default=500.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--detection-range", type=float, default=DEFAULT_DETECTION_RANGE_M)
    p.add_argument("--buffer", type=float, default=DEFAULT_CLEARANCE_BUFFER_S)
    p.add_argument("--max-hold", type=float, default=DEFAULT_MAX_HOLD_S)
    p.add_argument("--gui", action="store_true")
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()

    try:
        events = parse_events(a.events)
        cfg = SimulationConfig.with_uniform_demand(a.demand, duration=a.duration, seed=a.seed,
                                                   emergency_events=events, gui=a.gui)
        cfg.validate()
        make_base = (lambda: FixedTimeController()) if a.base == "fixed" else \
            (lambda ck=AdaptiveController.from_checkpoint(a.checkpoint): ck)
        print(f"Scenario: base={a.base} | duration {cfg.duration}s | demand {a.demand:.0f} veh/h/approach | "
              f"seed {cfg.seed}\nSimulated ambulances (time, approach, movement): {events}")
        print("Detection source: simulation_ground_truth (SIMULATED, not computer vision)\n")
        print("Run 1/2: WITHOUT priority ...")
        off = run_scenario(cfg, make_base(), False, "emerg_off", a)
        print("Run 2/2: WITH priority ...")
        on = run_scenario(cfg, make_base(), True, "emerg_on", a)
        if off["hash"] != on["hash"]:
            print("ERROR: demand differs between runs (route hashes differ); comparison invalid.")
            return 1
    except (CheckpointError, SumoNotFoundError, SimulationError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    table = ambulance_comparison(off["events"], on["events"])
    kpi = pd.DataFrame({"no_priority": [off["summary"][k] for k in KPIS],
                        "priority": [on["summary"][k] for k in KPIS]}, index=KPIS)
    kpi["difference"] = kpi["priority"] - kpi["no_priority"]
    print(f"\nAmbulances scheduled: {on['ambulances_scheduled']} | detected (with priority): {len(on['events'])}")
    print("\n=== Ambulances: without vs with priority (measured via TraCI) ===")
    print(table.round(1).to_string(index=False) if len(table) else "No ambulance was detected.")
    print("\n=== Overall traffic KPIs: cost/benefit of priority ===")
    print(kpi.round(2).to_string())
    print(f"\nPriority controller: {on['stats']['activations']} activation(s), {on['stats']['timeouts']} timeout(s)")
    print("\nEvent log (with priority):")
    for m in on["messages"]:
        print(f"  t={m['time']:6.0f}s  {m['message']}")

    if not a.no_save:
        report = {"base": a.base, "config": cfg.to_dict(), "events_scheduled": events,
                  "detection_source": "simulation_ground_truth (simulated)", "route_hash": on["hash"],
                  "stats_priority": on["stats"], "messages": on["messages"],
                  "settings": {"detection_range_m": a.detection_range, "clearance_buffer_s": a.buffer,
                               "max_hold_s": a.max_hold}}
        out = save_emergency_report(report, {
            "events_no_priority": off["events"], "events_priority": on["events"],
            "timeline_no_priority": off["timeline"], "timeline_priority": on["timeline"],
            "ambulance_comparison": table, "kpi_comparison": kpi.reset_index(names="metric")})
        print(f"\nSaved to {out} (copied to outputs\\emergency\\latest)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())