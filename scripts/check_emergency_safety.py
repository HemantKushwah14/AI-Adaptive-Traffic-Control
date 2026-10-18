"""Signal safety and priority behaviour with a simulated ambulance on the real SUMO signal.
    python scripts\\check_emergency_safety.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import SimulationConfig  # noqa: E402
from simulation.emergency_controller import EmergencyController  # noqa: E402
from simulation.fixed_time_controller import FixedTimeController  # noqa: E402
from simulation.sumo_environment import SumoEnvironment  # noqa: E402
from simulation.traci_controller import SignalMode  # noqa: E402


def run(enabled: bool):
    cfg = SimulationConfig.with_uniform_demand(300, duration=240, seed=11, emergency_events=[(30.0, "E", "left")])
    ctrl = EmergencyController(FixedTimeController(), enabled=enabled)
    bad = []

    def check(env) -> None:
        ctrl.observe(env)
        sig = env.signal
        state = sig.current_state
        greens = {sig._signals[i][0] for i, ch in enumerate(state) if ch == "G"}
        if sig.mode != SignalMode.GREEN and "G" in state:
            bad.append((env.sim_time, "green during transition"))
        if not (greens <= {"N", "S"} or greens <= {"E", "W"}):
            bad.append((env.sim_time, f"conflicting greens {greens}"))

    with SumoEnvironment(cfg, label=f"emsafe_{int(enabled)}") as env:
        env.reset()
        env.run(ctrl, on_step=check)
    return ctrl, bad


def main() -> int:
    on, bad_on = run(True)
    off, bad_off = run(False)
    ev_on, ev_off = on.events_dataframe(), off.events_dataframe()
    modes = set(on.timeline_dataframe()["mode"])
    print(f"safety violations  priority ON: {bad_on}  | priority OFF: {bad_off}")
    print(f"modes seen with priority: {sorted(modes)} | final mode: {on.mode.value}")
    cols = ["required_phase", "status", "wait_for_green_s", "detection_to_clear_s"]
    print("ON :", ev_on[cols].to_dict("records"))
    print("OFF:", ev_off[cols].to_dict("records"))
    ok = (not bad_on and not bad_off and len(ev_on) == 1 and len(ev_off) == 1
          and ev_on.iloc[0]["required_phase"] == "EW_LEFT" and ev_on.iloc[0]["status"] == "cleared"
          and ev_on.iloc[0]["granted_time_s"] == ev_on.iloc[0]["granted_time_s"]
          and {"PREEMPTING", "EMERGENCY_GREEN", "RECOVERY"} <= modes and on.mode.value == "NORMAL")
    print("[ OK ] emergency control is safe and completes the full cycle" if ok else "[FAIL] see output above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())