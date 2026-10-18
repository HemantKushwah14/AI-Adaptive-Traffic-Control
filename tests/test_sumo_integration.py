"""Integration tests on the REAL SUMO/TraCI simulation. Skipped automatically when SUMO is missing.
Short durations keep the whole file to a couple of minutes."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from config.constants import APPROACHES, PHASE_NAMES, STATE_DIM
from config.settings import SimulationConfig
from dashboard.session import SessionState, SimulationSession
from dashboard.sim_settings import default_settings
from dashboard.visualizations.intersection import build_intersection_figure, load_geometry
from evaluation.compare import compare_controllers
from reinforcement_learning.dqn_agent import DQNAgent, DQNConfig
from reinforcement_learning.reward import compute_reward
from simulation.adaptive_controller import AdaptiveController
from simulation.emergency_controller import EmergencyController
from simulation.fixed_time_controller import FixedTimeController
from simulation.sumo_environment import SumoEnvironment
from simulation.traci_controller import SignalMode, SimulationError
from tests.helpers import requires_sumo

pytestmark = [pytest.mark.integration, requires_sumo]


def cfg(duration=120, demand=450.0, seed=1, **kw):
    return SimulationConfig.with_uniform_demand(demand, duration=duration, seed=seed, **kw)


def check_signal_safe(env):
    """Called after every simulation step: the applied signal state must always be legal."""
    sig = env.signal
    state = sig.current_state
    assert len(state) == 12 and set(state) <= set("Gyr")
    if sig.mode != SignalMode.GREEN:
        assert "G" not in state, "no green while a transition is in progress"
    greens = {sig._signals[i][0] for i, ch in enumerate(state) if ch == "G"}
    assert greens <= {"N", "S"} or greens <= {"E", "W"}, f"conflicting greens {greens}"


# ------------------------------------------------------------------ network
def test_network_has_12_signal_links_and_expected_geometry(sumo_network):
    sumolib = pytest.importorskip("sumolib")
    net = sumolib.net.readNet(str(sumo_network), withPrograms=True)
    assert len(net.getTLS("C").getConnections()) == 12
    geo = load_geometry()
    incoming = [lane for lane in geo.lanes if lane.incoming]
    assert len(incoming) == 12 and len(geo.lanes) == 24 and {lane.approach for lane in incoming} == set(APPROACHES)


# ------------------------------------------------------------------ fixed-time
def fixed_run(seed, label):
    with SumoEnvironment(cfg(120, seed=seed), label=label) as env:
        env.reset()
        return env.run(FixedTimeController(), on_step=check_signal_safe)


def test_fixed_time_run_is_safe_and_reproducible(sumo_network):
    a, b, c = fixed_run(5, "t_fix_a"), fixed_run(5, "t_fix_b"), fixed_run(6, "t_fix_c")
    assert a["vehicles_entered"] > 0 and a["sim_time_s"] >= 119
    for key in ("vehicles_entered", "avg_waiting_time_s", "avg_queue_length", "throughput_veh_per_hour"):
        assert a[key] == b[key], f"{key} differs between identical runs"
    assert (a["vehicles_entered"], a["avg_waiting_time_s"]) != (c["vehicles_entered"], c["avg_waiting_time_s"])


# ------------------------------------------------------------------ DQN controlling the real signal
def test_dqn_respects_green_limits_and_signal_safety(sumo_network):
    ctrl = AdaptiveController(DQNAgent(DQNConfig(seed=0), with_buffer=False))   # UNTRAINED: only logic is tested
    c = cfg(240, seed=2)
    changes, prev = [0.0], [0]

    def on_step(env):
        check_signal_safe(env)
        sig = env.signal
        if sig.mode == SignalMode.GREEN and sig.active_phase != prev[0]:
            changes.append(env.sim_time)
            prev[0] = sig.active_phase

    with SumoEnvironment(c, label="t_dqn") as env:
        env.reset()
        env.run(ctrl, on_step=on_step)
    assert ctrl.stats()["decisions"] > 0 and len(changes) >= 2, "max-green must force at least one switch"
    gaps = np.diff(changes)
    assert (gaps >= c.min_green + c.yellow_time).all(), f"min green violated: {gaps}"
    assert (gaps <= c.max_green + c.yellow_time + c.all_red_time + c.decision_interval + 2).all(), f"max green violated: {gaps}"


# ------------------------------------------------------------------ RL step interface
def test_environment_step_interface(sumo_network):
    c = cfg(90, seed=3)
    with SumoEnvironment(c, label="t_step") as env:
        s = env.reset()
        assert s.shape == (STATE_DIM,) and s.dtype == np.float32 and ((s >= 0) & (s <= 1)).all()
        with pytest.raises(ValueError):
            env.step(9)
        last_t, n = 0.0, 0
        while not env.done:
            n += 1
            assert n < 60, "episode must terminate"
            ns, done, info = env.step(env.signal.active_phase)
            assert ns.shape == (STATE_DIM,) and np.isfinite(ns).all() and info["time"] > last_t
            assert np.isfinite(compute_reward(info, interval_s=c.decision_interval)[0])
            last_t = info["time"]
            if not done:
                assert env.signal.mode == SignalMode.GREEN
        assert env.summary()["vehicles_entered"] > 0
    with pytest.raises(SimulationError):
        env.step(0)                                        # closed environment fails cleanly
    env.close()                                            # double close is harmless


# ------------------------------------------------------------------ emergency priority
def run_emergency(enabled):
    c = cfg(240, demand=300.0, seed=11, emergency_events=[(30.0, "E", "left")])
    ctrl = EmergencyController(FixedTimeController(), enabled=enabled)
    with SumoEnvironment(c, label=f"t_em_{int(enabled)}") as env:
        env.reset()
        assert len(env.route_summary.ambulances) == 1
        env.run(ctrl, on_step=lambda e: (check_signal_safe(e), ctrl.observe(e)))
    return ctrl


def test_ambulance_is_detected_served_and_cleared(sumo_network):
    on, off = run_emergency(True), run_emergency(False)
    ev_on, ev_off = on.events_dataframe(), off.events_dataframe()
    assert len(ev_on) == 1 and len(ev_off) == 1
    row = ev_on.iloc[0]
    assert row["required_phase"] == "EW_LEFT" and row["status"] == "cleared" and not pd.isna(row["granted_time_s"])
    assert row["cleared_time_s"] >= row["detected_time_s"] and on.activations >= 1
    assert pd.isna(ev_off.iloc[0]["granted_time_s"]) and not bool(ev_off.iloc[0]["priority_enabled"])


# ------------------------------------------------------------------ dashboard session
def wait_for(cond, timeout, what):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return
        time.sleep(0.05)
    pytest.fail(f"timeout waiting for {what}")


def test_session_runs_to_completion_and_resets(sumo_network):
    s = default_settings()
    s.update(duration=120, controller="fixed_time", speed=0.0, seed=3, step_length=1.0)
    sess = SimulationSession()
    try:
        assert sess.snapshot() is None
        with pytest.raises(SimulationError):
            sess.dispatch_ambulance("N", "straight")
        assert sess.start(s) is None
        wait_for(lambda: sess.state in (SessionState.FINISHED, SessionState.ERROR), 120, "run to finish")
        snap = sess.snapshot()
        assert snap["state"] == "FINISHED" and snap["error"] is None and snap["summary"] is not None
        assert snap["kpi"]["vehicles_entered"] > 0 and len(sess.series_df()) >= 100
        sess.reset()
        assert sess.state == SessionState.IDLE and sess.snapshot() is None
    finally:
        sess.reset()


def test_session_pause_resume_dispatch_stop(sumo_network):
    s = default_settings()
    s.update(duration=300, controller="fixed_time", speed=2.0, seed=4, step_length=1.0)
    sess = SimulationSession()
    try:
        assert sess.start(s) is None
        wait_for(lambda: (sess.snapshot() or {}).get("time", 0) >= 4, 30, "t >= 4 s")
        sess.pause()
        time.sleep(0.6)
        t1 = sess.snapshot()["time"]
        time.sleep(1.0)
        assert sess.snapshot()["time"] == t1 and sess.state == SessionState.PAUSED
        sess.resume()
        wait_for(lambda: sess.snapshot()["time"] > t1 + 2, 30, "time to advance after resume")
        assert sess.dispatch_ambulance("N", "straight").startswith("amb_dash_")
        sess.stop()
        assert sess.state == SessionState.STOPPED and sess.final_summary is not None
    finally:
        sess.reset()


# ------------------------------------------------------------------ comparison
def test_comparison_uses_identical_demand(tmp_path, sumo_network):
    ck = tmp_path / "ck.pt"
    DQNAgent(DQNConfig(seed=0), with_buffer=False).save(ck)           # untrained: plumbing only
    res = compare_controllers(cfg(90, demand=400.0), ck, seeds=[5], fixed_greens=(35, 10, 35, 10))
    assert set(res.summaries["controller"]) == {"fixed_time", "dqn"}
    assert (res.summaries.groupby("seed")["demand_hash"].nunique() == 1).all()
    assert (res.summaries.groupby("seed")["vehicles_generated"].nunique() == 1).all()
    assert any("no training-config metadata" in w for w in res.warnings) and 5 in res.dqn_extras
    assert "improvement_%" in res.improvement_table().columns


# ------------------------------------------------------------------ drawing
def snapshot_with(vehicles):
    links = [(a, i) for a in APPROACHES for i in range(3)]
    state = "".join("G" if a in "NS" and i < 2 else "r" for a, i in links)
    return {"time": 5.0, "vehicles": vehicles,
            "signal": {"links": links, "state_string": state, "phase_names": list(PHASE_NAMES),
                       "active_phase": 0, "mode": "GREEN"},
            "approaches": {a: {"count": 1, "queue": 0, "green": a in "NS"} for a in APPROACHES}}


def test_intersection_figure_draws_only_given_vehicles(sumo_network):
    car = {"id": "v", "x": -3.0, "y": 50.0, "angle": 180.0, "type": "car", "speed": 5.0, "waiting": 0.0}
    amb = {**car, "id": "a", "type": "ambulance"}
    base = len(build_intersection_figure(snapshot_with([])).data)
    assert len(build_intersection_figure(snapshot_with([car])).data) == base + 1
    assert len(build_intersection_figure(snapshot_with([amb])).data) == base + 2        # polygon + highlight ring
    assert len(build_intersection_figure(None).data) > 5