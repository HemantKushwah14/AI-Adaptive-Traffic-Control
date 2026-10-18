"""Fixed-time, adaptive and emergency controller logic (stub environments, no SUMO)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from computer_vision.ambulance_detector import AmbulanceSighting
from computer_vision.detector import Detection
from config.settings import SimulationConfig
from reinforcement_learning.dqn_agent import DQNAgent, DQNConfig
from simulation.adaptive_controller import AdaptiveController, _selftest as adaptive_selftest, explain_decision
from simulation.emergency_controller import (EmergencyController, EmergencyMode, ExternalDetectionSource,
                                             SimulationTruthSource, _selftest as emergency_selftest,
                                             required_phase, submit_vision_sightings)
from simulation.fixed_time_controller import FixedTimeController


def sig_env(active, elapsed):
    return SimpleNamespace(signal=SimpleNamespace(active_phase=active, elapsed_green=elapsed))


# ------------------------------------------------------------------ fixed time
def test_fixed_time_cycles_through_all_phases():
    c = FixedTimeController((35, 10, 35, 10))
    assert c.select_phase(sig_env(0, 34.0)) == 0
    assert c.select_phase(sig_env(0, 35.0)) == 1
    assert c.select_phase(sig_env(1, 10.0)) == 2
    assert c.select_phase(sig_env(3, 12.0)) == 0          # wraps around


def test_fixed_time_validation():
    with pytest.raises(ValueError):
        FixedTimeController((35, 10, 35))
    with pytest.raises(ValueError):
        FixedTimeController((35, 2, 35, 10))


# ------------------------------------------------------------------ adaptive
def new_adaptive(**kw):
    return AdaptiveController(DQNAgent(DQNConfig(seed=0), with_buffer=False), **kw)


def test_adaptive_timing_selftest():
    assert adaptive_selftest() == 0


def test_explanation_uses_state_and_q_values_and_carries_caveat():
    state = np.zeros(17, np.float32)
    state[4] = 1.0                                           # North queue = 30 vehicles
    q = np.array([1.0, 0.0, 0.0, 0.0])
    ex = explain_decision(state, q, np.ones(4, bool), 0, 0, "", 0, 20.0)
    assert ex["ranking"][0]["name"] == "NS_THROUGH" and ex["margin"] == pytest.approx(1.0)
    assert ex["queue_by_approach"]["N"] == pytest.approx(30.0) and ex["agrees_with_queue_pressure"]
    assert "hold" in ex["summary"] and "not a complete causal" in ex["caveat"]


def test_explanation_reports_safety_override_and_single_legal_action():
    state = np.zeros(17, np.float32)
    ex = explain_decision(state, np.array([0.1, 0.9, 0.0, 0.0]), np.array([True, False, False, False]),
                          requested=1, applied=0, reason="min_green", active_before=0, elapsed=3.0)
    assert any("min_green" in line for line in ex["lines"]) and ex["margin"] is None


def test_compatibility_warnings():
    ctrl = new_adaptive()
    assert "no training-config metadata" in ctrl.compatibility_warnings(SimulationConfig())[0]
    ctrl.agent.meta = {"train_sim_config": {"min_green": 10, "max_green": 60, "decision_interval": 5,
                                            "yellow_time": 3, "all_red_time": 1, "step_length": 1.0},
                       "demand_range": [300, 800]}
    assert ctrl.compatibility_warnings(SimulationConfig.with_uniform_demand(500)) == []
    cfg = SimulationConfig.with_uniform_demand(100, max_green=90)
    text = " ".join(ctrl.compatibility_warnings(cfg))
    assert "max_green" in text and "outside the training range" in text


def test_adaptive_info_and_history_limit():
    ctrl = new_adaptive(history_size=3)
    info = ctrl.info()
    assert info["epsilon_in_use"] == 0.0 and info["parameters"] == 27332 and ctrl.history.maxlen == 3


# ------------------------------------------------------------------ emergency
def test_emergency_logic_selftest():
    assert emergency_selftest() == 0


@pytest.mark.parametrize("approach,lane,phase", [("N", 0, 0), ("N", 1, 0), ("N", 2, 1), ("S", 2, 1),
                                                 ("E", 0, 2), ("W", 1, 2), ("E", 2, 3), ("W", 2, 3)])
def test_required_phase_table(approach, lane, phase):
    assert required_phase(approach, lane) == phase


def test_required_phase_rejects_unknown():
    with pytest.raises(ValueError):
        required_phase("X", 0)
    with pytest.raises(ValueError, match="No phase serves"):
        required_phase("N", 5)


def test_emergency_constructor_validation():
    with pytest.raises(ValueError):
        EmergencyController(FixedTimeController(), max_hold_s=2.0)
    with pytest.raises(ValueError):
        EmergencyController(FixedTimeController(), preempt_min_green=-1.0)
    with pytest.raises(ValueError):
        SimulationTruthSource(detection_range_m=0)


def test_external_source_ttl_and_validation():
    src = ExternalDetectionSource(ttl_s=3.0)
    env = SimpleNamespace(sim_time=10.0)
    src.submit("N", 10.0, lane_index=1, confidence=0.8)
    reqs = src.poll(env)
    assert len(reqs) == 1 and reqs[0].approach == "N" and reqs[0].vehicle_id.startswith("vision_N_1_")
    env.sim_time = 12.0
    assert len(src.poll(env)) == 1
    env.sim_time = 14.0                                      # not refreshed for > ttl
    assert src.poll(env) == []
    with pytest.raises(ValueError):
        src.submit("Q", 1.0)
    with pytest.raises(ValueError):
        src.submit("N", 1.0, lane_index=7)


def test_submit_vision_sightings_skips_unlocated():
    d = Detection(0, 0, 10, 10, 0.9, "ambulance", "ambulance", 0)
    src = ExternalDetectionSource()
    n = submit_vision_sightings(src, 5.0, [AmbulanceSighting(d, "E", 2, True), AmbulanceSighting(d, None, None, True)])
    assert n == 1 and len(src.poll(SimpleNamespace(sim_time=5.0))) == 1


def emergency_env(t, active, elapsed):
    sig = SimpleNamespace(active_phase=active, elapsed_green=elapsed, mode=SimpleNamespace(value="GREEN"))
    return SimpleNamespace(sim_time=t, cfg=SimulationConfig(), signal=sig, route_summary=None,
                           observer=SimpleNamespace(emergency_vehicles=lambda: []))


def test_vision_request_preempts_holds_clears_and_restores():
    src = ExternalDetectionSource(ttl_s=3.0)
    ctrl = EmergencyController(FixedTimeController(), sources=[src])
    src.submit("E", 10.0, lane_index=2, confidence=0.9)
    assert ctrl.select_phase(emergency_env(10.0, 0, 20.0)) == 3          # EW_LEFT requested
    assert ctrl.mode == EmergencyMode.PREEMPTING and ctrl.activations == 1
    src.submit("E", 11.0, lane_index=2, confidence=0.9)
    assert ctrl.select_phase(emergency_env(11.0, 3, 0.0)) == 3           # signal now serves it -> hold
    assert ctrl.mode == EmergencyMode.EMERGENCY_GREEN
    assert ctrl.select_phase(emergency_env(20.0, 3, 9.0)) == 3           # request expired -> recovery buffer
    assert ctrl.mode == EmergencyMode.RECOVERY
    assert ctrl.select_phase(emergency_env(23.0, 3, 12.0)) == 0          # base fixed-time plan resumes
    assert ctrl.mode == EmergencyMode.NORMAL
    ev = ctrl.events_dataframe().iloc[0]
    assert ev["status"] == "cleared" and ev["granted_time_s"] == 11.0 and ev["cleared_time_s"] == 20.0
    assert ev["wait_for_green_s"] == 1.0 and ev["required_phase"] == "EW_LEFT"


def test_hold_timeout_releases_signal_and_does_not_regrant():
    src = ExternalDetectionSource(ttl_s=3.0)
    ctrl = EmergencyController(FixedTimeController(), sources=[src], max_hold_s=5.0)
    src.submit("E", 10.0, 2)
    ctrl.select_phase(emergency_env(10.0, 0, 20.0))
    src.submit("E", 11.0, 2)
    ctrl.select_phase(emergency_env(11.0, 3, 0.0))
    src.submit("E", 17.0, 2)                                             # still "present" after 6 s > max hold
    assert ctrl.select_phase(emergency_env(17.0, 3, 6.0)) == 3
    assert ctrl.timeouts == 1 and ctrl.mode == EmergencyMode.RECOVERY
    src.submit("E", 18.0, 2)
    ctrl.select_phase(emergency_env(18.0, 3, 7.0))
    src.submit("E", 25.0, 2)
    assert ctrl.select_phase(emergency_env(25.0, 3, 12.0)) == 0          # normal control; no endless priority
    assert ctrl.mode == EmergencyMode.NORMAL and len(ctrl.events) == 1
    assert bool(ctrl.events_dataframe().iloc[0]["timed_out"])