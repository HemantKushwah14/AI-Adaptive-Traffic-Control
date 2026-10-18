"""Replay buffer, reward, action mask, Q-network and DQN agent (including the Bellman target)."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from config.constants import ACTION_DIM, APPROACHES, STATE_DIM
from config.settings import SimulationConfig
from reinforcement_learning.action_mask import legal_action_mask
from reinforcement_learning.dqn_agent import (CheckpointError, DQNAgent, DQNConfig, _selftest,
                                              read_checkpoint_meta)
from reinforcement_learning.network import QNetwork
from reinforcement_learning.replay_buffer import ReplayBuffer
from reinforcement_learning.reward import RewardWeights, compute_reward
from simulation.sumo_environment import SumoEnvironment


# ------------------------------------------------------------------ replay buffer
def fill_buffer(buf, n, dim=17, act=4):
    for i in range(n):
        buf.add(np.full(dim, i, np.float32), i % act, float(i), np.full(dim, i + 1, np.float32),
                np.ones(act, bool), False)


def test_buffer_is_circular():
    buf = ReplayBuffer(5, 17, 4)
    fill_buffer(buf, 8)
    assert len(buf) == 5
    assert set(buf.s[:, 0].tolist()) == {3.0, 4.0, 5.0, 6.0, 7.0}


def test_buffer_sample_shapes_alignment_and_seed():
    b1, b2 = ReplayBuffer(100, 17, 4, seed=3), ReplayBuffer(100, 17, 4, seed=3)
    fill_buffer(b1, 50)
    fill_buffer(b2, 50)
    s, a, r, s2, term, m2 = b1.sample(16)
    assert s.shape == (16, 17) and s.dtype == torch.float32 and a.dtype == torch.int64
    assert m2.shape == (16, 4) and m2.dtype == torch.bool and term.shape == (16,)
    assert torch.allclose(r, s[:, 0]) and torch.allclose(s2[:, 0], s[:, 0] + 1), "columns must stay aligned"
    assert torch.equal(a, (s[:, 0].long() % 4))
    assert torch.equal(b2.sample(16)[1], a), "same seed must sample identically"


def test_buffer_rejects_bad_input():
    buf = ReplayBuffer(10, 17, 4)
    ok_s, ok_m = np.zeros(17, np.float32), np.ones(4, bool)
    with pytest.raises(ValueError):
        buf.sample(1)
    with pytest.raises(ValueError):
        buf.add(np.zeros(5), 0, 0.0, ok_s, ok_m, False)
    with pytest.raises(ValueError):
        buf.add(ok_s, 9, 0.0, ok_s, ok_m, False)
    with pytest.raises(ValueError):
        buf.add(ok_s, 0, 0.0, ok_s, np.ones(3, bool), False)
    with pytest.raises(ValueError):
        buf.add(np.full(17, np.nan, np.float32), 0, 0.0, ok_s, ok_m, False)
    with pytest.raises(ValueError):
        buf.add(ok_s, 0, float("inf"), ok_s, ok_m, False)
    with pytest.raises(ValueError):
        ReplayBuffer(0, 17, 4)


# ------------------------------------------------------------------ reward
def info(**kw):
    base = {"interval_waiting_mean": 0.0, "interval_queue_mean": 0.0,
            "waiting_by_approach": {a: 0.0 for a in APPROACHES}, "switched": False, "interval_arrived": 0}
    base.update(kw)
    return base


def test_reward_idle_is_zero():
    r, comps = compute_reward(info())
    assert r == 0.0 and all(v == 0.0 for v in comps.values())


def test_reward_terms_match_formula():
    assert compute_reward(info(interval_queue_mean=15.0))[0] == pytest.approx(-0.5 * 0.5)
    assert compute_reward(info(interval_waiting_mean=500.0))[0] == pytest.approx(-1.0 * 0.5)
    assert compute_reward(info(switched=True))[0] == pytest.approx(-0.05)
    assert compute_reward(info(interval_arrived=2.5))[0] == pytest.approx(0.3 * 1.0)
    unequal = info(waiting_by_approach={"N": 1000.0, "S": 0.0, "E": 0.0, "W": 0.0})
    assert compute_reward(unequal)[0] == pytest.approx(-0.3 * float(np.std([1, 0, 0, 0])))


def test_reward_clipping_and_monotonicity():
    assert compute_reward(info(interval_waiting_mean=1e9))[0] == pytest.approx(-3.0)
    assert compute_reward(info(interval_arrived=1000))[0] == pytest.approx(0.3 * 2.0)
    assert compute_reward(info(interval_queue_mean=20))[0] < compute_reward(info(interval_queue_mean=5))[0]


def test_negative_reward_weights_rejected():
    with pytest.raises(ValueError):
        compute_reward(info(), RewardWeights(waiting=-1.0))


# ------------------------------------------------------------------ action mask
def stub_env(active, elapsed):
    return SimpleNamespace(signal=SimpleNamespace(active_phase=active, elapsed_green=elapsed), cfg=SimulationConfig())


@pytest.mark.parametrize("active", range(4))
def test_mask_matches_environment_legalisation(active):
    for elapsed in range(0, 71):
        env = stub_env(active, float(elapsed))
        mask = legal_action_mask(env)
        assert mask.any()
        for action in range(ACTION_DIM):
            applied, reason = SumoEnvironment.legalise_action(env, action)
            if mask[action]:
                assert (applied, reason) == (action, ""), (active, elapsed, action)
            else:
                assert reason != "", (active, elapsed, action)


def test_mask_rules_and_invalid_action():
    assert legal_action_mask(stub_env(2, 3.0)).tolist() == [False, False, True, False]   # before min green
    assert legal_action_mask(stub_env(2, 30.0)).all()
    assert legal_action_mask(stub_env(2, 58.0)).tolist() == [True, True, False, True]    # near max green
    with pytest.raises(ValueError):
        SumoEnvironment.legalise_action(stub_env(0, 20.0), 9)


# ------------------------------------------------------------------ network
def test_qnetwork_shapes_and_validation():
    net = QNetwork(17, 4)
    assert net(torch.zeros(5, 17)).shape == (5, 4)
    assert net.n_parameters() == 27332
    with pytest.raises(ValueError):
        net(torch.zeros(5, 16))
    with pytest.raises(ValueError):
        net(torch.zeros(17))
    with pytest.raises(ValueError):
        QNetwork(0, 4)
    with pytest.raises(ValueError):
        QNetwork(17, 1)


# ------------------------------------------------------------------ DQN agent
def small_agent(**kw):
    cfg = DQNConfig(batch_size=8, buffer_size=200, learn_start=8, target_update_steps=2,
                    eps_decay_steps=100, seed=0, **kw)
    return DQNAgent(cfg)


def fill_agent(agent, n=40, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        agent.remember(rng.random(17).astype(np.float32), int(rng.integers(4)), float(rng.normal()),
                       rng.random(17).astype(np.float32), np.ones(4, bool), False)


def test_config_validation():
    for bad in (dict(gamma=1.0), dict(lr=0.0), dict(batch_size=100, buffer_size=50),
                dict(learn_start=10, batch_size=64), dict(eps_start=0.1, eps_end=0.5)):
        with pytest.raises(ValueError):
            DQNConfig(**bad).validate()


def test_epsilon_schedule():
    a = small_agent()
    assert a.epsilon == 1.0
    a.env_steps = 50
    assert a.epsilon == pytest.approx(0.525)
    a.env_steps = 100
    assert a.epsilon == pytest.approx(0.05)
    a.env_steps = 10_000
    assert a.epsilon == pytest.approx(0.05)


def test_exploration_never_picks_illegal_actions():
    a = small_agent()                                   # epsilon = 1.0 -> always random
    state = np.zeros(17, np.float32)
    mask = np.array([False, True, False, True])
    assert {a.select_action(state, mask) for _ in range(300)} <= {1, 3}


def test_greedy_picks_best_legal_action_and_validates_input():
    a = small_agent()
    state = np.random.default_rng(1).random(17).astype(np.float32)
    q = a.q_values(state)
    mask = np.array([False, True, True, False])
    assert a.select_action(state, mask, greedy=True) == (1 if q[1] >= q[2] else 2)
    with pytest.raises(ValueError):
        a.select_action(state, np.zeros(4, bool))
    with pytest.raises(ValueError):
        a.select_action(state, np.ones(3, bool))
    with pytest.raises(ValueError):
        a.select_action(np.full(17, np.nan, np.float32))
    with pytest.raises(ValueError):
        a.select_action(np.zeros(5, np.float32))


def test_learning_waits_for_enough_data_then_updates_weights():
    a = small_agent()
    fill_agent(a, 5)
    assert a.learn() is None
    fill_agent(a, 40)
    before = copy.deepcopy(a.policy.state_dict())
    out = a.learn()
    assert out is not None and np.isfinite(out["loss"]) and a.learn_steps == 1
    assert any(not torch.equal(before[k], v) for k, v in a.policy.state_dict().items())


def test_target_network_is_frozen_until_sync():
    a = small_agent()
    fill_agent(a)
    assert all(not p.requires_grad for p in a.target.parameters())
    init_target = copy.deepcopy(a.target.state_dict())
    a.learn()
    assert all(torch.equal(init_target[k], v) for k, v in a.target.state_dict().items())
    a.learn()                                              # learn_steps == target_update_steps -> sync
    assert all(torch.equal(a.policy.state_dict()[k], v) for k, v in a.target.state_dict().items())


def _zero(net, bias=None):
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        if bias is not None:
            net.net[-1].bias.copy_(torch.tensor(bias, dtype=torch.float32))


def bellman_loss(mask, double, terminal=False):
    cfg = DQNConfig(gamma=0.9, batch_size=8, buffer_size=64, learn_start=8, target_update_steps=1000,
                    seed=0, double_dqn=double)
    a = DQNAgent(cfg)
    _zero(a.policy)                                         # Q_policy = 0 everywhere
    _zero(a.target, bias=[100.0, 0.0, 0.0, 0.0])            # target values action 0 hugely
    z = np.zeros(17, np.float32)
    for i in range(8):
        a.remember(z, i % 4, 0.0, z, np.array(mask, bool), terminal)
    return a.learn()["loss"]


@pytest.mark.parametrize("double", [False, True])
def test_bellman_target_respects_legal_action_mask(double):
    # y = r + gamma * max_{legal a'} Q_target(s', a');  Q_policy = 0  ->  loss = Huber(0 - y)
    assert bellman_loss([False, True, True, True], double) == pytest.approx(0.0, abs=1e-6)   # y = 0
    assert bellman_loss([True, True, True, True], double) == pytest.approx(89.5, rel=1e-4)    # y = 90
    assert bellman_loss([True, True, True, True], double, terminal=True) == pytest.approx(0.0, abs=1e-6)


def test_toy_problem_selftest_passes():
    assert _selftest() == 0


def test_checkpoint_roundtrip(tmp_path):
    a = small_agent()
    fill_agent(a)
    a.learn()
    a.episodes = 3
    path = tmp_path / "m.pt"
    a.save(path, meta={"best_val_wait_s": 12.5})
    assert not (tmp_path / "m.pt.tmp").exists()
    b = DQNAgent.from_checkpoint(path)
    s = np.random.default_rng(2).random(17).astype(np.float32)
    np.testing.assert_allclose(a.q_values(s), b.q_values(s), rtol=1e-6)
    assert b.episodes == 3 and b.meta["best_val_wait_s"] == 12.5 and b.buffer is None
    assert b.config.hidden == a.config.hidden
    with pytest.raises(RuntimeError):
        b.remember(s, 0, 0.0, s, np.ones(4, bool))
    meta = read_checkpoint_meta(path)
    assert meta["episodes"] == 3 and meta["size_kb"] > 0 and "config" in meta


def test_bad_checkpoints_raise_checkpoint_error(tmp_path):
    with pytest.raises(CheckpointError):
        DQNAgent.from_checkpoint(tmp_path / "missing.pt")
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError):
        DQNAgent.from_checkpoint(bad)
    odd = tmp_path / "odd.pt"
    DQNAgent(DQNConfig(state_dim=10, action_dim=4), with_buffer=False).save(odd)
    with pytest.raises(CheckpointError, match="do not match"):
        DQNAgent.from_checkpoint(odd)
    assert STATE_DIM == 17