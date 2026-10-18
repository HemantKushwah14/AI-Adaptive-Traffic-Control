"""Deep Q-Network agent: policy net, target net, replay, epsilon-greedy, Bellman targets.

Self-test of the learning rule on a synthetic toy problem (NOT a traffic result):
    python -m reinforcement_learning.dqn_agent --selftest
"""
from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config.constants import ACTION_DIM, STATE_DIM
from reinforcement_learning.network import QNetwork
from reinforcement_learning.replay_buffer import ReplayBuffer
from utils.logging_utils import get_logger

log = get_logger(__name__)


class CheckpointError(RuntimeError):
    """Raised for missing / incompatible checkpoints."""


@dataclass
class DQNConfig:
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    hidden: Tuple[int, ...] = (128, 128, 64)
    gamma: float = 0.95
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    learn_start: int = 500            # min stored transitions before learning
    target_update_steps: int = 500    # learning steps between hard target syncs
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 8_000      # linear decay measured in stored transitions
    grad_clip: float = 10.0
    double_dqn: bool = False
    seed: int = 42
    device: str = "cpu"

    def validate(self) -> None:
        e = []
        if not (0.0 <= self.gamma < 1.0):
            e.append("gamma must be in [0,1)")
        if self.lr <= 0:
            e.append("lr must be > 0")
        if self.batch_size < 1 or self.buffer_size < self.batch_size:
            e.append("need 1 <= batch_size <= buffer_size")
        if self.learn_start < self.batch_size:
            e.append("learn_start must be >= batch_size")
        if self.target_update_steps < 1 or self.eps_decay_steps < 1:
            e.append("target_update_steps and eps_decay_steps must be >= 1")
        if not (0.0 <= self.eps_end <= self.eps_start <= 1.0):
            e.append("need 0 <= eps_end <= eps_start <= 1")
        if self.state_dim < 1 or self.action_dim < 2:
            e.append("invalid state/action dimensions")
        if e:
            raise ValueError("Invalid DQN config: " + "; ".join(e))


class DQNAgent:
    def __init__(self, config: Optional[DQNConfig] = None, with_buffer: bool = True) -> None:
        self.config = config or DQNConfig()
        c = self.config
        c.hidden = tuple(int(h) for h in c.hidden)
        c.validate()
        torch.manual_seed(c.seed)
        self.rng = np.random.default_rng(c.seed)
        self.device = torch.device(c.device)
        self.policy = QNetwork(c.state_dim, c.action_dim, c.hidden).to(self.device)
        self.target = QNetwork(c.state_dim, c.action_dim, c.hidden).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=c.lr)
        self.buffer = ReplayBuffer(c.buffer_size, c.state_dim, c.action_dim, c.seed) if with_buffer else None
        self.env_steps = 0       # stored transitions (drives epsilon)
        self.learn_steps = 0
        self.episodes = 0
        self.meta: Dict = {}

    # ---------------- policy ----------------
    @property
    def epsilon(self) -> float:
        c = self.config
        frac = min(self.env_steps / c.eps_decay_steps, 1.0)
        return float(c.eps_start + frac * (c.eps_end - c.eps_start))

    def _tensor(self, state) -> torch.Tensor:
        arr = np.asarray(state, dtype=np.float32)
        if arr.shape != (self.config.state_dim,):
            raise ValueError(f"State must have shape ({self.config.state_dim},), got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError("State contains NaN/inf")
        return torch.as_tensor(arr, device=self.device).unsqueeze(0)

    def q_values(self, state) -> np.ndarray:
        with torch.no_grad():
            return self.policy(self._tensor(state)).squeeze(0).cpu().numpy()

    def select_action(self, state, mask=None, greedy: bool = False) -> int:
        a_dim = self.config.action_dim
        mask = np.ones(a_dim, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if mask.shape != (a_dim,) or not mask.any():
            raise ValueError("mask must have one entry per action and allow at least one action")
        state_t = self._tensor(state)
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.choice(np.flatnonzero(mask)))
        with torch.no_grad():
            q = self.policy(state_t).squeeze(0).cpu().numpy()
        q[~mask] = -np.inf
        return int(np.argmax(q))

    # ---------------- learning ----------------
    def remember(self, state, action, reward, next_state, next_mask, terminal: bool = False) -> None:
        if self.buffer is None:
            raise RuntimeError("This agent was created without a replay buffer (inference mode).")
        self.buffer.add(state, action, reward, next_state, next_mask, terminal)
        self.env_steps += 1

    def learn(self) -> Optional[Dict[str, float]]:
        c = self.config
        if self.buffer is None or len(self.buffer) < max(c.batch_size, c.learn_start):
            return None
        s, a, r, s2, term, m2 = self.buffer.sample(c.batch_size, str(self.device))
        q = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            m2 = m2.clone()
            m2[~m2.any(dim=1)] = True                       # defensive: never an all-illegal row
            q2_target = self.target(s2)
            if c.double_dqn:
                a2 = self.policy(s2).masked_fill(~m2, float("-inf")).argmax(dim=1, keepdim=True)
                next_q = q2_target.gather(1, a2).squeeze(1)
            else:
                next_q = q2_target.masked_fill(~m2, float("-inf")).max(dim=1).values
            y = r + c.gamma * (1.0 - term) * next_q          # Bellman target
        loss = F.smooth_l1_loss(q, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("DQN loss became NaN/inf (check rewards / learning rate)")
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), c.grad_clip)
        self.optimizer.step()
        self.learn_steps += 1
        if self.learn_steps % c.target_update_steps == 0:
            self.sync_target()
        return {"loss": float(loss.item()), "mean_q": float(q.mean().item())}

    def sync_target(self) -> None:
        self.target.load_state_dict(self.policy.state_dict())

    # ---------------- persistence ----------------
    def save(self, path: Path, meta: Optional[dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "config": dataclasses.asdict(self.config),
            "policy_state": self.policy.state_dict(),
            "target_state": self.target.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "env_steps": self.env_steps, "learn_steps": self.learn_steps,
            "episodes": self.episodes, "epsilon": self.epsilon,
            "meta": meta if meta is not None else self.meta,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)       # atomic: a crash never leaves a half-written checkpoint

    @classmethod
    def from_checkpoint(cls, path: Path, device: str = "cpu", with_buffer: bool = False,
                        restore_optimizer: bool = False) -> "DQNAgent":
        path = Path(path)
        if not path.exists():
            raise CheckpointError(f"Checkpoint not found: {path}\nTrain first:  python -m reinforcement_learning.train")
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            for k in ("config", "policy_state", "target_state"):
                if k not in ckpt:
                    raise KeyError(k)
            d = dict(ckpt["config"])
            d["hidden"] = tuple(d["hidden"])
            d["device"] = device
            cfg = DQNConfig(**d)
        except Exception as exc:
            raise CheckpointError(f"Cannot read checkpoint {path}: {exc}") from exc
        if cfg.state_dim != STATE_DIM or cfg.action_dim != ACTION_DIM:
            raise CheckpointError(f"Checkpoint dims ({cfg.state_dim}->{cfg.action_dim}) do not match "
                                  f"the project ({STATE_DIM}->{ACTION_DIM}).")
        agent = cls(cfg, with_buffer=with_buffer)
        try:
            agent.policy.load_state_dict(ckpt["policy_state"])
            agent.target.load_state_dict(ckpt["target_state"])
            if restore_optimizer and "optimizer_state" in ckpt:
                agent.optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception as exc:
            raise CheckpointError(f"Checkpoint weights incompatible with network: {exc}") from exc
        agent.env_steps = int(ckpt.get("env_steps", 0))
        agent.learn_steps = int(ckpt.get("learn_steps", 0))
        agent.episodes = int(ckpt.get("episodes", 0))
        agent.meta = dict(ckpt.get("meta", {}))
        agent.policy.eval()
        return agent


def read_checkpoint_meta(path: Path) -> dict:
    """Light-weight info for the dashboard (config, steps, epsilon, training meta)."""
    a = DQNAgent.from_checkpoint(path, with_buffer=False)
    p = Path(path)
    return {"path": str(p), "size_kb": round(p.stat().st_size / 1024, 1),
            "modified": p.stat().st_mtime, "episodes": a.episodes, "env_steps": a.env_steps,
            "learn_steps": a.learn_steps, "epsilon": a.epsilon,
            "config": dataclasses.asdict(a.config), "meta": a.meta}


# --------------------------------------------------------------------------- #
def _selftest() -> int:
    """Checks the Bellman/learning machinery on a synthetic problem (gamma=0 contextual bandit):
    when s[0]==0 action 1 pays +1, when s[0]==1 action 2 pays +1, anything else pays -1."""
    cfg = DQNConfig(gamma=0.0, lr=5e-3, batch_size=32, buffer_size=2000, learn_start=64,
                    eps_decay_steps=1, seed=0)
    agent = DQNAgent(cfg)
    rng = np.random.default_rng(0)
    ones = np.ones(cfg.action_dim, dtype=bool)
    for _ in range(1500):
        k = int(rng.integers(2))
        s = np.zeros(cfg.state_dim, dtype=np.float32)
        s[0] = float(k)
        a = int(rng.integers(cfg.action_dim))
        agent.remember(s, a, 1.0 if a == (2 if k else 1) else -1.0, s, ones, False)
    first = [agent.learn()["loss"] for _ in range(20)]
    for _ in range(600):
        last = agent.learn()["loss"]
    ok = True
    for k, best in ((0, 1), (1, 2)):
        s = np.zeros(cfg.state_dim, dtype=np.float32)
        s[0] = float(k)
        chosen = agent.select_action(s, greedy=True)
        print(f"state type {k}: greedy action {chosen} (expected {best})")
        ok &= chosen == best
    masked = agent.select_action(np.zeros(cfg.state_dim, dtype=np.float32),
                                 mask=np.array([True, False, False, False]), greedy=True)
    ok &= masked == 0
    print(f"loss first-20 mean {np.mean(first):.4f} -> last {last:.4f}; masked action {masked} (expected 0)")
    print("[ OK ] DQN learning rule self-test passed" if ok else "[FAIL] DQN self-test failed")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print("Use --selftest to verify the learning rule, or python -m reinforcement_learning.train")