"""Reward function (see formulation in the Part 4 text). Weights are hyper-parameters, NOT tuned."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np

from config.constants import APPROACHES, QUEUE_NORM_VEH, WAITING_NORM_S

ARRIVAL_NORM_VEH_PER_S = 0.5   # arrivals/s regarded as "very high" throughput


@dataclass(frozen=True)
class RewardWeights:
    waiting: float = 1.0
    queue: float = 0.5
    throughput: float = 0.3
    switch: float = 0.05
    fairness: float = 0.3

    def validate(self) -> None:
        if any(v < 0 for v in asdict(self).values()):
            raise ValueError("Reward weights must be >= 0")

    def to_dict(self) -> dict:
        return asdict(self)


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def compute_reward(info: Dict, weights: RewardWeights = RewardWeights(),
                   interval_s: float = 5.0) -> Tuple[float, Dict[str, float]]:
    """`info` is the dict returned by SumoEnvironment.step(). Returns (reward, components)."""
    weights.validate()
    w = _clip(info["interval_waiting_mean"] / WAITING_NORM_S, 0.0, 3.0)
    q = _clip(info["interval_queue_mean"] / QUEUE_NORM_VEH, 0.0, 3.0)
    waits = np.array([info["waiting_by_approach"][a] for a in APPROACHES], dtype=float) / WAITING_NORM_S
    f = _clip(float(np.std(waits)), 0.0, 1.0)
    s = 1.0 if info["switched"] else 0.0
    t = _clip(info["interval_arrived"] / (ARRIVAL_NORM_VEH_PER_S * max(interval_s, 1e-6)), 0.0, 2.0)
    reward = (-weights.waiting * w - weights.queue * q - weights.fairness * f
              - weights.switch * s + weights.throughput * t)
    return float(reward), {"waiting": w, "queue": q, "fairness": f, "switch": s, "throughput": t}