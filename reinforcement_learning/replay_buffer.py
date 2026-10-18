"""Fixed-size circular experience replay buffer (numpy storage, seeded sampling)."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity, self.state_dim, self.action_dim = int(capacity), int(state_dim), int(action_dim)
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.term = np.zeros(capacity, dtype=np.float32)
        self.m2 = np.ones((capacity, action_dim), dtype=bool)   # legal actions in s'
        self._idx = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(self, state, action: int, reward: float, next_state, next_mask, terminal: bool) -> None:
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        next_mask = np.asarray(next_mask, dtype=bool)
        if state.shape != (self.state_dim,) or next_state.shape != (self.state_dim,):
            raise ValueError(f"State shape must be ({self.state_dim},)")
        if next_mask.shape != (self.action_dim,):
            raise ValueError(f"Mask shape must be ({self.action_dim},)")
        if not (0 <= int(action) < self.action_dim):
            raise ValueError(f"Invalid action {action}")
        if not (np.isfinite(state).all() and np.isfinite(next_state).all() and np.isfinite(reward)):
            raise ValueError("Non-finite value in transition")
        i = self._idx
        self.s[i], self.a[i], self.r[i] = state, int(action), float(reward)
        self.s2[i], self.term[i], self.m2[i] = next_state, float(bool(terminal)), next_mask
        self._idx = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cpu") -> Tuple[torch.Tensor, ...]:
        if self._size < batch_size:
            raise ValueError(f"Buffer has {self._size} < batch size {batch_size}")
        ix = self._rng.integers(0, self._size, size=batch_size)
        t = lambda arr, dt: torch.as_tensor(arr[ix], dtype=dt, device=device)  # noqa: E731
        return (t(self.s, torch.float32), t(self.a, torch.int64), t(self.r, torch.float32),
                t(self.s2, torch.float32), t(self.term, torch.float32), t(self.m2, torch.bool))