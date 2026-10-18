"""Q-network: MLP mapping the 17-D traffic state to one Q-value per signal phase."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from config.constants import ACTION_DIM, STATE_DIM


class QNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden: Sequence[int] = (128, 128, 64)) -> None:
        super().__init__()
        if state_dim < 1 or action_dim < 2 or not hidden or any(h < 1 for h in hidden):
            raise ValueError("Invalid network dimensions")
        self.state_dim, self.action_dim = int(state_dim), int(action_dim)
        layers, prev = [], self.state_dim
        for h in hidden:
            layers += [nn.Linear(prev, int(h)), nn.ReLU()]
            prev = int(h)
        layers.append(nn.Linear(prev, self.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.state_dim:
            raise ValueError(f"Expected input of shape (batch, {self.state_dim}), got {tuple(x.shape)}")
        return self.net(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())