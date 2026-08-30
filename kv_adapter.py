"""
Neural KV Transplant Adapter Module (v2 Architecture)
Implements a Residual MLP Adapter with analytical Ridge initialization and learned nonlinear delta correction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualMLPAdapter(nn.Module):
    """
    Combines analytical Ridge projection with a learned nonlinear delta MLP.
    Y = Linear_base(X) + scale * MLP_delta(X)
    """
    def __init__(
        self,
        in_dim: int = 1536,
        hidden_dim: int = 1024,
        out_dim: int = 1024,
        W_ridge: Optional[torch.Tensor] = None,
        b_ridge: Optional[torch.Tensor] = None,
        scale: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.scale = scale

        # Base linear path
        self.base_linear = nn.Linear(in_dim, out_dim)
        if W_ridge is not None and b_ridge is not None:
            with torch.no_grad():
                self.base_linear.weight.copy_(W_ridge.T)
                self.base_linear.bias.copy_(b_ridge)

        # Nonlinear delta path
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        # Zero-initialize the delta output layer for stable residual refinement
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_linear(x)
        delta = self.fc2(self.act(self.norm1(self.fc1(x))))
        return base + self.scale * delta
