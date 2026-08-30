"""
Position-Aware KV Transplant Adapter (Fix 4)
Extends the ResidualMLPAdapter to accept position encoding as additional input,
allowing the network to learn position-dependent RoPE corrections that account
for the 64-dim vs 128-dim frequency mismatch between 1B and 3B models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionAwareAdapter(nn.Module):
    """
    Combines analytical Ridge projection with a position-aware nonlinear delta MLP.
    Y = Linear_base(X) + scale * MLP_delta(concat(X, pos_encoding))

    The position encoding captures where each token sits in the sequence,
    allowing the adapter to learn position-dependent corrections for the
    RoPE frequency mismatch between 1B (head_dim=64) and 3B (head_dim=128).
    """
    def __init__(
        self,
        in_dim: int = 1536,
        hidden_dim: int = 1024,
        out_dim: int = 1024,
        pos_dim: int = 128,
        W_ridge: Optional[torch.Tensor] = None,
        b_ridge: Optional[torch.Tensor] = None,
        scale: float = 0.05,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pos_dim = pos_dim
        self.scale = scale

        # Base linear path (Ridge-initialized, position-independent)
        self.base_linear = nn.Linear(in_dim, out_dim)
        if W_ridge is not None and b_ridge is not None:
            with torch.no_grad():
                self.base_linear.weight.copy_(W_ridge.T)
                self.base_linear.bias.copy_(b_ridge)

        # Position-aware nonlinear delta path
        # Input: [KV_features (1536D) || position_encoding (128D)] = 1664D
        delta_in_dim = in_dim + pos_dim
        self.fc1 = nn.Linear(delta_in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        # Zero-initialize delta output for stable residual refinement
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        # Learnable position frequency basis (128D sinusoidal-like)
        self.pos_proj = nn.Linear(1, pos_dim, bias=False)
        self._init_pos_proj()

    def _init_pos_proj(self):
        """Initialize position projection with sinusoidal frequencies."""
        with torch.no_grad():
            freqs = torch.exp(torch.arange(0, self.pos_dim, dtype=torch.float32) * 
                            -(torch.log(torch.tensor(10000.0)) / self.pos_dim))
            self.pos_proj.weight.copy_(freqs.unsqueeze(1))

    def encode_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Creates position encoding for a sequence of tokens.
        Returns: (seq_len, pos_dim)
        """
        positions = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        raw = self.pos_proj(positions)  # (seq_len, pos_dim)
        # Interleave sin and cos
        pe = torch.zeros_like(raw)
        pe[:, 0::2] = torch.sin(raw[:, 0::2])
        pe[:, 1::2] = torch.cos(raw[:, 1::2])
        return pe

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) KV features
            positions: (N, pos_dim) position encodings. If None, generates sequential positions.
        """
        base = self.base_linear(x)

        if positions is None:
            positions = self.encode_positions(x.shape[0], x.device)

        x_pos = torch.cat([x, positions], dim=-1)  # (N, in_dim + pos_dim)
        delta = self.fc2(self.act(self.norm1(self.fc1(x_pos))))
        return base + self.scale * delta


def train_position_aware_adapter(
    X: torch.Tensor,
    Y: torch.Tensor,
    device: torch.device,
    in_dim: int = 1536,
    out_dim: int = 1024,
    pos_dim: int = 128,
    epochs: int = 80,
    lr: float = 1e-3,
):
    """Trains a position-aware adapter with Ridge initialization."""
    from train_mlp_mapper import compute_ridge_weights, compute_metrics

    W_ridge, b_ridge = compute_ridge_weights(X, Y, alpha=1.0)

    adapter = PositionAwareAdapter(
        in_dim=in_dim, hidden_dim=out_dim, out_dim=out_dim, pos_dim=pos_dim,
        W_ridge=W_ridge, b_ridge=b_ridge, scale=0.05,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for n, p in adapter.named_parameters() if not n.startswith("base_linear")],
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    X_dev = X.to(device)
    Y_dev = Y.to(device)

    adapter.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        # Recompute positions each iteration (pos_proj is learnable, graph must be fresh)
        positions = adapter.encode_positions(X.shape[0], device)
        pred = adapter(X_dev, positions)
        loss_mse = F.mse_loss(pred, Y_dev)
        loss_cos = 1.0 - F.cosine_similarity(pred, Y_dev, dim=-1).mean()
        total_loss = loss_mse + 0.1 * loss_cos
        total_loss.backward()
        optimizer.step()
        scheduler.step()

    adapter.eval()
    with torch.no_grad():
        positions = adapter.encode_positions(X.shape[0], device)
        final_pred = adapter(X_dev, positions)
        r2, cos_sim, mse = compute_metrics(Y_dev, final_pred)

    return adapter, r2, cos_sim, mse
