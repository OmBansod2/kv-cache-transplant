"""
RoPE Utilities for Cross-Model KV Cache Transplant
Supports exact forward and inverse RoPE rotation for Llama-3.2 models.
"""

import math
import torch
from typing import Optional, Tuple


def compute_rope_cos_sin(
    seq_len: int,
    head_dim: int,
    base: float = 500000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes cos and sin tables for Rotary Position Embedding (RoPE)
    matching Hugging Face Llama-3.2 implementation.

    Args:
        seq_len: Sequence length.
        head_dim: Dimensionality of each attention head.
        base: Rotary base theta (500000.0 for Llama-3 / Llama-3.2).
        device: Target torch device.
        dtype: Desired tensor dtype.
        position_ids: Optional tensor of shape (batch_size, seq_len) or (seq_len,).

    Returns:
        cos, sin tensors broadcastable to (batch_size, num_heads, seq_len, head_dim)
    """
    dim_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (dim_range / head_dim))

    if position_ids is None:
        pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    else:
        pos = position_ids.float().to(device)
        if pos.dim() > 1:
            pos = pos[0]

    freqs = torch.outer(pos, inv_freq)  # (seq_len, head_dim // 2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)

    cos = emb.cos().to(dtype)  # (seq_len, head_dim)
    sin = emb.sin().to(dtype)  # (seq_len, head_dim)

    # Reshape for broadcasting with (batch, num_heads, seq_len, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)

    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input tensor."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    key_tensor: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    head_dim: Optional[int] = None,
    base: float = 500000.0,
) -> torch.Tensor:
    """
    Applies RoPE rotation to unrotated key vectors.

    Args:
        key_tensor: Tensor of shape (batch, num_heads, seq_len, head_dim)
        position_ids: Optional position IDs tensor.
        head_dim: Head dimension (inferred from tensor if None).
        base: RoPE base theta.

    Returns:
        Rotated key tensor of same shape and dtype.
    """
    b, h, seq_len, d = key_tensor.shape
    if head_dim is None:
        head_dim = d

    cos, sin = compute_rope_cos_sin(
        seq_len=seq_len,
        head_dim=head_dim,
        base=base,
        device=key_tensor.device,
        dtype=key_tensor.dtype,
        position_ids=position_ids,
    )

    # Standard RoPE: (x * cos) + (rotate_half(x) * sin)
    return (key_tensor * cos) + (rotate_half(key_tensor) * sin)


def inverse_rope(
    key_tensor: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    head_dim: Optional[int] = None,
    base: float = 500000.0,
) -> torch.Tensor:
    """
    Computationally unwinds RoPE rotation so Keys become position-independent.
    Inverse of apply_rope: (x * cos) - (rotate_half(x) * sin)

    Args:
        key_tensor: Tensor of shape (batch, num_heads, seq_len, head_dim)
        position_ids: Optional position IDs tensor.
        head_dim: Head dimension (inferred from tensor if None).
        base: RoPE base theta.

    Returns:
        Unwound, position-independent key tensor of same shape and dtype.
    """
    b, h, seq_len, d = key_tensor.shape
    if head_dim is None:
        head_dim = d

    cos, sin = compute_rope_cos_sin(
        seq_len=seq_len,
        head_dim=head_dim,
        base=base,
        device=key_tensor.device,
        dtype=key_tensor.dtype,
        position_ids=position_ids,
    )

    # Inverse RoPE rotation: (x * cos) - (rotate_half(x) * sin)
    return (key_tensor * cos) - (rotate_half(key_tensor) * sin)


def test_rope_invertibility():
    """Unit test to verify exact round-trip invertibility."""
    torch.manual_seed(42)
    for head_dim in [64, 128]:
        for seq_len in [16, 128, 512]:
            x = torch.randn(1, 8, seq_len, head_dim, dtype=torch.float32)
            x_rot = apply_rope(x, head_dim=head_dim)
            x_rec = inverse_rope(x_rot, head_dim=head_dim)
            diff = (x - x_rec).abs().max().item()
            assert diff < 1e-5, f"Invertibility test failed: head_dim={head_dim}, seq_len={seq_len}, max_diff={diff}"
    print("All RoPE invertibility unit tests passed successfully!")


if __name__ == "__main__":
    test_rope_invertibility()
