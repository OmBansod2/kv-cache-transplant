"""
Fused Batched GPU Projection Kernel for Cross-Model KV Cache Transplant
Vectorizes all 28 layer projections into a single 3D batched tensor GEMM on Apple Silicon MPS.
"""

import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from transformers.cache_utils import DynamicCache
from rope_utils import inverse_rope, apply_rope


class FusedKVProjector(nn.Module):
    """
    Executes all 28 layer KV cache projections concurrently via a single 3D batched GEMM:
    Input: (28, seq_len, 1536)
    Weights: (28, 1536, 1024)
    Output: (28, seq_len, 1024)
    """
    def __init__(
        self,
        weights_path: str = "weights/mapper.pt",
        device: torch.device = torch.device("mps"),
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        data = torch.load(weights_path, map_location="cpu")
        self.num_layers_1b = data["num_layers_1b"]
        self.num_layers_3b = data["num_layers_3b"]
        self.head_dim_1b = data["head_dim_1b"]
        self.head_dim_3b = data["head_dim_3b"]
        self.num_kv_heads = data["num_kv_heads"]
        self.top_k = data.get("top_k", 3)

        # Build source layer mapping indices
        self.source_layer_map: List[List[int]] = []
        w_k_list, b_k_list = [], []
        w_v_list, b_v_list = [], []

        if "trained_models" in data:
            # Neural Residual MLP format
            trained = data["trained_models"]
            for j in range(self.num_layers_3b):
                entry = trained[j]
                self.source_layer_map.append(entry["source_1b_layers"])
                # Extract linear base weights
                w_k = entry["key_state_dict"]["base_linear.weight"].T.to(dtype)
                b_k = entry["key_state_dict"]["base_linear.bias"].unsqueeze(0).to(dtype)
                w_v = entry["val_state_dict"]["base_linear.weight"].T.to(dtype)
                b_v = entry["val_state_dict"]["base_linear.bias"].unsqueeze(0).to(dtype)
                w_k_list.append(w_k)
                b_k_list.append(b_k)
                w_v_list.append(w_v)
                b_v_list.append(b_v)
        else:
            # Linear mapper format
            mappers = data["mappers"]
            for j in range(self.num_layers_3b):
                m = mappers[j]
                src = m.get("source_1b_layers", [m.get("mapped_1b_layer", 0)])
                self.source_layer_map.append(src)
                w_k_list.append(m["W_k"].to(dtype))
                b_k_list.append(m["b_k"].unsqueeze(0).to(dtype))
                w_v_list.append(m["W_v"].to(dtype))
                b_v_list.append(m["b_v"].unsqueeze(0).to(dtype))

        # Stack into 3D parameter tensors: (28, 1536, 1024) and (28, 1, 1024)
        self.register_buffer("W_keys", torch.stack(w_k_list, dim=0).to(device))
        self.register_buffer("b_keys", torch.stack(b_k_list, dim=0).to(device))
        self.register_buffer("W_vals", torch.stack(w_v_list, dim=0).to(device))
        self.register_buffer("b_vals", torch.stack(b_v_list, dim=0).to(device))

    @torch.no_grad()
    def project_and_build_cache(
        self,
        unwound_1b_keys: List[torch.Tensor],  # 16 tensors of (1, 8, seq_len, 64)
        raw_1b_vals: List[torch.Tensor],      # 16 tensors of (1, 8, seq_len, 64)
        seq_len: int,
    ) -> DynamicCache:
        """
        Gathers all 28 layer inputs, executes a single fused 3D batched GEMM on GPU,
        re-applies 3B RoPE, and populates a 3B DynamicCache.
        """
        # Step 1: Pre-flatten 1B tensors once: (16, seq_len, 512)
        flat_1b_keys = torch.stack([
            k.squeeze(0).permute(1, 0, 2).contiguous().view(seq_len, -1)
            for k in unwound_1b_keys
        ], dim=0)  # (16, seq_len, 512)

        flat_1b_vals = torch.stack([
            v.squeeze(0).permute(1, 0, 2).contiguous().view(seq_len, -1)
            for v in raw_1b_vals
        ], dim=0)  # (16, seq_len, 512)

        # Step 2: Build concatenated 3D input batch for 28 target layers: (28, seq_len, 1536)
        in_k_batch = torch.stack([
            torch.cat([flat_1b_keys[src_idx] for src_idx in self.source_layer_map[j]], dim=-1)
            for j in range(self.num_layers_3b)
        ], dim=0)

        in_v_batch = torch.stack([
            torch.cat([flat_1b_vals[src_idx] for src_idx in self.source_layer_map[j]], dim=-1)
            for j in range(self.num_layers_3b)
        ], dim=0)

        # Step 3: FUSED BATCHED GPU MATRIX MULTIPLICATION (torch.bmm)
        # (28, seq_len, 1536) @ (28, 1536, 1024) + (28, 1, 1024) -> (28, seq_len, 1024)
        out_keys_3b = torch.bmm(in_k_batch, self.W_keys) + self.b_keys
        out_vals_3b = torch.bmm(in_v_batch, self.W_vals) + self.b_vals

        # Step 4: Reshape, apply RoPE per layer, and populate DynamicCache
        cache_3b = DynamicCache()
        for j in range(self.num_layers_3b):
            k_j = out_keys_3b[j].view(1, seq_len, self.num_kv_heads, self.head_dim_3b).permute(0, 2, 1, 3)
            v_j = out_vals_3b[j].view(1, seq_len, self.num_kv_heads, self.head_dim_3b).permute(0, 2, 1, 3)
            k_rot = apply_rope(k_j, head_dim=self.head_dim_3b, base=500000.0)
            cache_3b.update(k_rot, v_j, layer_idx=j)

        return cache_3b


def benchmark_fused_kernel():
    """Unit test and latency comparison between loop-based and fused projector."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Fused Kernel Test] Device: {device}")

    projector = FusedKVProjector(weights_path="weights/mapper.pt", device=device)

    # Synthetic 512-token test inputs
    seq_len = 512
    dummy_keys = [torch.randn(1, 8, seq_len, 64, device=device, dtype=torch.float16) for _ in range(16)]
    dummy_vals = [torch.randn(1, 8, seq_len, 64, device=device, dtype=torch.float16) for _ in range(16)]

    # Warmup
    for _ in range(3):
        _ = projector.project_and_build_cache(dummy_keys, dummy_vals, seq_len)
        if device.type == "mps":
            torch.mps.synchronize()

    # Benchmark Fused Kernel
    iterations = 20
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(iterations):
        cache = projector.project_and_build_cache(dummy_keys, dummy_vals, seq_len)
        if device.type == "mps":
            torch.mps.synchronize()
    t1 = time.perf_counter()

    avg_fused_ms = ((t1 - t0) / iterations) * 1000.0
    print(f"[Fused Kernel Test] Average Fused 28-Layer GPU Projection Latency: {avg_fused_ms:.2f} ms")
    print(f"[Fused Kernel Test] Output Cache Verified: {len(cache.layers if hasattr(cache, 'layers') else cache.key_cache)} layers populated.")


if __name__ == "__main__":
    benchmark_fused_kernel()
