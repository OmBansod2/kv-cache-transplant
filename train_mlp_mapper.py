"""
Train Neural Residual MLP Mappers for Cross-Model KV Cache Transplant (v2 Architecture)
Combines analytical Ridge projection with learned nonlinear delta corrections on Apple Silicon MPS.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from kv_adapter import ResidualMLPAdapter


def compute_ridge_weights(
    X: torch.Tensor,
    Y: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes closed-form Ridge weights W and bias b."""
    mu_x = X.mean(dim=0, keepdim=True)
    mu_y = Y.mean(dim=0, keepdim=True)
    X_c = X - mu_x
    Y_c = Y - mu_y

    XtX = torch.matmul(X_c.T, X_c)
    reg_matrix = alpha * torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
    XtY = torch.matmul(X_c.T, Y_c)

    W = torch.linalg.solve(XtX + reg_matrix, XtY)
    b = (mu_y - torch.matmul(mu_x, W)).squeeze(0)
    return W, b


def get_topk_source_layers(
    target_layer_idx: int,
    num_layers_target: int = 28,
    num_layers_source: int = 16,
    k: int = 3,
) -> List[int]:
    """Selects top-k distinct source layers centered around proportional depth."""
    center = int(round(target_layer_idx * (num_layers_source - 1) / (num_layers_target - 1)))
    if k == 1:
        return [center]
    half_k = k // 2
    start = max(0, center - half_k)
    end = start + k
    if end > num_layers_source:
        end = num_layers_source
        start = max(0, end - k)
    return list(range(start, end))


def compute_metrics(Y_true: torch.Tensor, Y_pred: torch.Tensor) -> tuple[float, float, float]:
    """Computes R^2 score, Mean Cosine Similarity, and MSE loss."""
    ss_res = torch.sum((Y_true - Y_pred) ** 2).item()
    mu_y = Y_true.mean(dim=0, keepdim=True)
    ss_tot = torch.sum((Y_true - mu_y) ** 2).item()
    r2 = 1.0 - (ss_res / (ss_tot + 1e-10))

    cos_sim = F.cosine_similarity(Y_true, Y_pred, dim=-1).mean().item()
    mse = F.mse_loss(Y_pred, Y_true).item()
    return r2, cos_sim, mse


def train_adapter_for_layer(
    X: torch.Tensor,
    Y: torch.Tensor,
    device: torch.device,
    in_dim: int = 1536,
    out_dim: int = 1024,
    epochs: int = 80,
    lr: float = 1e-3,
) -> tuple[ResidualMLPAdapter, float, float, float]:
    """Initializes with closed-form Ridge and refines with nonlinear delta MLP."""
    # 1. Closed-form Ridge baseline
    W_ridge, b_ridge = compute_ridge_weights(X, Y, alpha=1.0)
    
    # 2. Residual MLP Adapter
    adapter = ResidualMLPAdapter(
        in_dim=in_dim,
        hidden_dim=out_dim,
        out_dim=out_dim,
        W_ridge=W_ridge,
        b_ridge=b_ridge,
        scale=0.05,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for n, p in adapter.named_parameters() if not n.startswith("base_linear")],
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    X_dev = X.to(device)
    Y_dev = Y.to(device)

    adapter.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = adapter(X_dev)
        
        loss_mse = F.mse_loss(pred, Y_dev)
        loss_cos = 1.0 - F.cosine_similarity(pred, Y_dev, dim=-1).mean()
        total_loss = loss_mse + 0.1 * loss_cos

        total_loss.backward()
        optimizer.step()
        scheduler.step()

    adapter.eval()
    with torch.no_grad():
        final_pred = adapter(X_dev)
        r2, cos_sim, mse = compute_metrics(Y_dev, final_pred)

    return adapter, r2, cos_sim, mse


def train_neural_mappers(
    data_path: str = "data/calibration_kv_pairs.pt",
    weights_path: str = "weights/mapper_mlp_v2.pt",
    top_k: int = 3,
    epochs: int = 80,
) -> Dict:
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Train Residual MLP] Acceleration Device: {device}")
    print(f"[Train Residual MLP] Loading dataset from {data_path}...")
    dataset = torch.load(data_path, map_location="cpu")

    num_layers_1b = dataset["num_layers_1b"]
    num_layers_3b = dataset["num_layers_3b"]
    data_1b = dataset["data_1b"]
    data_3b = dataset["data_3b"]
    num_prompts = len(data_1b)

    in_dim = top_k * dataset["num_kv_heads"] * dataset["head_dim_1b"]  # 3 * 8 * 64 = 1536
    out_dim = dataset["num_kv_heads"] * dataset["head_dim_3b"]         # 8 * 128 = 1024

    print(f"[Train Residual MLP] Architecture: Residual MLP (Ridge Base + Nonlinear Delta) ({in_dim}D -> {out_dim}D)")

    tokens_1b_k = {l: [] for l in range(num_layers_1b)}
    tokens_1b_v = {l: [] for l in range(num_layers_1b)}
    tokens_3b_k = {l: [] for l in range(num_layers_3b)}
    tokens_3b_v = {l: [] for l in range(num_layers_3b)}

    for p_idx in range(num_prompts):
        seq_len = data_1b[p_idx]["seq_len"]
        for l in range(num_layers_1b):
            k_clean = data_1b[p_idx]["layers"][l]["k_clean"].squeeze(0)
            v = data_1b[p_idx]["layers"][l]["v"].squeeze(0)
            tokens_1b_k[l].append(k_clean.permute(1, 0, 2).contiguous().view(seq_len, -1).float())
            tokens_1b_v[l].append(v.permute(1, 0, 2).contiguous().view(seq_len, -1).float())

        for l in range(num_layers_3b):
            k_clean = data_3b[p_idx]["layers"][l]["k_clean"].squeeze(0)
            v = data_3b[p_idx]["layers"][l]["v"].squeeze(0)
            tokens_3b_k[l].append(k_clean.permute(1, 0, 2).contiguous().view(seq_len, -1).float())
            tokens_3b_v[l].append(v.permute(1, 0, 2).contiguous().view(seq_len, -1).float())

    flat_1b_k = {l: torch.cat(tokens_1b_k[l], dim=0) for l in range(num_layers_1b)}
    flat_1b_v = {l: torch.cat(tokens_1b_v[l], dim=0) for l in range(num_layers_1b)}
    flat_3b_k = {l: torch.cat(tokens_3b_k[l], dim=0) for l in range(num_layers_3b)}
    flat_3b_v = {l: torch.cat(tokens_3b_v[l], dim=0) for l in range(num_layers_3b)}

    trained_models = {}
    metrics = {
        "layer_3b": [],
        "source_layers_1b": [],
        "r2_k": [],
        "r2_v": [],
        "cos_k": [],
        "cos_v": [],
    }

    t0 = time.perf_counter()
    print("\n[Train Residual MLP] Training layer-wise Residual MLP adapters on MPS...")
    print(f"{'3B Layer':<9} | {'Source 1B':<16} | {'Key R^2':<10} | {'Key Cos':<10} | {'Val R^2':<10} | {'Val Cos':<10}")
    print("-" * 75)

    for j in range(num_layers_3b):
        source_layers = get_topk_source_layers(j, num_layers_3b, num_layers_1b, k=top_k)
        
        X_k = torch.cat([flat_1b_k[l] for l in source_layers], dim=-1)
        X_v = torch.cat([flat_1b_v[l] for l in source_layers], dim=-1)
        Y_k = flat_3b_k[j]
        Y_v = flat_3b_v[j]

        # Train Key & Value Adapters
        k_adapter, r2_k, cos_k, _ = train_adapter_for_layer(X_k, Y_k, device, in_dim, out_dim, epochs)
        v_adapter, r2_v, cos_v, _ = train_adapter_for_layer(X_v, Y_v, device, in_dim, out_dim, epochs)

        trained_models[j] = {
            "source_1b_layers": source_layers,
            "key_state_dict": {k: v.half().cpu() for k, v in k_adapter.state_dict().items()},
            "val_state_dict": {k: v.half().cpu() for k, v in v_adapter.state_dict().items()},
            "r2_k": r2_k,
            "r2_v": r2_v,
            "cos_k": cos_k,
            "cos_v": cos_v,
        }

        metrics["layer_3b"].append(j)
        metrics["source_layers_1b"].append(source_layers)
        metrics["r2_k"].append(r2_k)
        metrics["r2_v"].append(r2_v)
        metrics["cos_k"].append(cos_k)
        metrics["cos_v"].append(cos_v)

        src_str = ",".join(f"L{l:02d}" for l in source_layers)
        print(f"Layer {j:02d}   | [{src_str:<13}] | {r2_k:.4f}     | {cos_k:.4f}     | {r2_v:.4f}     | {cos_v:.4f}")

    train_time = time.perf_counter() - t0
    mean_r2_k = np.mean(metrics["r2_k"])
    mean_r2_v = np.mean(metrics["r2_v"])
    mean_cos_k = np.mean(metrics["cos_k"])
    mean_cos_v = np.mean(metrics["cos_v"])

    print("-" * 75)
    print(f"[Train Residual MLP] Completed 28 Layers in {train_time:.2f}s!")
    print(f"[Train Residual MLP] Average Key R^2:   {mean_r2_k:.4f} (Cosine: {mean_cos_k:.4f})")
    print(f"[Train Residual MLP] Average Value R^2: {mean_r2_v:.4f} (Cosine: {mean_cos_v:.4f})")

    payload = {
        "version": "v2_residual_mlp",
        "top_k": top_k,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "trained_models": trained_models,
        "metrics": metrics,
        "num_layers_1b": num_layers_1b,
        "num_layers_3b": num_layers_3b,
        "head_dim_1b": dataset["head_dim_1b"],
        "head_dim_3b": dataset["head_dim_3b"],
        "num_kv_heads": dataset["num_kv_heads"],
    }

    print(f"[Train Residual MLP] Saving weights to {weights_path}...")
    torch.save(payload, weights_path)
    torch.save(payload, "weights/mapper.pt")
    print("[Train Residual MLP] Done!")
    return payload


if __name__ == "__main__":
    train_neural_mappers()
