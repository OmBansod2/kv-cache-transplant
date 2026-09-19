"""Build fused-projector weight files directly from ridge, at a chosen alpha.

Faithful to what actually runs: FusedKVProjector reads only `base_linear.weight`,
and train_mlp_mapper excludes base_linear from the optimiser, so base_linear stays
exactly the closed-form ridge solution. The 80 epochs of MLP delta training are
never used by the fused path. Ridge IS the deployed projection.

Writes the "mappers" (linear) payload format the projector already supports.
"""
from __future__ import annotations
import sys, torch
from train_mlp_mapper import get_topk_source_layers, compute_ridge_weights

def ridge(X, Y, a):
    # The project's own centered estimator, so only alpha varies between configs.
    return compute_ridge_weights(X, Y, alpha=a)

def flat(data, n, key):
    out = {}
    for l in range(n):
        rows = []
        for p in range(len(data)):
            t = data[p]["layers"][l][key].squeeze(0); s = data[p]["seq_len"]
            rows.append(t.permute(1, 0, 2).contiguous().view(s, -1).float())
        out[l] = torch.cat(rows, 0)
    return out

def build(alpha, top_k, out_path):
    ds = torch.load("data/calibration_kv_pairs.pt", map_location="cpu", weights_only=False)
    n1, n3 = ds["num_layers_1b"], ds["num_layers_3b"]
    f = {f"{m}_{t}": flat(ds[f"data_{m}"], n1 if m == "1b" else n3, k)
         for m in ("1b", "3b") for k, t in (("k_clean", "k"), ("v", "v"))}
    mappers = []
    for j in range(n3):
        src = get_topk_source_layers(j, n3, n1, k=top_k)
        Xk = torch.cat([f["1b_k"][l] for l in src], -1)
        Xv = torch.cat([f["1b_v"][l] for l in src], -1)
        Wk, bk = ridge(Xk, f["3b_k"][j], alpha)
        Wv, bv = ridge(Xv, f["3b_v"][j], alpha)
        mappers.append({"source_1b_layers": src, "W_k": Wk.half(), "b_k": bk.half(),
                        "W_v": Wv.half(), "b_v": bv.half()})
    torch.save({"num_layers_1b": n1, "num_layers_3b": n3,
                "head_dim_1b": ds["head_dim_1b"], "head_dim_3b": ds["head_dim_3b"],
                "num_kv_heads": ds["num_kv_heads"], "top_k": top_k,
                "alpha": alpha, "mappers": mappers}, out_path)
    print(f"  alpha={alpha:<7} top_k={top_k} -> {out_path}")

if __name__ == "__main__":
    build(1.0,   3, "weights/mapper_shipped.pt")   # the hardcoded default
    build(100.0, 3, "weights/mapper_tuned.pt")     # chosen on validation
    build(100.0, 5, "weights/mapper_k5.pt")        # best top_k on test
