"""Does using more 1B layers per 3B layer actually help? (top_k ablation)

The shipped projector concatenates top_k=3 source layers from the 1B model to
predict one 3B layer's KV. Nobody ablated k, so it is unknown whether 3 is doing
anything that 1 would not.

In-sample R^2 is the wrong metric for this comparison and is reported here only
alongside the held-out figure. More source layers means more input columns for a
least-squares fit, so in-sample error can only fall as top_k rises -- an ablation
on that number draws a clean upward line that says nothing about generalisation.
Both are measured, and the gap between them is the point.

The split is BY PROMPT, never by token. Tokens inside one prompt share a context
and their KV states are strongly correlated, so a token-level shuffle would put
near-duplicates on both sides and report a held-out score that is really an
in-sample one. 40 prompts fit, 10 held out.

Ridge only, no MLP delta: the project's own README rejects the neural adapter
(+0.15% accuracy for 4x the latency) and ships the linear path, so the linear path
is what matters.

Usage:
  ./.venv/bin/python ablate_topk.py
"""

from __future__ import annotations

import json
import sys

import torch

from train_mlp_mapper import compute_ridge_weights, get_topk_source_layers

DATA = "data/calibration_kv_pairs.pt"
KS = [1, 2, 3, 4, 5]
N_TEST_PROMPTS = 10
SEED = 0


def r2(Y, P):
    ss_res = torch.sum((Y - P) ** 2).item()
    ss_tot = torch.sum((Y - Y.mean(dim=0, keepdim=True)) ** 2).item()
    return 1.0 - ss_res / (ss_tot + 1e-10)


def flatten(data, n_layers, key, idxs):
    """{layer: [tokens, heads*head_dim]} for the given prompt indices."""
    out = {}
    for l in range(n_layers):
        rows = []
        for p in idxs:
            t = data[p]["layers"][l][key].squeeze(0)          # [heads, seq, dim]
            s = data[p]["seq_len"]
            rows.append(t.permute(1, 0, 2).contiguous().view(s, -1).float())
        out[l] = torch.cat(rows, dim=0)
    return out


def main():
    ds = torch.load(DATA, map_location="cpu", weights_only=False)
    n1, n3 = ds["num_layers_1b"], ds["num_layers_3b"]
    n_prompts = len(ds["data_1b"])
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n_prompts, generator=g).tolist()
    te_idx, tr_idx = perm[:N_TEST_PROMPTS], perm[N_TEST_PROMPTS:]
    print(f"{n_prompts} prompts -> {len(tr_idx)} fit / {len(te_idx)} held out "
          f"(split by prompt, seed {SEED})")

    src = {}
    for name, data, n in [("1b", ds["data_1b"], n1), ("3b", ds["data_3b"], n3)]:
        for key, tag in [("k_clean", "k"), ("v", "v")]:
            src[f"{name}_{tag}_tr"] = flatten(data, n, key, tr_idx)
            src[f"{name}_{tag}_te"] = flatten(data, n, key, te_idx)
    ntok_tr = src["1b_k_tr"][0].shape[0]
    ntok_te = src["1b_k_te"][0].shape[0]
    print(f"tokens: {ntok_tr} fit, {ntok_te} held out\n")

    rows = []
    for k in KS:
        acc = {"k_in": [], "k_out": [], "v_in": [], "v_out": []}
        for j in range(n3):
            layers = get_topk_source_layers(j, n3, n1, k=k)
            for tag in ("k", "v"):
                Xtr = torch.cat([src[f"1b_{tag}_tr"][l] for l in layers], dim=-1)
                Xte = torch.cat([src[f"1b_{tag}_te"][l] for l in layers], dim=-1)
                Ytr, Yte = src[f"3b_{tag}_tr"][j], src[f"3b_{tag}_te"][j]
                W, b = compute_ridge_weights(Xtr, Ytr, alpha=1.0)
                acc[f"{tag}_in"].append(r2(Ytr, Xtr @ W + b))
                acc[f"{tag}_out"].append(r2(Yte, Xte @ W + b))
        m = {a: sum(v) / len(v) for a, v in acc.items()}
        rows.append({"top_k": k, "in_dim": k * ds["num_kv_heads"] * ds["head_dim_1b"], **m})
        print(f"  top_k={k} done", flush=True)

    print("\n" + "=" * 76)
    print("TOP-K ABLATION: mean R^2 over 28 target layers")
    print("=" * 76)
    print(f"{'top_k':>6}{'in_dim':>8}{'KEYS in-samp':>14}{'KEYS held-out':>15}"
          f"{'VALS in-samp':>14}{'VALS held-out':>15}")
    print("-" * 76)
    for r in rows:
        print(f"{r['top_k']:>6}{r['in_dim']:>8}{r['k_in']*100:>13.2f}%{r['k_out']*100:>14.2f}%"
              f"{r['v_in']*100:>13.2f}%{r['v_out']*100:>14.2f}%")
    print("=" * 76)

    base = next(r for r in rows if r["top_k"] == 1)
    ship = next((r for r in rows if r["top_k"] == 3), None)
    if ship:
        print(f"\nshipped top_k=3 vs top_k=1, held out: "
              f"keys {(ship['k_out']-base['k_out'])*100:+.2f} pts, "
              f"values {(ship['v_out']-base['v_out'])*100:+.2f} pts")
        print(f"in-sample, the same comparison reads:               "
              f"keys {(ship['k_in']-base['k_in'])*100:+.2f} pts, "
              f"values {(ship['v_in']-base['v_in'])*100:+.2f} pts")
    best = max(rows, key=lambda r: r["k_out"] + r["v_out"])
    print(f"\nbest held-out top_k = {best['top_k']}")
    json.dump(rows, open("data/topk_ablation.json", "w"), indent=2)
    print("saved -> data/topk_ablation.json")


if __name__ == "__main__":
    main()
