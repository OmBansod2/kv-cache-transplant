"""Calibrate the projector on ~100x more data, then pick alpha honestly.

Calibration size and the ridge penalty are the two variables that dominate
downstream quality: a 1,406-token set fits 1.57M parameters per layer at k=3, and
alpha is worth several times more than any architectural choice measured here.

This gives the method its best available setup: calibration from WikiText-103
TRAIN, in-domain with the WikiText-103 TEST used for evaluation. If the transplant
cannot clear its own native-1B floor under in-domain calibration with two orders of
magnitude more data, the limit is not data.

Memory trick: ridge only needs X'X and X'Y, so those are accumulated in a
streaming pass and the tokens are thrown away. Calibration size is then unbounded
in tokens while memory stays fixed at ~900 MB of Gram matrices.

A separate validation chunk is held out from TRAIN to choose alpha. Test is never
touched here.
"""
from __future__ import annotations
import argparse, json, math, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rope_utils import inverse_rope
from train_mlp_mapper import get_topk_source_layers

M1, M3 = "unsloth/Llama-3.2-1B", "unsloth/Llama-3.2-3B"
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]


def kv(past):
    if hasattr(past, "layers"):
        return ([getattr(l, "keys", getattr(l, "key_states", None)) for l in past.layers],
                [getattr(l, "values", getattr(l, "value_states", None)) for l in past.layers])
    if hasattr(past, "key_cache"):
        return past.key_cache, past.value_cache
    return [l[0] for l in past], [l[1] for l in past]


def states(m1, m3, ids, hd1):
    """Per-token KV for one window: {('1b'|'3b', 'k'|'v'): {layer: [T, D]}}"""
    with torch.no_grad():
        k1, v1 = kv(m1(ids, use_cache=True).past_key_values)
        k3, v3 = kv(m3(ids, use_cache=True).past_key_values)
    k1 = [inverse_rope(k, head_dim=hd1, base=500000.0) for k in k1]
    k3 = [inverse_rope(k, head_dim=k3[0].shape[-1], base=500000.0) for k in k3]
    out = {}
    for tag, src in [(("1b", "k"), k1), (("1b", "v"), v1), (("3b", "k"), k3), (("3b", "v"), v3)]:
        out[tag] = {l: t.squeeze(0).permute(1, 0, 2).contiguous()
                     .view(t.shape[2], -1).float().cpu() for l, t in enumerate(src)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=200_000)
    ap.add_argument("--val-tokens", type=int, default=8_192)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--out", default="weights/mapper_large.pt")
    args = ap.parse_args()

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(M1)
    m1 = AutoModelForCausalLM.from_pretrained(M1, torch_dtype=torch.float16).to(dev).eval()
    m3 = AutoModelForCausalLM.from_pretrained(M3, torch_dtype=torch.float16).to(dev).eval()
    n1, n3 = m1.config.num_hidden_layers, m3.config.num_hidden_layers
    hd1 = m1.config.hidden_size // m1.config.num_attention_heads
    d1 = m1.config.num_key_value_heads * hd1
    d3 = m3.config.num_key_value_heads * (m3.config.hidden_size // m3.config.num_attention_heads)
    D = args.top_k * d1
    print(f"1B {n1}L d={d1} | 3B {n3}L d={d3} | in_dim {D}")

    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids_all = tok(text[: args.tokens * 8], return_tensors="pt")["input_ids"][0]
    W = args.window
    n_fit = args.tokens // W
    n_val = args.val_tokens // W
    print(f"calibration windows: {n_fit} fit + {n_val} val ({n_fit*W:,} / {n_val*W:,} tokens)")

    srcmap = [get_topk_source_layers(j, n3, n1, k=args.top_k) for j in range(n3)]
    G = {(j, t): torch.zeros(D, D, dtype=torch.float64) for j in range(n3) for t in "kv"}
    C = {(j, t): torch.zeros(D, d3, dtype=torch.float64) for j in range(n3) for t in "kv"}
    SX = {(j, t): torch.zeros(D, dtype=torch.float64) for j in range(n3) for t in "kv"}
    SY = {(j, t): torch.zeros(d3, dtype=torch.float64) for j in range(n3) for t in "kv"}
    N = 0

    for i in range(n_fit):
        w = ids_all[i * W:(i + 1) * W].unsqueeze(0).to(dev)
        if w.shape[1] < W: break
        S = states(m1, m3, w, hd1)
        for j in range(n3):
            for t in "kv":
                X = torch.cat([S[("1b", t)][l] for l in srcmap[j]], -1).double()
                Y = S[("3b", t)][j].double()
                G[(j, t)] += X.T @ X; C[(j, t)] += X.T @ Y
                SX[(j, t)] += X.sum(0); SY[(j, t)] += Y.sum(0)
        N += W
        if (i + 1) % 50 == 0: print(f"  accumulated {N:,} tokens", flush=True)

    val = [states(m1, m3, ids_all[(n_fit + i) * W:(n_fit + i + 1) * W].unsqueeze(0).to(dev), hd1)
           for i in range(n_val)]
    VX = {(j, t): torch.cat([torch.cat([v[("1b", t)][l] for l in srcmap[j]], -1) for v in val], 0)
          for j in range(n3) for t in "kv"}
    VY = {(j, t): torch.cat([v[("3b", t)][j] for v in val], 0) for j in range(n3) for t in "kv"}

    def solve(j, t, a):
        mx = (SX[(j, t)] / N).unsqueeze(0); my = (SY[(j, t)] / N).unsqueeze(0)
        Gc = G[(j, t)] - N * (mx.T @ mx); Cc = C[(j, t)] - N * (mx.T @ my)
        W_ = torch.linalg.solve(Gc + a * torch.eye(D, dtype=torch.float64), Cc)
        return W_, (my - mx @ W_).squeeze(0)

    def r2(Y, P):
        return 1 - torch.sum((Y - P) ** 2).item() / (torch.sum((Y - Y.mean(0, keepdim=True)) ** 2).item() + 1e-10)

    mappers, chosen, scores = [], [], {"k": [], "v": []}
    for j in range(n3):
        e = {"source_1b_layers": srcmap[j]}
        for t in "kv":
            best = (-9e9, None, None, None)
            for a in ALPHAS:
                W_, b_ = solve(j, t, a)
                s = r2(VY[(j, t)], VX[(j, t)].double() @ W_ + b_)
                if s > best[0]: best = (s, W_, b_, a)
            s, W_, b_, a = best
            e[f"W_{t}"] = W_.float().half(); e[f"b_{t}"] = b_.float().half()
            chosen.append(a); scores[t].append(s)
        mappers.append(e)
        if (j + 1) % 7 == 0: print(f"  solved layer {j+1}/{n3}", flush=True)

    torch.save({"num_layers_1b": n1, "num_layers_3b": n3, "head_dim_1b": hd1,
                "head_dim_3b": d3 // m3.config.num_key_value_heads,
                "num_kv_heads": m3.config.num_key_value_heads, "top_k": args.top_k,
                "calibration_tokens": N, "mappers": mappers}, args.out)
    import statistics
    print(f"\ncalibrated on {N:,} tokens ({N/1406:.0f}x the original 1,406)")
    print(f"validation R^2  keys {statistics.mean(scores['k'])*100:.1f}%  "
          f"values {statistics.mean(scores['v'])*100:.1f}%")
    print(f"median alpha chosen: {statistics.median(chosen):.0f}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
