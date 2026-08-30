"""
Evaluate Perplexity (PPL) Matrix for Cross-Model KV Cache Transplant
Measures Cross-Entropy Loss and Perplexity across 4 distinct domains:
1. Systems Architecture
2. Python Code & Logic
3. Scientific & Mathematical Analysis
4. General Knowledge & Reasoning
"""

import os
import math
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from rope_utils import inverse_rope
from fused_projector import FusedKVProjector


def resolve_model_id(model_id: str) -> tuple[str, dict]:
    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token:
        return model_id, {"token": hf_token}
    mirror_map = {
        "meta-llama/Llama-3.2-1B": "unsloth/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B": "unsloth/Llama-3.2-3B",
    }
    return mirror_map.get(model_id, model_id), {}


# Multi-domain evaluation texts
EVALUATION_DATASET = [
    {
        "domain": "Systems Architecture",
        "prefix": (
            "Distributed Key-Value Store Architecture and Replication Consensus.\n"
            "In high-throughput distributed database engines, linearizable consistency is maintained using Multi-Raft state machine replication. "
            "Each shard leader sequences write requests into an append-only write-ahead log (WAL) on NVMe storage, broadcasting log entries to followers."
        ),
        "target": (
            " Once a quorum of replicas acknowledge persistence, the entry is committed to the immutable memtable and flushed asynchronously into Log-Structured Merge (LSM) SSTables on disk."
        ),
    },
    {
        "domain": "Python Code & Algorithms",
        "prefix": (
            "def compute_hierarchical_kmeans(vectors: torch.Tensor, k: int = 8, max_iters: int = 100):\n"
            "    '''Performs fast GPU-accelerated K-Means clustering over high-dimensional dense embeddings.'''\n"
            "    N, D = vectors.shape\n"
            "    centroids = vectors[torch.randperm(N)[:k]].clone()\n"
        ),
        "target": (
            "    for iteration in range(max_iters):\n"
            "        distances = torch.cdist(vectors, centroids)\n"
            "        cluster_assignments = torch.argmin(distances, dim=-1)\n"
            "        new_centroids = torch.stack([vectors[cluster_assignments == j].mean(dim=0) for j in range(k)])\n"
            "        if torch.norm(new_centroids - centroids) < 1e-4:\n"
            "            break\n"
            "        centroids = new_centroids\n"
            "    return centroids, cluster_assignments"
        ),
    },
    {
        "domain": "Scientific & Mathematics",
        "prefix": (
            "Thermodynamics of Black Holes and Holographic Entropy Bounds.\n"
            "According to the Bekenstein-Hawking formula, the entropy of a Schwarzschild black hole is directly proportional to the surface area of its event horizon divided by four Planck areas."
        ),
        "target": (
            " This fundamental relationship establishes that the maximum information density contained within any spatial volume is bounded by its boundary area rather than its three-dimensional volume, laying the mathematical foundation for the holographic principle in quantum gravity."
        ),
    },
    {
        "domain": "General Reasoning & Logic",
        "prefix": (
            "Economic Game Theory: Nash Equilibrium in Oligopoly Pricing.\n"
            "In a Bertrand duopoly competition with homogeneous goods and symmetric marginal costs, two competing firms choose prices simultaneously."
        ),
        "target": (
            " The unique Nash equilibrium occurs where both firms set prices exactly equal to marginal cost, eliminating all economic profit despite the presence of only two market competitors."
        ),
    },
]


def evaluate_perplexity_matrix(
    weights_path: str = "weights/mapper.pt",
    output_path: str = "data/perplexity_results.json",
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Perplexity Eval] Acceleration Device: {device}")

    model_1b_id, auth_1b = resolve_model_id("meta-llama/Llama-3.2-1B")
    model_3b_id, auth_3b = resolve_model_id("meta-llama/Llama-3.2-3B")

    tokenizer = AutoTokenizer.from_pretrained(model_1b_id, **auth_1b)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Fused Projector
    print(f"[Perplexity Eval] Loading Fused GPU Projector from {weights_path}...")
    projector = FusedKVProjector(weights_path=weights_path, device=device)
    head_dim_1b = projector.head_dim_1b

    # Load Models
    print(f"[Perplexity Eval] Loading 1B Model ({model_1b_id})...")
    model_1b = AutoModelForCausalLM.from_pretrained(model_1b_id, torch_dtype=torch.float16, **auth_1b).to(device).eval()

    print(f"[Perplexity Eval] Loading 3B Model ({model_3b_id})...")
    model_3b = AutoModelForCausalLM.from_pretrained(model_3b_id, torch_dtype=torch.float16, **auth_3b).to(device).eval()

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()

    results = []

    print("\n" + "=" * 90)
    print(f"{'Evaluation Domain':<26} | {'Native 3B PPL':<15} | {'Transplant 3B PPL':<18} | {'Delta PPL':<12} | {'Parity %':<10}")
    print("-" * 90)

    for item in EVALUATION_DATASET:
        domain = item["domain"]
        prefix_text = item["prefix"]
        target_text = item["target"]

        prefix_ids = tokenizer(prefix_text, return_tensors="pt")["input_ids"].to(device)
        target_ids = tokenizer(target_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

        full_ids = torch.cat([prefix_ids, target_ids], dim=-1)
        prefix_len = prefix_ids.shape[1]
        target_len = target_ids.shape[1]

        # -------------------------------------------------------------
        # 1. Native 3B Perplexity
        # -------------------------------------------------------------
        sync()
        with torch.no_grad():
            out_native = model_3b(full_ids)
            # Logits for target tokens: positions prefix_len-1 to full_len-2 predict prefix_len to full_len-1
            logits_native = out_native.logits[:, prefix_len - 1 : -1, :]  # (1, target_len, vocab_size)
            loss_native = F.cross_entropy(logits_native.view(-1, logits_native.shape[-1]), target_ids.view(-1)).item()
            ppl_native = math.exp(loss_native)

        # -------------------------------------------------------------
        # 2. Transplanted 3B Perplexity (Prefill with 1B, transplant, then evaluate target on 3B)
        # -------------------------------------------------------------
        sync()
        with torch.no_grad():
            out_1b_prefix = model_1b(prefix_ids, use_cache=True)
            past_kv_1b = out_1b_prefix.past_key_values

            if hasattr(past_kv_1b, "layers"):
                raw_keys = [getattr(l, "keys", getattr(l, "key_states", None)) for l in past_kv_1b.layers]
                raw_vals = [getattr(l, "values", getattr(l, "value_states", None)) for l in past_kv_1b.layers]
            elif hasattr(past_kv_1b, "key_cache"):
                raw_keys = past_kv_1b.key_cache
                raw_vals = past_kv_1b.value_cache
            else:
                raw_keys = [l[0] for l in past_kv_1b]
                raw_vals = [l[1] for l in past_kv_1b]

            unwound_keys = [inverse_rope(k, head_dim=head_dim_1b, base=500000.0) for k in raw_keys]
            transplanted_cache = projector.project_and_build_cache(unwound_keys, raw_vals, prefix_len)

            # Pass full target sequence conditioned on transplanted cache
            # The first token is prefix_ids[:, -1:] to get the prediction for target_ids[:, 0]
            eval_input = torch.cat([prefix_ids[:, -1:], target_ids[:, :-1]], dim=-1)
            out_trans = model_3b(eval_input, past_key_values=transplanted_cache, use_cache=True)
            logits_trans = out_trans.logits  # (1, target_len, vocab_size)

            loss_trans = F.cross_entropy(logits_trans.view(-1, logits_trans.shape[-1]), target_ids.view(-1)).item()
            ppl_trans = math.exp(loss_trans)

        # -------------------------------------------------------------
        # 3. Native 1B Perplexity (Reference)
        # -------------------------------------------------------------
        with torch.no_grad():
            out_1b_full = model_1b(full_ids)
            logits_1b = out_1b_full.logits[:, prefix_len - 1 : -1, :]
            loss_1b = F.cross_entropy(logits_1b.view(-1, logits_1b.shape[-1]), target_ids.view(-1)).item()
            ppl_1b = math.exp(loss_1b)

        delta_ppl = ppl_trans - ppl_native
        parity_pct = (ppl_native / max(ppl_trans, 1e-5)) * 100.0

        entry = {
            "domain": domain,
            "prefix_tokens": prefix_len,
            "target_tokens": target_len,
            "native_3b_loss": loss_native,
            "native_3b_ppl": ppl_native,
            "transplant_3b_loss": loss_trans,
            "transplant_3b_ppl": ppl_trans,
            "native_1b_loss": loss_1b,
            "native_1b_ppl": ppl_1b,
            "delta_ppl": delta_ppl,
            "parity_percentage": parity_pct,
        }
        results.append(entry)

        print(f"{domain:<26} | {ppl_native:>15.2f} | {ppl_trans:>18.2f} | {delta_ppl:>+12.2f} | {parity_pct:>9.1f}%")

    avg_native_ppl = sum(r["native_3b_ppl"] for r in results) / len(results)
    avg_trans_ppl = sum(r["transplant_3b_ppl"] for r in results) / len(results)
    avg_1b_ppl = sum(r["native_1b_ppl"] for r in results) / len(results)
    avg_parity = sum(r["parity_percentage"] for r in results) / len(results)

    print("-" * 90)
    print(f"{'OVERALL AVERAGE':<26} | {avg_native_ppl:>15.2f} | {avg_trans_ppl:>18.2f} | {avg_trans_ppl - avg_native_ppl:>+12.2f} | {avg_parity:>9.1f}%")
    print(f"Native 1B Reference PPL: {avg_1b_ppl:.2f}")
    print("=" * 90)

    summary_payload = {
        "summary": {
            "avg_native_3b_ppl": avg_native_ppl,
            "avg_transplant_3b_ppl": avg_trans_ppl,
            "avg_native_1b_ppl": avg_1b_ppl,
            "avg_delta_ppl": avg_trans_ppl - avg_native_ppl,
            "avg_parity_percentage": avg_parity,
        },
        "domain_results": results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"[Perplexity Eval] Results saved to {output_path}")
    return summary_payload


if __name__ == "__main__":
    evaluate_perplexity_matrix()
