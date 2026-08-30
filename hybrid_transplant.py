"""
Hybrid Selective Layer Transplant Evaluator
Evaluates the Pareto frontier between Perplexity (PPL) preservation and Prefill speedup
by varying the transplant layer cutoff: Full (28), Hybrid (24), Hybrid (20), Native (0).
"""

import os
import time
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


def evaluate_hybrid_frontier(
    weights_path: str = "weights/mapper.pt",
    output_path: str = "data/hybrid_benchmark_results.json",
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Hybrid Eval] Target Device: {device}")

    model_1b_id, auth_1b = resolve_model_id("meta-llama/Llama-3.2-1B")
    model_3b_id, auth_3b = resolve_model_id("meta-llama/Llama-3.2-3B")

    tokenizer = AutoTokenizer.from_pretrained(model_1b_id, **auth_1b)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    projector = FusedKVProjector(weights_path=weights_path, device=device)
    head_dim_1b = projector.head_dim_1b

    print(f"[Hybrid Eval] Loading Models...")
    model_1b = AutoModelForCausalLM.from_pretrained(model_1b_id, torch_dtype=torch.float16, **auth_1b).to(device).eval()
    model_3b = AutoModelForCausalLM.from_pretrained(model_3b_id, torch_dtype=torch.float16, **auth_3b).to(device).eval()

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()

    eval_prefix = (
        "Distributed Key-Value Store Architecture and Replication Consensus.\n"
        "In high-throughput distributed database engines, linearizable consistency is maintained using Multi-Raft state machine replication. "
        "Each shard leader sequences write requests into an append-only write-ahead log (WAL) on NVMe storage, broadcasting log entries to followers."
    )
    eval_target = (
        " Once a quorum of replicas acknowledge persistence, the entry is committed to the immutable memtable and flushed asynchronously into Log-Structured Merge (LSM) SSTables on disk."
    )

    prefix_ids = tokenizer(eval_prefix, return_tensors="pt")["input_ids"].to(device)
    target_ids = tokenizer(eval_target, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    full_ids = torch.cat([prefix_ids, target_ids], dim=-1)
    prefix_len = prefix_ids.shape[1]

    # Configurations: Cutoff layer index (0 to cutoff transplanted from 1B, remaining from 3B)
    cutoffs = [28, 24, 20, 0]
    frontier_results = []

    print("\n" + "=" * 85)
    print(f"{'Configuration':<24} | {'Transplanted':<14} | {'Perplexity (PPL)':<18} | {'Prefill Latency':<16}")
    print("-" * 85)

    for cutoff in cutoffs:
        sync()
        t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        t_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

        # Time prefill
        t0_time = time.perf_counter()
        with torch.no_grad():
            if cutoff == 0:
                # 100% Native 3B
                out_native = model_3b(full_ids)
                sync()
                t_prefill_ms = (time.perf_counter() - t0_time) * 1000.0
                logits = out_native.logits[:, prefix_len - 1 : -1, :]
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target_ids.view(-1)).item()
                ppl = math.exp(loss)
                label = "Native 3B (0/28)"
            else:
                # Transplanted prefix
                out_1b = model_1b(prefix_ids, use_cache=True)
                raw_keys = [getattr(l, "keys", getattr(l, "key_states", None)) for l in out_1b.past_key_values.layers]
                raw_vals = [getattr(l, "values", getattr(l, "value_states", None)) for l in out_1b.past_key_values.layers]
                unwound_keys = [inverse_rope(k, head_dim=head_dim_1b, base=500000.0) for k in raw_keys]

                # Full transplanted cache
                cache = projector.project_and_build_cache(unwound_keys, raw_vals, prefix_len)

                # If hybrid (cutoff < 28), replace top layers with native 3B states
                if cutoff < 28:
                    out_top = model_3b(prefix_ids, use_cache=True)
                    for j in range(cutoff, 28):
                        k_top = out_top.past_key_values.layers[j].keys
                        v_top = out_top.past_key_values.layers[j].values
                        cache.layers[j].keys = k_top
                        cache.layers[j].values = v_top

                sync()
                t_prefill_ms = (time.perf_counter() - t0_time) * 1000.0

                eval_input = torch.cat([prefix_ids[:, -1:], target_ids[:, :-1]], dim=-1)
                out_eval = model_3b(eval_input, past_key_values=cache, use_cache=True)
                logits = out_eval.logits
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target_ids.view(-1)).item()
                ppl = math.exp(loss)
                label = f"Transplant ({cutoff}/28)" if cutoff == 28 else f"Hybrid ({cutoff}/28)"

        entry = {
            "label": label,
            "cutoff_layers": cutoff,
            "transplanted_count": cutoff,
            "loss": loss,
            "perplexity": ppl,
            "prefill_ms": t_prefill_ms,
        }
        frontier_results.append(entry)
        print(f"{label:<24} | {cutoff:>2}/28 layers   | {ppl:>18.2f} | {t_prefill_ms:>13.2f} ms")

    print("=" * 85)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"hybrid_frontier": frontier_results}, f, indent=2)

    print(f"[Hybrid Eval] Frontier results saved to {output_path}")
    return frontier_results


if __name__ == "__main__":
    evaluate_hybrid_frontier()
