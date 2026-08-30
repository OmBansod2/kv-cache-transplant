"""
Multi-Context Length Scaling Benchmark for Cross-Model KV Cache Transplant
Measures Prefill Latency and TTFT Speedup across sequence lengths: 128, 512, 1024, 2048, and 4096 tokens.
"""

import os
import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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


def run_scaling_benchmark(
    model_1b_name: str = "meta-llama/Llama-3.2-1B",
    model_3b_name: str = "meta-llama/Llama-3.2-3B",
    weights_path: str = "weights/mapper.pt",
    output_path: str = "data/scaling_benchmark_results.json",
    context_lengths: list[int] = [128, 512, 1024, 2048, 4096],
):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Scaling Benchmark] Target Device: {device}")
    print(f"[Scaling Benchmark] Context Lengths: {context_lengths} tokens")

    model_1b_id, auth_1b = resolve_model_id(model_1b_name)
    model_3b_id, auth_3b = resolve_model_id(model_3b_name)

    tokenizer = AutoTokenizer.from_pretrained(model_1b_id, **auth_1b)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Fused Projector
    print(f"[Scaling Benchmark] Loading Fused GPU Projector from {weights_path}...")
    projector = FusedKVProjector(weights_path=weights_path, device=device)
    head_dim_1b = projector.head_dim_1b

    # Load Models in float16
    print(f"[Scaling Benchmark] Loading 1B Model ({model_1b_id})...")
    model_1b = AutoModelForCausalLM.from_pretrained(model_1b_id, torch_dtype=torch.float16, **auth_1b).to(device).eval()

    print(f"[Scaling Benchmark] Loading 3B Model ({model_3b_id})...")
    model_3b = AutoModelForCausalLM.from_pretrained(model_3b_id, torch_dtype=torch.float16, **auth_3b).to(device).eval()

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()

    # Warmup
    print("[Scaling Benchmark] Warming up GPU kernels...")
    dummy = torch.randint(100, 5000, (1, 64), device=device)
    with torch.no_grad():
        _ = model_1b(dummy, use_cache=True)
        _ = model_3b(dummy, use_cache=True)
        sync()

    results = []

    print("\n" + "=" * 85)
    print(f"{'Length (Tokens)':<15} | {'Native 3B TTFT':<16} | {'Transplant TTFT':<16} | {'TTFT Speedup':<14} | {'Prefill Speedup':<15}")
    print("-" * 85)

    base_vocab_sample = torch.randint(100, 8000, (1, max(context_lengths)), device=device)

    for seq_len in context_lengths:
        input_ids = base_vocab_sample[:, :seq_len]

        # -------------------------------------------------------------
        # 1. Native 3B Benchmark
        # -------------------------------------------------------------
        sync()
        t0_native = time.perf_counter()
        with torch.no_grad():
            out_3b = model_3b(input_ids, use_cache=True)
            sync()
            t_prefill_3b = time.perf_counter()
            _ = torch.argmax(out_3b.logits[:, -1, :], dim=-1)
            sync()
            t_ttft_3b = time.perf_counter()

        native_prefill_ms = (t_prefill_3b - t0_native) * 1000.0
        native_ttft_ms = (t_ttft_3b - t0_native) * 1000.0

        # Clean memory
        del out_3b
        sync()

        # -------------------------------------------------------------
        # 2. Transplant Pipeline Benchmark (1B Prefill + Fused Proj + 3B TTFT)
        # -------------------------------------------------------------
        sync()
        t0_trans = time.perf_counter()
        with torch.no_grad():
            # Step 1: 1B Prefill
            out_1b = model_1b(input_ids, use_cache=True)
            sync()
            t_1b_done = time.perf_counter()
            t_1b_prefill_ms = (t_1b_done - t0_trans) * 1000.0

            # Step 2: Unwind RoPE
            past_kv_1b = out_1b.past_key_values
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
            sync()
            t_unwind_done = time.perf_counter()
            t_unwind_ms = (t_unwind_done - t_1b_done) * 1000.0

            # Step 3: Fused Batched GPU Projection & RoPE re-application
            transplanted_cache = projector.project_and_build_cache(unwound_keys, raw_vals, seq_len)
            sync()
            t_proj_done = time.perf_counter()
            t_proj_ms = (t_proj_done - t_unwind_done) * 1000.0
            t_trans_prefill_total_ms = (t_proj_done - t0_trans) * 1000.0

            # Step 4: 3B First Token
            last_token = input_ids[:, -1:]
            out_first = model_3b(input_ids=last_token, past_key_values=transplanted_cache, use_cache=True)
            _ = torch.argmax(out_first.logits[:, -1, :], dim=-1)
            sync()
            t_ttft_trans = time.perf_counter()

        trans_ttft_ms = (t_ttft_trans - t0_trans) * 1000.0

        del out_1b, out_first, transplanted_cache
        sync()

        ttft_speedup = native_ttft_ms / max(trans_ttft_ms, 1e-5)
        prefill_speedup = native_prefill_ms / max(t_trans_prefill_total_ms, 1e-5)

        entry = {
            "seq_len": seq_len,
            "native_prefill_ms": native_prefill_ms,
            "native_ttft_ms": native_ttft_ms,
            "transplant_1b_prefill_ms": t_1b_prefill_ms,
            "transplant_unwind_ms": t_unwind_ms,
            "transplant_fused_proj_ms": t_proj_ms,
            "transplant_prefill_total_ms": t_trans_prefill_total_ms,
            "transplant_ttft_ms": trans_ttft_ms,
            "ttft_speedup": ttft_speedup,
            "prefill_speedup": prefill_speedup,
        }
        results.append(entry)

        print(f"{seq_len:<15} | {native_ttft_ms:>13.2f} ms | {trans_ttft_ms:>13.2f} ms | {ttft_speedup:>12.2f}x | {prefill_speedup:>13.2f}x")

    print("=" * 85)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"scaling_results": results}, f, indent=2)

    print(f"[Scaling Benchmark] Complete results saved to {output_path}")
    return results


if __name__ == "__main__":
    run_scaling_benchmark()
