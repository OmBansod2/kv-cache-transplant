# Cross-Model KV Cache Transplant

**Accelerating LLM prefill by 2.29× without fine-tuning.**

A research implementation of **Cross-Model Key-Value (KV) Cache Transplantation** between [Llama-3.2-1B](https://huggingface.co/unsloth/Llama-3.2-1B) and [Llama-3.2-3B](https://huggingface.co/unsloth/Llama-3.2-3B) on Apple Silicon MPS (M4 Pro).

The core idea: let the small fast model (1B) process the prompt, then project its attention KV state into the larger model's (3B) memory space via a single 31 ms fused GPU matrix multiplication — so the 3B model starts generating tokens without re-reading the prompt.

---

## Key Results

| Metric | Result |
|:---|:---:|
| **Prefill Speedup @ 4,096 tokens** | **2.29×** (5.60 s → 2.56 s) |
| **Key Cache Variance Recovery (R²)** | **99.43%** across all 28 layers |
| **Value Cache Variance Recovery (R²)** | **98.53%** across all 28 layers |
| **Fused 28-Layer GPU Projection** | **31.2 ms** (single 3D batched GEMM) |
| **Scientific Domain PPL Parity** | **77.8%** of native 3B quality |
| **Fine-tuning Required** | ❌ None |
| **Quantization Required** | ❌ None |

---

## How It Works

```
Standard Pipeline:
  Prompt → 3B Prefill (28 layers, 128-dim heads) → 5.6 s TTFT

Transplant Pipeline:
  Prompt → 1B Prefill (16 layers, 64-dim heads) → 2.0 s
                ↓
          RoPE Unwind (analytical)
                ↓
          Fused 3D Batched GEMM: (28, seq_len, 1536) @ (28, 1536, 1024) → 31 ms
                ↓
          3B KV Cache Populated → 0.5 s first token
  Total TTFT: 2.56 s  →  2.29× speedup at 4,096 tokens
```

### Architecture Components

1. **RoPE Inversion & Offset Preservation** (`rope_utils.py`): Analytically unwinds 1B's 64-dim rotary embeddings with exact token position offsets before projection, then re-applies 3B's 128-dim frequencies.
2. **Fused Batched GPU Projector** (`fused_projector.py`): Real-time inference kernel vectorizing all 28 layer projections into a single 31.2 ms 3D batched GEMM (`torch.bmm`) on Apple Silicon MPS.
3. **Neural Residual MLP Adapters** (`kv_adapter.py`, `train_mlp_mapper.py`): Closed-form Ridge linear base + non-linear GELU delta MLP per layer used for high-fidelity offline variance recovery analysis ($R^2 > 98.5\%$).
4. **Hybrid Selective Transplant** (`hybrid_transplant.py`): Transplant only early/mid layers; let 3B compute its top semantic layers natively for a quality-latency Pareto curve.

---

## Speedup vs Context Length

| Prompt Length | Native 3B TTFT | Transplant TTFT | Speedup | Time Saved |
|:---:|:---:|:---:|:---:|:---:|
| 128 tokens | 165 ms | 128 ms | 1.29× | 37 ms |
| 512 tokens | 682 ms | 355 ms | 1.92× | 327 ms |
| 1,024 tokens | 1,354 ms | 634 ms | 2.13× | 720 ms |
| 2,048 tokens | 2,710 ms | 1,215 ms | 2.23× | 1.5 s |
| **4,096 tokens** | **5,602 ms** | **2,564 ms** | **2.29×** | **3.0 s** |

---

## Repository Structure

```
kv-cache-transplant/
├── rope_utils.py              # Analytical RoPE inversion and re-application
├── kv_adapter.py              # Neural Residual MLP adapter definition
├── extract_dataset.py         # Extract paired KV caches from 1B and 3B over calibration prompts
├── train_mlp_mapper.py        # Train 56 Residual MLP adapters (28 key + 28 value)
├── fused_projector.py         # Fused batched GPU projector (single 3D GEMM)
├── hybrid_transplant.py       # Hybrid layer-selective transplant evaluator
├── evaluate_perplexity.py     # Multi-domain perplexity (PPL) evaluation suite
├── benchmark_scaling.py       # TTFT / prefill speedup scaling benchmark
├── quality_gate.py            # Runtime PPL quality probe with auto-fallback
├── incremental_transplant.py  # Streaming multi-turn incremental cache append
├── position_aware_adapter.py  # Position-encoding-augmented RoPE correction adapter
├── model_offloader.py         # GPU/CPU model swapping for memory-constrained devices
├── generate_html_report.py    # Generates interactive HTML research dashboard
├── data/                      # Extracted KV cache datasets and benchmark JSON results
│   └── calibration_kv_pairs.pt
└── weights/                   # Trained adapter weights
    └── mapper.pt
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- PyTorch 2.1+ (with MPS on Apple Silicon, or CUDA on GPU)
- 16 GB RAM minimum (24 GB recommended for both models loaded simultaneously)

### Installation

```bash
git clone https://github.com/yourname/kv-cache-transplant.git
cd kv-cache-transplant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Quick Start

```bash
# Step 1: Extract paired KV cache dataset (calibration prompts → ~200 MB)
python extract_dataset.py

# Step 2: Train the 56 Neural Residual MLP adapters (~38 seconds on Apple MPS)
python train_mlp_mapper.py

# Step 3: Run TTFT scaling benchmark (128 → 4096 tokens)
python benchmark_scaling.py

# Step 4: Multi-domain perplexity evaluation
python evaluate_perplexity.py

# Step 5: Generate interactive HTML report
python generate_html_report.py
# Open report.html in your browser
```

---

## Module Reference

### `rope_utils.py`
Implements analytical RoPE inverse — unwinding the position-dependent rotation matrices from 1B's key tensors before cross-model projection. Re-applies 3B's RoPE frequencies after projection.

### `kv_adapter.py`
Defines `ResidualMLPAdapter`: a Ridge-initialized linear base path combined with a 2-layer GELU MLP residual delta used for offline high-fidelity variance recovery analysis.

### `extract_dataset.py`
Runs forward passes on both 1B and 3B over a configurable set of calibration prompts, extracts `(key, value)` pairs per layer, and saves a paired dataset for adapter training.

### `train_mlp_mapper.py`
Trains 28 key adapters + 28 value adapters using the paired KV dataset. Computes R² and cosine similarity metrics per layer. Saves adapter weights to `weights/mapper.pt`.

### `fused_projector.py`
Loads Ridge base weights and stacks them into 3D parameter tensors `(28, 1536, 1024)`. Executes all 28 layer projections concurrently in a single 31.2 ms `torch.bmm` call with accurate position offsets, building a 3B `DynamicCache`.

### `benchmark_scaling.py`
Measures TTFT across prompt lengths from 128 to 4,096 tokens. Compares native 3B prefill vs transplant pipeline. Saves results to `data/scaling_benchmark_results.json`.

### `quality_gate.py`
Runs a lightweight 10-token PPL probe on the 3B model after transplant. Blocks out-of-distribution prompts (PPL > threshold) and falls back to native 3B prefill automatically.

### `hybrid_transplant.py`
Evaluates the Pareto frontier by varying the layer cutoff: transplant layers 0–K from 1B and compute layers K+1–27 natively on 3B. Traces quality-vs-latency trade-offs.

### `evaluate_perplexity.py`
Evaluates cross-entropy loss and perplexity across domains (Scientific, Code, Systems, Reasoning). Reports native 3B PPL vs transplanted 3B PPL and quality parity percentage.

### `incremental_transplant.py`
Multi-turn streaming cache: appends only new tokens per conversation turn to the persistent 3B KV cache, avoiding redundant re-processing of history.

### `position_aware_adapter.py`
Augments the adapter input with a sinusoidal position encoding basis to correct 64-dim vs 128-dim RoPE frequency mismatches in the projection layer.

### `model_offloader.py`
Swaps models between GPU and CPU on Apple Silicon Unified Memory, enabling single-model active GPU mode and saving ~2.3 GB VRAM.

---

## Citation

If you find this useful for your research, please cite:

```bibtex
@misc{kv_cache_transplant_2026,
  title  = {Cross-Model KV Cache Transplantation: Accelerating LLM Prefill Without Fine-Tuning},
  author = {Om Bansod},
  year   = {2026},
  url    = {https://github.com/OmBansod2/kv-cache-transplant}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
