# Cross-Model KV Cache Transplant

[![Interactive Report](https://img.shields.io/badge/Interactive_Research_Report-Live_on_GitHub_Pages-0284c7?style=for-the-badge&logo=github)](https://ombansod2.github.io/kv-cache-transplant/)

**A linear map recovers ~70% of attention-key variance across models in the same family — and still cannot beat simply running the smaller model.**

**Author:** [Om Bansod](https://github.com/OmBansod2)

A research implementation of **Cross-Model Key-Value (KV) Cache Transplantation** between [Llama-3.2-1B](https://huggingface.co/unsloth/Llama-3.2-1B) and [Llama-3.2-3B](https://huggingface.co/unsloth/Llama-3.2-3B) on Apple Silicon MPS (M4 Pro).

> 📊 **Read the Full Interactive Research Report:** [https://ombansod2.github.io/kv-cache-transplant/](https://ombansod2.github.io/kv-cache-transplant/)

The core idea: let the small fast model (1B) process the prompt, then project its attention KV state into the larger model's (3B) memory space via a single 31 ms fused GPU matrix multiplication — so the 3B model starts generating tokens without re-reading the prompt.

---

## Key Results

### Cache reconstruction

The projection is fitted on calibration prompts and scored on held-out ones. Both
columns are reported, because the in-sample figure is the one a naive setup would
produce and the gap between them is the point: at `top_k=3` the regression fits
1.57M parameters per layer, so a small calibration set interpolates.

| calibration | KEYS in-sample | KEYS **held out** | VALUES in-sample | VALUES **held out** |
|:---|---:|---:|---:|---:|
| 1,406 tokens | 99.78% | **69.3%** | 98.53% | **47.4%** |
| 199,936 tokens | — | **73.0%** | — | **53.5%** |

Splits are **by prompt**, never by token: tokens inside one prompt share a context
and their KV states are strongly correlated, so a token-level shuffle reports an
in-sample number as held-out.

### Downstream quality — WikiText-103 test, n = 200 windows

Paired bootstrap CIs on per-window loss. Prefix 128 tokens, target 128 tokens.

| condition | PPL | vs native 3B | 95% CI |
|:---|---:|---:|:---|
| native 3B (ceiling) | **9.87** | — | — |
| **native 1B (floor)** | **12.68** | +0.2507 | [+0.2379, +0.2634] |
| transplant, 1.4k calibration, α=1 | 253.94 | +3.2475 | [+3.1667, +3.3274] |
| transplant, 1.4k calibration, α=100 | 39.27 | +1.3808 | [+1.3292, +1.4347] |
| transplant, 200k calibration, α tuned | **14.87** | +0.4100 | [+0.3780, +0.4433] |

**The transplant does not beat the native 1B model.** Calibration size and the
ridge penalty dominate everything else — moving from 1,406 tokens at α=1 to 199,936
tokens with a tuned penalty improves perplexity **17×** (253.94 → 14.87). The gap
to the 1B floor survives that.

### Source-layer count (`top_k`)

Ridge penalty tuned on a separate validation split (30 fit / 10 validation /
10 test by prompt, 3 seeds). Concatenating several 1B layers per 3B layer helps:

| top_k | in_dim | KEYS test | VALUES test |
|---:|---:|---:|---:|
| 1 | 512 | 64.5% | 39.7% |
| **3** (shipped) | 1536 | **69.3%** | **47.4%** |
| 5 | 2560 | 71.1% | 49.6% |

### Engineering

| Metric | Result |
|:---|:---:|
| **Fused 28-Layer GPU Projection** | **31.2 ms** (single 3D batched GEMM) |
| **Prefill Speedup @ 4,096 tokens** | 2.22× — full 28/28 transplant; see [Methodology notes](#methodology-notes) |
| **Fine-tuning Required** | ❌ None |
| **Quantization Required** | ❌ None |

---

## Methodology notes

Four things that materially change the numbers, recorded because they are easy to
get wrong.

**Held-out splits must be by prompt, not by token.** Tokens inside one prompt share
a context and their KV states are strongly correlated, so a token-level shuffle
places near-duplicates on both sides and reports an in-sample score as held-out.

**Calibration size dominates.** At `top_k=3` the projection fits 1.57M parameters
per layer per k/v. A 1,406-token calibration set (50 prompts, mean 28 tokens) has
fewer rows than parameters, so in-sample R² approaches 100% by interpolation.
`calibrate_large.py` accumulates X'X and X'Y in a streaming pass, so calibration is
unbounded in tokens at fixed memory.

**The ridge penalty has to be tuned, and tuned per `top_k`.** `in_dim` ranges
512→2560 across the sweep, so a fixed α is a different amount of regularisation at
each k and confounds the comparison. α=100 rather than α=1 is worth a 6.5×
perplexity improvement on its own.

**Prefill savings depend on configuration and context length.** The 2.22× figure is
the full 28/28 transplant at 4,096 tokens, where 3B prefill dominates. A hybrid
configuration that computes upper layers natively cannot save prefill at all:
obtaining layer 21's KV requires a full forward pass over the prefix, since there
is no way to reach the top of a transformer without computing the bottom
(`hybrid_transplant.py:104-111`). At shorter context the full transplant is slower
than native 3B — `data/hybrid_benchmark_results.json` records 331.3 ms against
187.6 ms. Always quote the configuration and the context length together.

**On the MLP delta:** `FusedKVProjector` reads only `base_linear.weight`, and
`train_mlp_mapper` excludes `base_linear` from the optimiser, so it remains the
closed-form ridge solution. The deployed projection is pure ridge; the delta
network does not reach it.

### Analysis scripts

```
ablate_topk.py           in-sample vs held-out, single split
ablate_topk_seeds.py     5 prompt-level splits at fixed alpha
ablate_topk_tuned.py     3-way split, alpha tuned on validation — the result to trust
calibrate_large.py       streaming Gram-matrix calibration, unbounded tokens, fixed memory
build_mappers.py         ridge mappers at a chosen alpha and top_k
evaluate_ppl_proper.py   WikiText-103 + a technical-prose set, paired bootstrap CIs
FINDINGS.md              full write-up
```

### What the result actually is

The projection's only input is the 1B's cache. The 3B's advantage over the 1B is
exactly the information the 1B does not have, so no map — linear, MLP or otherwise
— can recover it. A perfect projection yields 1B-level information in 3B format,
which places the ceiling below the point of usefulness for quality-matched
acceleration. Recovering ~70% of key variance across independently-sized models in
one family is a real and somewhat surprising amount of shared structure; it is not
enough to make the transplant worth using over the smaller model.

---

## How It Works

```
Standard Pipeline:
  Prompt → 3B Prefill (28 layers, 128-dim heads) → 5.5 s TTFT

Transplant Pipeline:
  Prompt → 1B Prefill (16 layers, 64-dim heads) → 2.0 s
                ↓
          RoPE Unwind (analytical)
                ↓
          Fused 3D Batched GEMM: (28, seq_len, 1536) @ (28, 1536, 1024) → 31 ms
                ↓
          3B KV Cache Populated → 0.5 s first token
  Total TTFT: 2.50 s  →  2.22× speedup at 4,096 tokens
```

## Mathematical Formulation

### 1. Analytical RoPE Inversion
Llama-3.2 applies Rotary Position Embedding (RoPE) to key and query states. Given a key vector $x \in \mathbb{R}^d$, the rotated key $x_{rot}$ is computed as:

$$ x_{rot} = x \odot \cos(\theta) + \text{rotate\_half}(x) \odot \sin(\theta) $$

To map 1B's keys to 3B's memory space, we must first computationally unwind the 1B RoPE ($d=64$, $\text{base}=500000$) to recover the position-independent semantic key state:

$$ x = x_{rot} \odot \cos(\theta) - \text{rotate\_half}(x_{rot}) \odot \sin(\theta) $$

This inverse is algebraically exact and preserves tensor variance with $< 10^{-5}$ numerical error prior to cross-model projection.

### 2. Ridge Regression Mapping
The core projection maps the concatenated top-$k$ source layers from 1B ($X \in \mathbb{R}^{N \times 1536}$) to the target layer in 3B ($Y \in \mathbb{R}^{N \times 1024}$). The closed-form analytical Ridge projection weight matrix $W$ is computed as:

$$ W = (X_c^T X_c + \alpha I)^{-1} X_c^T Y_c $$

where $X_c$ and $Y_c$ are the mean-centered activation matrices, and $\alpha = 1.0$ is the $L2$ regularization term to ensure stable rank conditioning on Apple Silicon MPS.

### 3. Fused 3D Batched GEMM Projector
Instead of iterating sequentially across Transformer layers, we stack the trained linear base weights into a 3D tensor $\mathbf{W} \in \mathbb{R}^{28 \times 1536 \times 1024}$. During real-time prefill inference, all 28 target layers are projected concurrently via a single Batched General Matrix Multiply (bmm):

$$ \mathbf{Y}_{3B} = \mathbf{X}_{1B} \circledast \mathbf{W} + \mathbf{b} $$

This vectorization bypasses PyTorch's MPS dispatch overhead, achieving an ultra-low latency of $31.2\text{ ms}$ for the entire cache translation matrix.

### Architecture Components

1. **RoPE Inversion & Offset Preservation** (`rope_utils.py`): Analytically unwinds 1B's 64-dim rotary embeddings with exact token position offsets before projection, then re-applies 3B's 128-dim frequencies.
2. **Fused Batched GPU Projector** (`fused_projector.py`): Real-time inference kernel vectorizing all 28 layer projections into a single 31.2 ms 3D batched GEMM (`torch.bmm`) on Apple Silicon MPS.
3. **Neural Residual MLP Adapters** (`kv_adapter.py`, `train_mlp_mapper.py`): Closed-form Ridge linear base + non-linear GELU delta MLP per layer used for high-fidelity offline variance recovery analysis ($R^2 > 98.5\%$).
4. **Hybrid Selective Transplant** (`hybrid_transplant.py`): Transplant only early/mid layers; let 3B compute its top semantic layers natively for a quality-latency Pareto curve.

---

## Speedup vs Context Length

| Prompt Length | Native 3B TTFT | Transplant TTFT | Speedup | Time Saved |
|:---:|:---:|:---:|:---:|:---:|
| 128 tokens | 196 ms | 147 ms | 1.33× | 49 ms |
| 512 tokens | 663 ms | 337 ms | 1.97× | 326 ms |
| 1,024 tokens | 1,310 ms | 641 ms | 2.04× | 669 ms |
| 2,048 tokens | 2,680 ms | 1,424 ms | 1.88× | 1.2 s |
| **4,096 tokens** | **5,551 ms** | **2,497 ms** | **2.22×** | **3.0 s** |

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
git clone https://github.com/OmBansod2/kv-cache-transplant.git
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
