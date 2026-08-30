"""
Extract Dataset for Cross-Model KV Cache Transplant
Extracts paired KV caches from Llama-3.2-1B and Llama-3.2-3B over calibration prompts.
"""

import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
import gc
import json
import torch
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from rope_utils import inverse_rope, apply_rope


CALIBRATION_PROMPTS = [
    # Technical & Systems Programming
    "Explain the architecture of Apple Silicon Unified Memory Architecture (UMA) and why it provides high bandwidth for LLM KV cache operations.",
    "Write a high-performance concurrent queue in Rust using atomic operations, lock-free nodes, and memory fences.",
    "Describe the mathematical foundation of Rotary Position Embeddings (RoPE) and how relative positional encodings are represented in complex vector space.",
    "How does Grouped Query Attention (GQA) reduce memory bandwidth pressure during autoregressive decoding compared to Multi-Head Attention?",
    "Explain the internal implementation of PyTorch MPS backend and how Metal performance shaders dispatch GEMM kernels on Apple GPUs.",
    "Write an efficient C++ implementation of a B-tree indexing structure with concurrent read-write locks.",
    "Detail the difference between FlashAttention-2 and standard multi-head attention in terms of IO complexity and GPU SRAM tiling.",
    "Explain the Raft consensus algorithm, covering leader election, log replication, and safety invariants under network partitions.",
    "Write a Python script using asyncio and uvloop to handle 10,000 concurrent HTTP requests with rate limiting and retry backoff.",
    "Analyze the computational complexity of the Fast Fourier Transform (FFT) algorithm and its applications in signal processing.",

    # Reasoning, Logic & Mathematics
    "Prove that the square root of 2 is irrational using a proof by contradiction.",
    "Solve the following system of linear equations using Gaussian elimination step-by-step: 2x + 3y - z = 5, 4x + 4y - 3z = 3, -2x + 3y - z = 1.",
    "Explain the concept of Eigenvalues and Eigenvectors in linear transformations, with a geometric interpretation of principal component analysis.",
    "Derive the closed-form formula for Ridge Regression and explain how the L2 regularization parameter prevents ill-conditioned matrix inversion.",
    "Explain Bayes' Theorem with an intuitive medical diagnosis example, explaining prior probability, likelihood, and posterior distribution.",
    "What is the Halting Problem in theoretical computer science, and how did Alan Turing prove its undecidability using diagonalization?",
    "Describe the difference between P, NP, NP-Complete, and NP-Hard complexity classes with representative problem examples.",
    "Derive the backpropagation equations for a two-layer Multi-Layer Perceptron (MLP) with cross-entropy loss and softmax output.",
    "Explain the Central Limit Theorem and why the sum of independent, identically distributed random variables approaches a Gaussian distribution.",
    "What is dynamic programming? Illustrate the principle of optimality using the 0/1 Knapsack problem with pseudo-code.",

    # Science, Physics & Engineering
    "Describe the thermodynamic principles behind Carnot engines and explain why maximum efficiency is bounded by temperature differences.",
    "How does CRISPR-Cas9 genome editing locate target DNA sequences and introduce double-strand breaks for gene knockouts?",
    "Explain Quantum Entanglement and the Einstein-Podolsky-Rosen (EPR) paradox, along with Bell's Inequality tests.",
    "Detail the mechanisms of action of mRNA vaccines, covering lipid nanoparticle delivery, ribosome translation, and immune activation.",
    "Explain how semiconductor photolithography and Extreme Ultraviolet (EUV) light enable 3nm transistor fabrication.",
    "Describe the fluid dynamic Navier-Stokes equations and why numerical turbulence modeling (DNS, LES, RANS) is computationally difficult.",
    "How do transformer neural networks model long-range dependencies in protein folding predictions like AlphaFold?",
    "Explain the Doppler effect in electromagnetic waves and its application in astronomical redshift measurements of expanding universe.",
    "What are superconductor materials, how does the Meissner effect work, and what are the hurdles to room-temperature superconductivity?",
    "Describe the lifecycle of a massive star leading to core collapse, supernova nucleosynthesis, and black hole formation.",

    # Summaries & Essays
    "Summarize the historical evolution of modern operating systems from MULTICS and UNIX to modern monolithic and microkernel architectures.",
    "Write a comparative analysis of macroeconomic monetary policy vs fiscal policy during economic recessions and inflationary shocks.",
    "Summarize the key events and geopolitical consequences of the 1944 Bretton Woods conference on global financial institutions.",
    "Explain the history and philosophical evolution of cognitive science from behaviorism to symbolic AI and modern connectionism.",
    "Write an overview of international maritime law regarding the United Nations Convention on the Law of the Sea (UNCLOS) and exclusive economic zones.",
    "Analyze the structural shifts in global supply chains caused by semiconductor reshoring and just-in-case inventory management.",
    "Summarize the major findings and ethical considerations surrounding large language model alignment, RLHF, and constitutional AI.",
    "Explain the history of cryptography from the Caesar cipher and Enigma machine to RSA public-key encryption and lattice-based post-quantum cryptography.",
    "Write a comprehensive summary of how the human brain processes visual input from the retina through LGN to the primary visual cortex (V1-V4).",
    "Describe the rise and development of renewable energy microgrids, battery energy storage systems (BESS), and smart grid load balancing.",

    # Multi-Paragraph Complex Contexts
    "Design a scalable distributed key-value store like DynamoDB. Cover consistent hashing, virtual nodes, vector clocks for versioning, sloppy quorums, hinted handoff, and anti-entropy with Merkle trees. Provide architectural trade-offs.",
    "Provide a detailed technical breakdown of the Transformer architecture: Scaled Dot-Product Attention, Multi-Head Attention, Feed-Forward Networks, RMSNorm, SwiGLU activations, and Rotary Position Embeddings.",
    "Explain the complete compilation pipeline of an optimizing LLVM-based compiler: lexical analysis, AST construction, intermediate representation (IR), SSA form, optimization passes (DCE, LICM, GVN), and target machine code generation.",
    "Discuss the trade-offs of database isolation levels (Read Uncommitted, Read Committed, Repeatable Read, Serializable) and concurrency control anomalies (dirty reads, non-repeatable reads, phantom reads, write skew).",
    "Analyze the architectural differences between x86-64 out-of-order CISC microarchitectures and modern Apple Silicon ARM64 wide-decode cores.",
    "Explain how zero-knowledge proofs (ZK-SNARKs and ZK-STARKs) work mathematically using polynomial commitment schemes, arithmetic circuits, and QAPs.",
    "Describe the architecture of modern deep learning training clusters: NVLink, InfiniBand, RDMA, tensor parallelism, pipeline parallelism, and ZeRO stage 1/2/3.",
    "Explain the operation of Linux memory management: virtual memory, page tables, TLB shootdowns, buddy allocator, slab allocator, and swap mechanisms.",
    "Write an in-depth essay exploring how generative AI models affect software engineering practices, developer productivity, and automated testing paradigms.",
    "Detail the mechanics of speculative decoding in large language models: draft model verification, target model acceptance criteria, and latency scaling bounds."
]


def extract_kv_from_cache(past_key_values) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extracts (key, value) tensors from Hugging Face past_key_values
    supporting transformers 5.x DynamicLayer, 4.x DynamicCache, tuple of tuples, and Cache objects.
    """
    layers_kv = []
    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            k = getattr(layer, "keys", getattr(layer, "key_states", None))
            v = getattr(layer, "values", getattr(layer, "value_states", None))
            if k is not None and v is not None:
                layers_kv.append((k, v))
            elif isinstance(layer, (tuple, list)):
                layers_kv.append((layer[0], layer[1]))
    elif hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for i in range(len(past_key_values.key_cache)):
            k = past_key_values.key_cache[i]
            v = past_key_values.value_cache[i]
            layers_kv.append((k, v))
    elif isinstance(past_key_values, (tuple, list)):
        for layer_kv in past_key_values:
            layers_kv.append((layer_kv[0], layer_kv[1]))
    else:
        raise ValueError(f"Unsupported past_key_values type: {type(past_key_values)}")
    return layers_kv


def load_hf_token() -> Optional[str]:
    """Retrieves Hugging Face token from environment or default cache file."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        try:
            with open(token_path, "r") as f:
                return f.read().strip()
        except Exception:
            pass
    return None


def resolve_model_id(model_id: str, token: Optional[str] = None) -> str:
    """Attempts to load model config; if gated 403 occurs, falls back to ungated mirror."""
    from transformers import AutoConfig
    try:
        AutoConfig.from_pretrained(model_id, token=token)
        return model_id
    except Exception as e:
        fallback_map = {
            "meta-llama/Llama-3.2-1B": "unsloth/Llama-3.2-1B",
            "meta-llama/Llama-3.2-3B": "unsloth/Llama-3.2-3B",
            "meta-llama/Llama-3.2-1B-Instruct": "unsloth/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct": "unsloth/Llama-3.2-3B-Instruct",
        }
        if model_id in fallback_map:
            fallback = fallback_map[model_id]
            print(f"[Model Loader] Notice: {model_id} is gated ({e}). Automatically using ungated mirror: {fallback}")
            return fallback
        raise


def extract_and_save_dataset(
    model_1b_id: str = "meta-llama/Llama-3.2-1B",
    model_3b_id: str = "meta-llama/Llama-3.2-3B",
    output_path: str = "data/calibration_kv_pairs.pt",
    prompts: List[str] = CALIBRATION_PROMPTS,
    device_str: Optional[str] = None,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if device_str is None:
        device_str = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"[Extract] Using device: {device}")

    hf_token = load_hf_token()
    auth_kwargs = {"token": hf_token} if hf_token else {}

    model_1b_id = resolve_model_id(model_1b_id, hf_token)
    model_3b_id = resolve_model_id(model_3b_id, hf_token)

    print(f"[Extract] Loading Tokenizer from {model_1b_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_1b_id, **auth_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Extract KV caches from Llama-3.2-1B
    print(f"[Extract] Loading 1B Model: {model_1b_id} in float16 on {device}...")
    model_1b = AutoModelForCausalLM.from_pretrained(
        model_1b_id,
        torch_dtype=torch.float16,
        **auth_kwargs
    ).to(device)
    model_1b.eval()

    head_dim_1b = model_1b.config.hidden_size // model_1b.config.num_attention_heads
    num_layers_1b = model_1b.config.num_hidden_layers
    rope_theta_1b = getattr(model_1b.config, "rope_theta", 500000.0)
    print(f"[Extract] 1B Config: {num_layers_1b} layers, {model_1b.config.num_key_value_heads} KV heads, head_dim={head_dim_1b}, rope_theta={rope_theta_1b}")

    raw_1b_data = []
    print(f"[Extract] Running forward passes on 1B model over {len(prompts)} prompts...")
    for idx, prompt in enumerate(tqdm(prompts, desc="1B Forward Passes")):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model_1b(**inputs, use_cache=True)
            kv_layers = extract_kv_from_cache(outputs.past_key_values)
            
            # For each layer, unwind RoPE from Key
            clean_kv_per_layer = []
            for layer_idx, (k, v) in enumerate(kv_layers):
                # k shape: (1, num_kv_heads, seq_len, head_dim_1b)
                # v shape: (1, num_kv_heads, seq_len, head_dim_1b)
                k_clean = inverse_rope(k, head_dim=head_dim_1b, base=rope_theta_1b)
                clean_kv_per_layer.append({
                    "k_clean": k_clean.cpu(),
                    "v": v.cpu(),
                })
            raw_1b_data.append({
                "prompt_idx": idx,
                "seq_len": inputs["input_ids"].shape[1],
                "layers": clean_kv_per_layer
            })

    del model_1b
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # 2. Extract KV caches from Llama-3.2-3B
    print(f"[Extract] Loading 3B Model: {model_3b_id} in float16 on {device}...")
    model_3b = AutoModelForCausalLM.from_pretrained(
        model_3b_id,
        torch_dtype=torch.float16,
        **auth_kwargs
    ).to(device)
    model_3b.eval()

    head_dim_3b = model_3b.config.hidden_size // model_3b.config.num_attention_heads
    num_layers_3b = model_3b.config.num_hidden_layers
    rope_theta_3b = getattr(model_3b.config, "rope_theta", 500000.0)
    print(f"[Extract] 3B Config: {num_layers_3b} layers, {model_3b.config.num_key_value_heads} KV heads, head_dim={head_dim_3b}, rope_theta={rope_theta_3b}")

    raw_3b_data = []
    print(f"[Extract] Running forward passes on 3B model over {len(prompts)} prompts...")
    for idx, prompt in enumerate(tqdm(prompts, desc="3B Forward Passes")):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model_3b(**inputs, use_cache=True)
            kv_layers = extract_kv_from_cache(outputs.past_key_values)
            
            clean_kv_per_layer = []
            for layer_idx, (k, v) in enumerate(kv_layers):
                # k shape: (1, num_kv_heads, seq_len, head_dim_3b)
                # v shape: (1, num_kv_heads, seq_len, head_dim_3b)
                k_clean = inverse_rope(k, head_dim=head_dim_3b, base=rope_theta_3b)
                clean_kv_per_layer.append({
                    "k_clean": k_clean.cpu(),
                    "v": v.cpu(),
                })
            raw_3b_data.append({
                "prompt_idx": idx,
                "seq_len": inputs["input_ids"].shape[1],
                "layers": clean_kv_per_layer
            })

    del model_3b
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Save paired dataset
    dataset = {
        "model_1b_id": model_1b_id,
        "model_3b_id": model_3b_id,
        "num_layers_1b": num_layers_1b,
        "num_layers_3b": num_layers_3b,
        "head_dim_1b": head_dim_1b,
        "head_dim_3b": head_dim_3b,
        "num_kv_heads": 8,
        "data_1b": raw_1b_data,
        "data_3b": raw_3b_data,
        "prompts": prompts,
    }

    print(f"[Extract] Saving extracted KV dataset to {output_path}...")
    torch.save(dataset, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Extract] Dataset successfully saved! Size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    extract_and_save_dataset()
