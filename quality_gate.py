"""
Quality Gate Module for Cross-Model KV Cache Transplant (Fix 2)
Runs a quick perplexity probe after transplanting to detect out-of-distribution failure.
If PPL exceeds threshold, automatically falls back to native 3B prefill.
"""

import time
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


class QualityGate:
    """
    After transplanting KV cache from 1B to 3B, probes the quality of the transplanted
    memory by computing cross-entropy loss over a small window of the prompt's own tokens.
    If perplexity exceeds a threshold, falls back to native 3B prefill.
    """
    def __init__(
        self,
        model_3b: AutoModelForCausalLM,
        ppl_threshold: float = 30.0,
        probe_tokens: int = 10,
    ):
        self.model_3b = model_3b
        self.ppl_threshold = ppl_threshold
        self.probe_tokens = probe_tokens

    @torch.no_grad()
    def check_transplant_quality(
        self,
        input_ids: torch.Tensor,
        transplanted_cache: DynamicCache,
    ) -> tuple[bool, float, float]:
        """
        Runs a quick perplexity probe on the last `probe_tokens` of the prompt.
        Returns: (passed: bool, measured_ppl: float, gate_latency_ms: float)
        """
        seq_len = input_ids.shape[1]
        n_probe = min(self.probe_tokens, seq_len - 1)
        if n_probe < 2:
            return True, 0.0, 0.0

        # We check: given the transplanted cache for tokens [0..seq_len-n_probe-1],
        # can the 3B model correctly predict tokens [seq_len-n_probe..seq_len-1]?
        probe_start = seq_len - n_probe - 1
        probe_input = input_ids[:, probe_start:-1]  # (1, n_probe)
        probe_target = input_ids[:, probe_start + 1:]  # (1, n_probe)

        # Crop cache to before probe window
        probe_cache = self._crop_cache_copy(transplanted_cache, probe_start)

        t0 = time.perf_counter()
        out = self.model_3b(input_ids=probe_input, past_key_values=probe_cache, use_cache=False)
        logits = out.logits  # (1, n_probe, vocab_size)

        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            probe_target.view(-1),
        ).item()
        ppl = math.exp(min(loss, 20.0))  # Cap to avoid overflow

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
        gate_ms = (time.perf_counter() - t0) * 1000.0

        passed = ppl <= self.ppl_threshold
        return passed, ppl, gate_ms

    def _crop_cache_copy(self, cache: DynamicCache, target_len: int) -> DynamicCache:
        """Creates a shallow copy of cache cropped to target_len."""
        new_cache = DynamicCache()
        if hasattr(cache, "layers"):
            for i, layer in enumerate(cache.layers):
                k = getattr(layer, "keys", getattr(layer, "key_states", None))
                v = getattr(layer, "values", getattr(layer, "value_states", None))
                if k is not None and v is not None:
                    new_cache.update(
                        k[:, :, :target_len, :],
                        v[:, :, :target_len, :],
                        layer_idx=i,
                    )
        elif hasattr(cache, "key_cache"):
            for i in range(len(cache.key_cache)):
                new_cache.update(
                    cache.key_cache[i][:, :, :target_len, :],
                    cache.value_cache[i][:, :, :target_len, :],
                    layer_idx=i,
                )
        return new_cache
