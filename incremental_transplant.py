"""
Incremental Streaming KV Cache Transplant (Fix 3)
Maintains persistent 1B and 3B KV caches across conversation turns.
On each new message, only processes new tokens through 1B, projects them, and appends
to the existing transplanted 3B cache with correct RoPE position offsets.
"""

import torch
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from rope_utils import inverse_rope, apply_rope
from fused_projector import FusedKVProjector


class IncrementalTransplantCache:
    """
    Maintains persistent cross-model KV caches for multi-turn conversations.
    Only processes NEW tokens incrementally on each turn.
    """
    def __init__(
        self,
        model_1b: AutoModelForCausalLM,
        model_3b: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        projector: FusedKVProjector,
        device: torch.device,
    ):
        self.model_1b = model_1b
        self.model_3b = model_3b
        self.tokenizer = tokenizer
        self.projector = projector
        self.device = device
        self.head_dim_1b = projector.head_dim_1b

        # Persistent state
        self.past_kv_1b: Optional[DynamicCache] = None
        self.past_kv_3b: Optional[DynamicCache] = None
        self.total_tokens: int = 0
        self.turn_count: int = 0

    def reset(self):
        """Resets all cached state for a new conversation."""
        self.past_kv_1b = None
        self.past_kv_3b = None
        self.total_tokens = 0
        self.turn_count = 0

    @torch.no_grad()
    def append_turn(self, new_text: str) -> DynamicCache:
        """
        Processes a new conversation turn incrementally:
        1. Tokenize only the new text
        2. Run 1B forward on new tokens with existing 1B cache
        3. Extract and project only the NEW KV entries
        4. Append projected entries to existing 3B cache
        """
        new_ids = self.tokenizer(new_text, return_tensors="pt", add_special_tokens=(self.turn_count == 0))["input_ids"].to(self.device)
        new_len = new_ids.shape[1]
        self.turn_count += 1

        # Step 1: Run 1B forward on new tokens (with existing cache if available)
        out_1b = self.model_1b(input_ids=new_ids, past_key_values=self.past_kv_1b, use_cache=True)
        self.past_kv_1b = out_1b.past_key_values

        # Step 2: Extract ONLY the new KV entries (last new_len positions)
        if hasattr(self.past_kv_1b, "layers"):
            new_keys = [getattr(l, "keys", getattr(l, "key_states", None))[:, :, -new_len:, :] for l in self.past_kv_1b.layers]
            new_vals = [getattr(l, "values", getattr(l, "value_states", None))[:, :, -new_len:, :] for l in self.past_kv_1b.layers]
        elif hasattr(self.past_kv_1b, "key_cache"):
            new_keys = [self.past_kv_1b.key_cache[i][:, :, -new_len:, :] for i in range(len(self.past_kv_1b.key_cache))]
            new_vals = [self.past_kv_1b.value_cache[i][:, :, -new_len:, :] for i in range(len(self.past_kv_1b.value_cache))]
        else:
            new_keys = [l[0][:, :, -new_len:, :] for l in self.past_kv_1b]
            new_vals = [l[1][:, :, -new_len:, :] for l in self.past_kv_1b]

        # Step 3: Unwind RoPE from new keys using true token positions
        pos_ids = torch.arange(self.total_tokens, self.total_tokens + new_len, device=self.device)
        unwound_new_keys = [
            inverse_rope(k, position_ids=pos_ids, head_dim=self.head_dim_1b, base=500000.0)
            for k in new_keys
        ]

        # Step 4: Project new KV entries through fused projector with position offset
        new_total = self.total_tokens + new_len
        projected_new = self.projector.project_and_build_cache(
            unwound_new_keys, new_vals, new_len, position_offset=self.total_tokens
        )

        # Step 5: Append to existing 3B cache
        if self.past_kv_3b is None:
            self.past_kv_3b = projected_new
        else:
            # Concatenate new projected entries to existing cache
            if hasattr(self.past_kv_3b, "layers"):
                for i, layer in enumerate(self.past_kv_3b.layers):
                    proj_layer = projected_new.layers[i]
                    existing_k = getattr(layer, "keys", getattr(layer, "key_states", None))
                    existing_v = getattr(layer, "values", getattr(layer, "value_states", None))
                    new_k = getattr(proj_layer, "keys", getattr(proj_layer, "key_states", None))
                    new_v = getattr(proj_layer, "values", getattr(proj_layer, "value_states", None))
                    combined_k = torch.cat([existing_k, new_k], dim=2)
                    combined_v = torch.cat([existing_v, new_v], dim=2)
                    if hasattr(layer, "keys"):
                        layer.keys = combined_k
                        layer.values = combined_v
                    else:
                        layer.key_states = combined_k
                        layer.value_states = combined_v
            elif hasattr(self.past_kv_3b, "key_cache"):
                for i in range(len(self.past_kv_3b.key_cache)):
                    self.past_kv_3b.key_cache[i] = torch.cat([self.past_kv_3b.key_cache[i], projected_new.key_cache[i]], dim=2)
                    self.past_kv_3b.value_cache[i] = torch.cat([self.past_kv_3b.value_cache[i], projected_new.value_cache[i]], dim=2)

        self.total_tokens = new_total
        return self.past_kv_3b

    def get_cache_info(self) -> dict:
        """Returns current cache state info."""
        return {
            "total_tokens_cached": self.total_tokens,
            "turns_processed": self.turn_count,
            "3b_cache_layers": len(self.past_kv_3b.layers) if self.past_kv_3b and hasattr(self.past_kv_3b, "layers") else 0,
        }
