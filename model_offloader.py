"""
Model Offloader for Low-Memory Devices (Fix 7)
Swaps models between GPU and CPU to halve peak GPU memory usage.
On Apple Silicon UMA, this is a metadata-level operation (near-zero copy cost).
"""

import time
import torch
from transformers import AutoModelForCausalLM


class ModelOffloader:
    """
    Manages GPU memory by keeping only the active model on-device.
    On Apple Silicon Unified Memory, CPU<->GPU transfers are fast (~50-200ms)
    because they share the same physical memory pool.
    """
    def __init__(
        self,
        model_1b: AutoModelForCausalLM,
        model_3b: AutoModelForCausalLM,
        gpu_device: torch.device,
    ):
        self.model_1b = model_1b
        self.model_3b = model_3b
        self.gpu_device = gpu_device
        self.cpu_device = torch.device("cpu")
        self.active_model = None

    def activate_1b(self) -> float:
        """Moves 1B to GPU, 3B to CPU. Returns swap latency in ms."""
        if self.active_model == "1b":
            return 0.0
        t0 = time.perf_counter()
        self.model_3b.to(self.cpu_device)
        self.model_1b.to(self.gpu_device)
        if self.gpu_device.type == "mps":
            torch.mps.synchronize()
        swap_ms = (time.perf_counter() - t0) * 1000.0
        self.active_model = "1b"
        return swap_ms

    def activate_3b(self) -> float:
        """Moves 3B to GPU, 1B to CPU. Returns swap latency in ms."""
        if self.active_model == "3b":
            return 0.0
        t0 = time.perf_counter()
        self.model_1b.to(self.cpu_device)
        self.model_3b.to(self.gpu_device)
        if self.gpu_device.type == "mps":
            torch.mps.synchronize()
        swap_ms = (time.perf_counter() - t0) * 1000.0
        self.active_model = "3b"
        return swap_ms

    def activate_both(self) -> float:
        """Moves both models to GPU (for speculative decoding). Returns swap latency in ms."""
        if self.active_model == "both":
            return 0.0
        t0 = time.perf_counter()
        self.model_1b.to(self.gpu_device)
        self.model_3b.to(self.gpu_device)
        if self.gpu_device.type == "mps":
            torch.mps.synchronize()
        swap_ms = (time.perf_counter() - t0) * 1000.0
        self.active_model = "both"
        return swap_ms

    def get_memory_info(self) -> dict:
        """Returns current memory allocation status."""
        if self.gpu_device.type == "mps":
            allocated = torch.mps.current_allocated_memory() / (1024 ** 3)
            return {
                "active_model": self.active_model,
                "gpu_allocated_gb": round(allocated, 2),
            }
        return {"active_model": self.active_model}
