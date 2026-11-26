import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from glassbox.config import (
    DEFAULT_DEVICE,
    Device,
)

from typing import Literal


@dataclass(frozen=True)
class CompletionResult:
    logits: torch.Tensor
    gpu_memory_allocated_mb: float | None
    gpu_peak_memory_allocated_mb: float | None
    gpu_memory_reserved_mb: float | None


class Runtime:
    """Owns model weights forward pass."""

    def __init__(self) -> None:
        self.model: PreTrainedModel | None = None
        self.model_name: str | None = None
        self.model_load_seconds: float | None = None
        self.device: Device = DEFAULT_DEVICE
        self.dtype: Literal[torch.float32, torch.float16] = torch.float32

    def set_device(self, device: Device) -> None:
        #self.dtype = torch.float16 if device == "cuda" else torch.float32 # TODO optimize fp16 on quadro t2000
        self.device = device
        

    def load(self, model_name: str) -> PreTrainedModel:
        """Load weights for model_name, reusing them when that model is already resident."""
        if self.model is not None and self.model_name == model_name:
            # if model already loaded on requested device
            if self._weights_are_placed():
                return self.model
            started_at = time.perf_counter()
            # move weights to requested device
            self.model.to(device=self.device, dtype=self.dtype)
            self.model_load_seconds = time.perf_counter() - started_at
            return self.model

        started_at = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=self.dtype).to(
            self.device
        )
        self.model = model
        self.model_name = model_name
        self.model_load_seconds = time.perf_counter() - started_at
        return model

    def _weights_are_placed(self) -> bool:
        """Return whether the resident weights already use the selected device and dtype."""
        if self.model is None:
            return False
        parameter = next(self.model.parameters(), None)
        if parameter is None:
            return False
        return parameter.device.type == self.device and parameter.dtype == self.dtype

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> CompletionResult:
        """Return logits and device memory stats for the given inputs."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        gpu_memory_allocated_mb = None
        gpu_peak_memory_allocated_mb = None
        gpu_memory_reserved_mb = None

        with torch.inference_mode():
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            if self.device == "cuda":
                torch.cuda.synchronize()
                gpu_memory_allocated_mb = torch.cuda.memory_allocated() / (1024**2)
                gpu_peak_memory_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
                gpu_memory_reserved_mb = torch.cuda.memory_reserved() / (1024**2)

        return CompletionResult(
            logits=outputs.logits,
            gpu_memory_allocated_mb=gpu_memory_allocated_mb,
            gpu_peak_memory_allocated_mb=gpu_peak_memory_allocated_mb,
            gpu_memory_reserved_mb=gpu_memory_reserved_mb,
        )

    def unload(self) -> None:
        """Drop the loaded weights."""
        self.model = None
        self.model_name = None
        self.model_load_seconds = None
