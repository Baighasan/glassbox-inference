import time

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from glassbox.config import DEFAULT_DEVICE, DEFAULT_DTYPE, Device, Dtype


class Runtime:
    """Owns model weights forward pass."""

    def __init__(self) -> None:
        self.model: PreTrainedModel | None = None
        self.model_name: str | None = None
        self.model_load_seconds: float | None = None
        self.device: Device = DEFAULT_DEVICE
        self.dtype: Dtype = DEFAULT_DTYPE

    def load(self, model_name: str) -> PreTrainedModel:
        """Load weights for model_name, reusing them when that model is already resident."""
        if self.model is not None and self.model_name == model_name:
            return self.model

        started_at = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model = model
        self.model_name = model_name
        self.model_load_seconds = time.perf_counter() - started_at
        return model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return logits for the given inputs."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits
        return logits

    def unload(self) -> None:
        """Drop the loaded weights."""
        self.model = None
        self.model_name = None
        self.model_load_seconds = None
