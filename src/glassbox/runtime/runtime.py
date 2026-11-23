import time

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from glassbox.config import DEFAULT_DEVICE


class Runtime:
    """Owns model weights and the black-box generation call."""

    def __init__(self) -> None:
        self.model: PreTrainedModel | None = None
        self.model_name: str | None = None
        self.model_load_seconds: float | None = None

    def load(self, model_name: str) -> PreTrainedModel:
        """Load weights for model_name, reusing them when that model is already resident."""
        if self.model is not None and self.model_name == model_name:
            return self.model

        started_at = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(model_name).to(DEFAULT_DEVICE)
        self.model = model
        self.model_name = model_name
        self.model_load_seconds = time.perf_counter() - started_at
        return model

    def generate(
        self,
        input_ids: torch.Tensor,
        model_name: str,
        max_tokens: int,
        temperature: float,
    ) -> torch.Tensor:
        """Return new token ids for an already-tokenized prompt."""
        model = self.load(model_name)

        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        prompt_length = input_ids.shape[1]
        return generated_ids[:, prompt_length:]

    def unload(self) -> None:
        """Drop the loaded weights."""
        self.model = None
        self.model_name = None
        self.model_load_seconds = None
