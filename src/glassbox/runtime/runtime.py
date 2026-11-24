import time

import torch
from transformers import AutoModelForCausalLM, BatchEncoding, PreTrainedModel

from glassbox.config import DEFAULT_DEVICE, DEFAULT_DTYPE, Device, Dtype
from typing import Any


class Runtime:
    """Owns model weights and the black-box generation call."""

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

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        model_name: str,
        max_tokens: int,
        temperature: float,
    ) -> torch.Tensor:
        """Return new token ids for an already-tokenized prompt."""
        # move the input ids and attention mask to the device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        #past_key_values = None    # TODO kv cache

        prompt_length = input_ids.shape[1]
        
        # load the model
        model = self.load(model_name) 

        # Look up the stop integer(s) directly from the model's config
        eos_id = model.config.eos_token_id

        # Handle both single integers (e.g., 50256) and lists of integers (e.g., Llama 3)
        # We won't run into lists of integers for now, but this is the correct way to handle it
        eos_ids = {eos_id} if isinstance(eos_id, int) else set[Any](eos_id)

        # generate tokens
        with torch.no_grad():
            for _ in range(max_tokens):
                # get the last token from each batch                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Greedy decoding: take logits of the last token and select the highest value
                next_token_logits = outputs.logits[:, -1, :]    # tensor is [batch, vocab]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True).to(self.device)

                # Append predicted token to the full sequence
                input_ids = torch.cat([input_ids, next_token_id], dim=-1)

                # Extend attention_mask by 1 to account for the new token
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=torch.long)],
                    dim=-1,
                )

                # Stop if End-of-Sequence token is generated
                if next_token_id.item() in eos_ids:
                    break

        # Return the generated tokens slicing off the prompt
        return input_ids[:, prompt_length:]

    def unload(self) -> None:
        """Drop the loaded weights."""
        self.model = None
        self.model_name = None
        self.model_load_seconds = None
