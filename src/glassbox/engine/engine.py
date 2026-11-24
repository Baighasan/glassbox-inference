from transformers import AutoTokenizer, PreTrainedTokenizerBase

from glassbox.config import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_TEMPERATURE
from glassbox.engine.exceptions import UnsupportedModelError
from glassbox.runtime.runtime import Runtime

import time
from dataclasses import dataclass
from typing import Any
import torch

@dataclass(frozen=True)
class CompletionResult:
    generated_text: str
    prompt_text: str
    prompt_tokens: int
    completion_tokens: int
    completion_token_ids: tuple[int, ...]
    latency_ms: float
    tokens_per_second: float

class Engine:
    """Tokenizes a prompt, calls the runtime, and decodes new token ids."""

    def __init__(self) -> None:
        self.runtime = Runtime()
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.tokenizer_name: str | None = None

    @property
    def loaded_model(self) -> str | None:
        """The model id held by the runtime, or None after shutdown."""
        return self.runtime.model_name

    async def generate(
        self,
        model_name: str,
        text: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> CompletionResult:
        """Generate completion text for one request."""
        if model_name != DEFAULT_MODEL:
            raise UnsupportedModelError(model_name)

        if self.tokenizer is None or self.tokenizer_name != model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer_name = model_name

        self.runtime.load(model_name)

        started_at = time.perf_counter()

        # tokenize the input
        tokenized_input = self.tokenizer(text, return_tensors="pt")

        # get the input ids and attention mask
        input_ids = tokenized_input.input_ids
        attention_mask = tokenized_input.attention_mask

        prompt_length = input_ids.shape[1]

        # get stop token id
        eos_id = self.tokenizer.eos_token_id

        # Handle both single integers (e.g., 50256) and lists of integers (e.g., Llama 3)
        # We won't run into lists of integers for now, but this is the correct way to handle it
        eos_ids = {eos_id} if isinstance(eos_id, int) else set[Any](eos_id)

         # generate tokens
        for _ in range(max_tokens):
            # get the logits for the input     
            logits = self.runtime.forward(input_ids, attention_mask)

            # Greedy decoding: take logits of the last token and select the highest value
            next_token_logits = logits[:, -1, :]    # tensor is [batch, vocab]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True).to(input_ids.device)

            # Append predicted token to the full sequence
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

            # Extend attention_mask by 1 to account for the new token
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1), 
                        device=attention_mask.device,
                        dtype=attention_mask.dtype
                    )
                ],
                dim=-1,
            )

            # Stop if End-of-Sequence token is generated
            if next_token_id.item() in eos_ids:
                break

        # Return the generated tokens slicing off the prompt
        generated_ids = input_ids[:, prompt_length:]
        completion_token_ids = tuple(int(token_id) for token_id in generated_ids[0].tolist())

        # decode the generated ids
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        elapsed_seconds = time.perf_counter() - started_at
        completion_tokens = generated_ids.shape[1]
        completion_result = CompletionResult(
            generated_text=generated_text,
            prompt_text=text,
            prompt_tokens=prompt_length,
            completion_tokens=completion_tokens,
            completion_token_ids=completion_token_ids,
            latency_ms=elapsed_seconds * 1000,
            tokens_per_second=completion_tokens / max(elapsed_seconds, 1e-9),
        )
        return completion_result

    def shutdown(self) -> None:
        """Release the tokenizer and the runtime's weights."""
        self.tokenizer = None
        self.tokenizer_name = None
        self.runtime.unload()