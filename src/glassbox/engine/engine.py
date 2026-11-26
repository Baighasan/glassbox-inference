import logging

import time
from dataclasses import dataclass
from typing import Literal

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase, Cache

from glassbox.config import (
    DEFAULT_DEVICE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_MODEL,
    SupportedModels,
    Device,
)
from glassbox.engine.exceptions import UnsupportedModelError
from glassbox.runtime.runtime import Runtime

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class CompletionResult:
    generated_text: str
    prompt_text: str
    prompt_tokens: int
    completion_tokens: int
    completion_token_ids: tuple[int, ...]
    finish_reason: Literal["stop", "length"]
    latency_ms: float
    tokens_per_second: float
    gpu_memory_allocated_mb: float | None
    gpu_peak_memory_allocated_mb: float | None
    gpu_memory_reserved_mb: float | None

class Engine:
    """Tokenizes a prompt, calls the runtime, and decodes new token ids."""

    def __init__(self, device: Device = DEFAULT_DEVICE) -> None:
        self.runtime = Runtime()
        self.runtime.set_device(device)
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
        device: Device = DEFAULT_DEVICE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> CompletionResult:
        """Generate completion text for one request."""
        if model_name not in SupportedModels:
            raise UnsupportedModelError(model_name)

        # check if tokenizer is already initialized
        if self.tokenizer is None or self.tokenizer_name != model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer_name = model_name
            logger.info(f"Tokenizer initialized for model: {model_name}")

        self.runtime.set_device(device)
        self.runtime.load(model_name)
        logger.info("Runtime loaded model: %s on device: %s", model_name, device)

        # initialize metric variables
        started_at = time.perf_counter()
        gpu_peak_memory_allocated_mb = None

        # tokenize the input, applying the chat template and adding the generation prompt
        tokenized_input = None
        if self.tokenizer.chat_template: 
            tokenized_input = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
        else:
            tokenized_input = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=True,
            )

        # get the input ids and attention mask
        input_ids = tokenized_input.input_ids
        attention_mask = tokenized_input.attention_mask

        prompt_length = input_ids.shape[1]

        # None means this tokenizer has no stop id. Integers and lists both occur.
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_ids: set[int] = set()
        elif isinstance(eos_id, int):
            eos_ids = {eos_id}
        else:
            eos_ids = {int(token_id) for token_id in eos_id}
        
        logger.debug("Input IDs for model %s: %s", model_name, input_ids.tolist())
        logger.debug("eos_ids for model %s: %s", model_name, eos_ids)
        
        kv_cache: Cache | None = None
        step_token_id: torch.Tensor | None = None

        finish_reason: Literal["stop", "length"] = "length"
        for _ in range(max_tokens):
            # get the logits for the input
            step_input_ids = input_ids if kv_cache is None else input_ids[:, -1:]
            output = self.runtime.forward(step_input_ids, attention_mask, kv_cache)
            kv_cache = output.past_key_values

            logits = output.logits
            if self.runtime.device == "cuda":
                gpu_peak_memory_allocated_mb = (
                    max(gpu_peak_memory_allocated_mb, output.gpu_peak_memory_allocated_mb)
                    if gpu_peak_memory_allocated_mb is not None
                    else output.gpu_peak_memory_allocated_mb
                )

            # Greedy decoding: take logits of the last token and select the highest value
            step_token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True).to(input_ids.device)

            # Append predicted token to the full sequence
            input_ids = torch.cat([input_ids, step_token_id], dim=-1)

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

            # An EOS token on the last allowed step finishes as stop.
            if step_token_id.item() in eos_ids:
                finish_reason = "stop"
                break
        
        logger.info("Generation completed in %.2f seconds with finish reason: %s", time.perf_counter() - started_at, finish_reason)
        # Return the generated tokens slicing off the prompt
        generated_ids = input_ids[:, prompt_length:]
        completion_token_ids = tuple(int(token_id) for token_id in generated_ids[0].tolist())
        logger.debug("Generated token IDs: %s", completion_token_ids)


        # decode the generated ids
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        logger.debug("Generated text: %s", generated_text)

        elapsed_seconds = time.perf_counter() - started_at
        completion_tokens = generated_ids.shape[1]
        completion_result = CompletionResult(
            generated_text=generated_text,
            prompt_text=text,
            prompt_tokens=prompt_length,
            completion_tokens=completion_tokens,
            completion_token_ids=completion_token_ids,
            finish_reason=finish_reason,
            latency_ms=elapsed_seconds * 1000,
            tokens_per_second=completion_tokens / max(elapsed_seconds, 1e-9),
            gpu_memory_allocated_mb=output.gpu_memory_allocated_mb,
            gpu_peak_memory_allocated_mb=gpu_peak_memory_allocated_mb,
            gpu_memory_reserved_mb=output.gpu_memory_reserved_mb,
        )
        return completion_result

    def shutdown(self) -> None:
        """Release the tokenizer and the runtime's weights."""
        self.tokenizer = None
        self.tokenizer_name = None
        self.runtime.unload()
        logger.info("Engine shutdown complete.")