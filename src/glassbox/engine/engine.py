from transformers import AutoTokenizer, PreTrainedTokenizerBase

from glassbox.config import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_TEMPERATURE
from glassbox.engine.exceptions import UnsupportedModelError
from glassbox.runtime.runtime import Runtime

import time
from dataclasses import dataclass

@dataclass(frozen=True)
class CompletionResult:
    generated_text: str
    prompt_text: str
    prompt_tokens: int
    completion_tokens: int
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

        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(
            self.runtime.device
        )
        generated_ids = self.runtime.generate(
            input_ids,
            model_name,
            max_tokens,
            temperature,
        )
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        elapsed_seconds = time.perf_counter() - started_at
        completion_tokens = generated_ids.shape[1]
        completion_result = CompletionResult(
            generated_text=generated_text,
            prompt_text=text,
            prompt_tokens=input_ids.shape[1],
            completion_tokens=completion_tokens,
            latency_ms=elapsed_seconds * 1000,
            tokens_per_second=completion_tokens / max(elapsed_seconds, 1e-9),
        )
        return completion_result

    def shutdown(self) -> None:
        """Release the tokenizer and the runtime's weights."""
        self.tokenizer = None
        self.tokenizer_name = None
        self.runtime.unload()