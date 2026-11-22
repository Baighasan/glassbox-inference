from transformers import AutoTokenizer, PreTrainedTokenizerBase

from glassbox.config import DEFAULT_DEVICE
from glassbox.runtime.runtime import Runtime


class Engine:
    """Tokenizes a prompt, calls the runtime, and decodes new token ids."""

    def __init__(self) -> None:
        self.runtime = Runtime()
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.tokenizer_name: str | None = None

    @property
    def loaded_model(self) -> str | None:
        """The model id held by the runtime, or None before the first load."""
        return self.runtime.model_name

    async def generate(
        self,
        model_name: str,
        text: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate completion text for one request."""
        if self.tokenizer is None or self.tokenizer_name != model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer_name = model_name

        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(DEFAULT_DEVICE)
        generated_ids = self.runtime.generate(
            input_ids,
            model_name,
            max_tokens,
            temperature,
        )
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def shutdown(self) -> None:
        """Release the tokenizer and the runtime's weights."""
        self.tokenizer = None
        self.tokenizer_name = None
        self.runtime.unload()
