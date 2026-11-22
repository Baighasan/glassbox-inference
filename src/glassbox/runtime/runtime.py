from transformers import AutoModelForCausalLM
from glassbox.config import DEFAULT_MODEL, DEFAULT_DEVICE, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
import torch


# runtime should take input ids, and return output ids. we may also need to take the model name
def runtime(
    input_ids: torch.Tensor,
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """ Generate text using the model. """
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEFAULT_DEVICE)
    generated_ids = model.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature)

    return generated_ids