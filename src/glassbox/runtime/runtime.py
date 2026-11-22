from transformers import AutoModelForCausalLM
from glassbox.config import DEFAULT_MODEL, DEFAULT_DEVICE, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
import torch

model: AutoModelForCausalLM | None = None

# runtime should take input ids, and return output ids. we may also need to take the model name
def runtime(
    input_ids: torch.Tensor,
    model_name: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """ Generate text using the model. """
    global model
    if not model or model.name_or_path != model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(DEFAULT_DEVICE)
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        )
    
    prompt_length = input_ids.shape[1]
    generated_ids = generated_ids[:, prompt_length:]    # Remove the prompt from the generated ids

    return generated_ids