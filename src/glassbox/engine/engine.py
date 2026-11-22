from glassbox.config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from transformers import AutoTokenizer
from glassbox.config import DEFAULT_MODEL, DEFAULT_DEVICE
from glassbox.runtime.runtime import runtime

async def engine(
    model_name: str = DEFAULT_MODEL,
    text: str = "",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """ Generate text using the model. """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEFAULT_DEVICE)

    generated_ids = runtime(input_ids, model_name, max_tokens, temperature)

    completion = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return completion