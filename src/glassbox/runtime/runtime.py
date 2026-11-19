from glassbox.config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE
from glassbox.runtime.model import ModelManager


async def generate(
    model_manager: ModelManager,
    model_name: str = "openai-community/gpt2",
    text: str = "",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """ Generate text using the model. """
    generator = await model_manager.get_model(task="text-generation", model_name=model_name)

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": False if temperature == 0.0 else True,
    }
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature

    # Generate text
    raw_output = generator(text, **gen_kwargs)
    cleaned_output = raw_output[0]["generated_text"]

    return cleaned_output