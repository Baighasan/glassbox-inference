from transformers import pipeline
from glassbox.config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE


def generate(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    # Initialize the pipeline with the base gpt2 model
    generator = pipeline(
        "text-generation",
        model="openai-community/gpt2",
        device="cpu",
    )

    # Build dynamic config
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
