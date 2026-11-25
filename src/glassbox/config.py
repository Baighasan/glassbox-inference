import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal


Device = Literal["cpu", "cuda"]
Dtype = Literal["float32", "float16"]
SupportedModels = (
    "openai-community/gpt2", 
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

DEFAULT_MODEL = "openai-community/gpt2"
DEFAULT_DEVICE: Device = "cpu"
DEFAULT_DTYPE: Dtype = "float32"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 64
MAX_ALLOWED_TOKENS = 128
DEFAULT_LOGGING_LEVEL = "INFO"


@dataclass(frozen=True)
class Settings:
    model: str
    device: Device
    max_tokens: int
    logging_level: str = DEFAULT_LOGGING_LEVEL


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Load and validate Glassbox settings from environment variables."""
    source = os.environ if environ is None else environ

    model = source.get("GLASSBOX_MODEL", DEFAULT_MODEL).strip()
    if not model:
        raise ValueError("GLASSBOX_MODEL must not be empty")

    raw_device = source.get("GLASSBOX_DEVICE", DEFAULT_DEVICE).strip().lower()
    if raw_device == "cpu":
        device: Device = "cpu"
    elif raw_device == "cuda":
        device = "cuda"
    else:
        raise ValueError("GLASSBOX_DEVICE must be either 'cpu' or 'cuda'")

    raw_max_tokens = source.get(
        "GLASSBOX_MAX_TOKENS",
        str(DEFAULT_MAX_TOKENS),
    )
    try:
        max_tokens = int(raw_max_tokens)
    except ValueError as error:
        raise ValueError("GLASSBOX_MAX_TOKENS must be an integer") from error

    if not 1 <= max_tokens <= MAX_ALLOWED_TOKENS:
        raise ValueError(
            f"GLASSBOX_MAX_TOKENS must be between 1 and {MAX_ALLOWED_TOKENS}"
        )
    
    logging_level = source.get("GLASSBOX_LOGGING_LEVEL", DEFAULT_LOGGING_LEVEL).strip().upper()
    if logging_level not in {"DEBUG", "INFO",}:
        raise ValueError("GLASSBOX_LOGGING_LEVEL must be DEBUG or INFO")

    return Settings(
        model=model,
        device=device,
        max_tokens=max_tokens,
        logging_level=logging_level
    )
