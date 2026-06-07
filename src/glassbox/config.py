import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal


DEFAULT_MODEL = "gpt2"
DEFAULT_DEVICE = "cpu"
DEFAULT_MAX_TOKENS = 128
MAX_ALLOWED_TOKENS = 128


@dataclass(frozen=True)
class Settings:
    model: str
    device: Literal["cpu", "cuda"]
    max_tokens: int


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Load and validate Glassbox settings from environment variables."""
    source = os.environ if environ is None else environ

    model = source.get("GLASSBOX_MODEL", DEFAULT_MODEL).strip()
    if not model:
        raise ValueError("GLASSBOX_MODEL must not be empty")

    device = source.get("GLASSBOX_DEVICE", DEFAULT_DEVICE).strip().lower()
    if device not in {"cpu", "cuda"}:
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

    return Settings(
        model=model,
        device=device,
        max_tokens=max_tokens,
    )
