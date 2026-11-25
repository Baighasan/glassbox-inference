import pytest

from glassbox.config import Settings, load_settings


def test_load_settings_uses_defaults() -> None:
    assert load_settings({}) == Settings(
        model="openai-community/gpt2",
        device="cpu",
        max_tokens=64,
    )


def test_load_settings_reads_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GLASSBOX_MODEL", "example/model")
    monkeypatch.setenv("GLASSBOX_DEVICE", "cuda")
    monkeypatch.setenv("GLASSBOX_MAX_TOKENS", "64")

    settings = load_settings()

    assert settings == Settings(
        model="example/model",
        device="cuda",
        max_tokens=64,
    )


@pytest.mark.parametrize("device", ["gpu", "mps", ""])
def test_load_settings_rejects_unsupported_device(device: str) -> None:
    with pytest.raises(ValueError, match="GLASSBOX_DEVICE"):
        load_settings({"GLASSBOX_DEVICE": device})


@pytest.mark.parametrize("max_tokens", ["not-an-integer", "0", "129"])
def test_load_settings_rejects_invalid_max_tokens(max_tokens: str) -> None:
    with pytest.raises(ValueError, match="GLASSBOX_MAX_TOKENS"):
        load_settings({"GLASSBOX_MAX_TOKENS": max_tokens})


@pytest.mark.parametrize("model", ["", "   ", "\t"])
def test_load_settings_rejects_blank_model(model: str) -> None:
    with pytest.raises(ValueError, match="GLASSBOX_MODEL"):
        load_settings({"GLASSBOX_MODEL": model})


def test_load_settings_strips_model_whitespace() -> None:
    settings = load_settings({"GLASSBOX_MODEL": "  example/model  "})

    assert settings.model == "example/model"


@pytest.mark.parametrize("max_tokens", ["1", "128"])
def test_load_settings_accepts_max_token_bounds(max_tokens: str) -> None:
    settings = load_settings({"GLASSBOX_MAX_TOKENS": max_tokens})

    assert settings.max_tokens == int(max_tokens)


@pytest.mark.parametrize(("raw", "device"), [("CPU", "cpu"), ("CUDA", "cuda")])
def test_load_settings_normalizes_device_case(raw: str, device: str) -> None:
    settings = load_settings({"GLASSBOX_DEVICE": raw})

    assert settings.device == device
