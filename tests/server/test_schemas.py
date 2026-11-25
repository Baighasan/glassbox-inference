import pytest
from pydantic import ValidationError

from glassbox.server.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
)


def test_completion_request_leaves_device_and_max_tokens_unset() -> None:
    request = CompletionRequest(model="test-model", prompt="Hello")

    assert request.device is None
    assert request.max_tokens is None
    assert request.temperature == 0
    assert request.stream is False


def test_chat_completion_request_uses_generation_defaults() -> None:
    request = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Hello")],
    )

    assert request.max_tokens == 64
    assert request.temperature == 0
    assert request.n == 1
    assert request.stream is False


@pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, -0.1])
def test_completion_request_rejects_nonzero_temperature(temperature: float) -> None:
    with pytest.raises(ValidationError):
        CompletionRequest(model="test-model", prompt="Hello", temperature=temperature)


@pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, -0.1])
def test_chat_completion_request_rejects_nonzero_temperature(
    temperature: float,
) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=temperature,
        )
