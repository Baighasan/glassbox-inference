from glassbox.server.schemas import ChatCompletionRequest, CompletionRequest


def test_completion_request_uses_generation_defaults() -> None:
    request = CompletionRequest(model="test-model", prompt="Hello")

    assert request.max_tokens == 64
    assert request.temperature == 0
    assert request.n == 1
    assert request.stream is False


def test_chat_completion_request_uses_generation_defaults() -> None:
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert request.max_tokens == 64
    assert request.temperature == 0
    assert request.n == 1
    assert request.stream is False
