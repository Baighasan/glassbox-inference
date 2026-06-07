import pytest

from conftest import ASGITestClient


def test_chat_completion_returns_openai_style_mock_response(
    client: ASGITestClient,
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("chatcmpl-")
    assert body["object"] == "chat.completion"
    assert body["model"] == "test-model"
    assert body["choices"] == [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a simulated mock response",
            },
            "finish_reason": "stop",
        }
    ]
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )
    assert body["glassbox_metrics"]["device"] == "cpu"
    assert body["glassbox_metrics"]["dtype"] == "float32"


def test_chat_completion_rejects_unavailable_model(
    client: ASGITestClient,
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "other-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 404


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 0.5),
        ("top_p", 0.9),
        ("n", 2),
        ("stream", True),
        ("max_tokens", 0),
        ("max_tokens", 129),
        ("tools", []),
        ("tool_choice", "auto"),
        ("functions", []),
        ("function_call", "auto"),
    ],
)
def test_chat_completion_rejects_unsupported_or_invalid_fields(
    client: ASGITestClient,
    field: str,
    value: object,
) -> None:
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        field: value,
    }

    response = client.post("/v1/chat/completions", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "invalid", "content": "Hello"}],
        [{"role": "user", "content": ""}],
    ],
)
def test_chat_completion_rejects_invalid_messages(
    client: ASGITestClient,
    messages: list[dict[str, str]],
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "test-model", "messages": messages},
    )

    assert response.status_code == 422


def test_chat_completion_endpoint_requires_post(
    client: ASGITestClient,
) -> None:
    response = client.get("/v1/chat/completions")

    assert response.status_code == 405
