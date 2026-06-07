import pytest

from conftest import ASGITestClient


def test_completion_returns_openai_style_mock_response(
    client: ASGITestClient,
) -> None:
    prompt = "A prompt that must not be echoed"

    response = client.post(
        "/v1/completions",
        json={"model": "test-model", "prompt": prompt},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"].startswith("cmpl-")
    assert body["object"] == "text_completion"
    assert body["model"] == "test-model"
    assert body["choices"] == [
        {
            "text": "This is a simulated completion.",
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop",
        }
    ]
    assert prompt not in body["choices"][0]["text"]
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )
    assert body["glassbox_metrics"]["device"] == "cpu"
    assert body["glassbox_metrics"]["dtype"] == "float32"


def test_completion_rejects_unavailable_model(
    client: ASGITestClient,
) -> None:
    response = client.post(
        "/v1/completions",
        json={"model": "other-model", "prompt": "Hello"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Model 'other-model' is not available"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 0.5),
        ("top_p", 0.9),
        ("n", 2),
        ("stream", True),
        ("max_tokens", 0),
        ("max_tokens", 129),
        ("echo", True),
        ("tools", []),
        ("functions", []),
    ],
)
def test_completion_rejects_unsupported_or_invalid_fields(
    client: ASGITestClient,
    field: str,
    value: object,
) -> None:
    payload = {
        "model": "test-model",
        "prompt": "Hello",
        field: value,
    }

    response = client.post("/v1/completions", json=payload)

    assert response.status_code == 422


def test_completion_requires_string_prompt(client: ASGITestClient) -> None:
    response = client.post(
        "/v1/completions",
        json={"model": "test-model", "prompt": ["Hello", "world"]},
    )

    assert response.status_code == 422


def test_completion_endpoint_requires_post(client: ASGITestClient) -> None:
    response = client.get("/v1/completions")

    assert response.status_code == 405
