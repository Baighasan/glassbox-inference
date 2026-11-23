import pytest

from conftest import ASGITestClient


def test_completion_rejects_unavailable_model(
    client: ASGITestClient,
) -> None:
    response = client.post(
        "/v1/completions",
        json={"model": "other-model", "prompt": "Hello"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Model other-model not found"


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
