import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import pytest
import torch

from conftest import ASGITestClient
from glassbox.config import DEFAULT_MODEL
from glassbox.engine.engine import CompletionResult


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


@dataclass(frozen=True)
class GenerateCall:
    model_name: str
    text: str
    device: str
    max_tokens: int
    temperature: float


class RecordingRuntime:
    def __init__(self) -> None:
        self.device = "cpu"
        self.dtype = torch.float32
        self.model: object | None = None
        self.model_name: str | None = None
        self.model_load_seconds: float | None = None

    def set_device(self, device: str) -> None:
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32

    def unload(self) -> None:
        self.model = None
        self.model_name = None
        self.model_load_seconds = None


class RecordingEngine:
    def __init__(self, device: str = "cpu") -> None:
        self.runtime = RecordingRuntime()
        self.runtime.set_device(device)
        self.calls: list[GenerateCall] = []
        self.finish_reason: Literal["stop", "length"] = "stop"

    @property
    def loaded_model(self) -> str | None:
        return self.runtime.model_name

    async def generate(
        self,
        model_name: str,
        text: str,
        device: str = "cpu",
        max_tokens: int = 64,
        temperature: float = 0,
    ) -> CompletionResult:
        self.calls.append(
            GenerateCall(
                model_name=model_name,
                text=text,
                device=device,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
        self.runtime.set_device(device)
        self.runtime.model_name = model_name
        self.runtime.model = object()
        self.runtime.model_load_seconds = 1.25
        return CompletionResult(
            generated_text="Paris",
            prompt_text=text,
            prompt_tokens=4,
            completion_tokens=1,
            completion_token_ids=(1,),
            finish_reason=self.finish_reason,
            latency_ms=12.5,
            tokens_per_second=80.0,
            gpu_memory_allocated_mb=None,
            gpu_peak_memory_allocated_mb=None,
            gpu_memory_reserved_mb=None,
        )

    def shutdown(self) -> None:
        self.runtime.unload()


@contextmanager
def recording_client(
    monkeypatch: pytest.MonkeyPatch,
    environ: dict[str, str] | None = None,
) -> Iterator[tuple[ASGITestClient, RecordingEngine]]:
    for name in ("GLASSBOX_MODEL", "GLASSBOX_DEVICE", "GLASSBOX_MAX_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    if environ is not None:
        for key, value in environ.items():
            monkeypatch.setenv(key, value)
    # The server package re-exports the FastAPI app as `app`, which hides the
    # submodule from a dotted monkeypatch path.
    app_module = sys.modules["glassbox.server.app"]
    monkeypatch.setattr(app_module, "Engine", RecordingEngine)
    client = ASGITestClient()
    try:
        engine = client.app.state.engine
        assert isinstance(engine, RecordingEngine)
        yield client, engine
    finally:
        client.close()


def test_completion_returns_engine_text_usage_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with recording_client(monkeypatch) as (client, engine):
        engine.finish_reason = "stop"
        response = client.post(
            "/v1/completions",
            json={"model": DEFAULT_MODEL, "prompt": "Hello"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["model"] == DEFAULT_MODEL
    assert body["choices"] == [
        {
            "text": "Paris",
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop",
        }
    ]
    assert body["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 1,
        "total_tokens": 5,
    }
    assert body["glassbox_metrics"] == {
        "total_latency_ms": 12.5,
        "tokens_per_second": 80.0,
        "device": "cpu",
        "dtype": "float32",
        "model_load_seconds": 1.25,
        "gpu_memory_allocated_mb": None,
        "gpu_peak_memory_allocated_mb": None,
        "gpu_memory_reserved_mb": None,
    }


def test_completion_reports_length_when_generation_hits_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with recording_client(monkeypatch) as (client, engine):
        engine.finish_reason = "length"
        response = client.post(
            "/v1/completions",
            json={"model": DEFAULT_MODEL, "prompt": "Hello"},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "length"


def test_health_reports_the_loaded_runtime_after_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with recording_client(monkeypatch) as (client, _engine):
        client.post(
            "/v1/completions",
            json={"model": DEFAULT_MODEL, "prompt": "Hello"},
        )
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "alive",
        "loaded": True,
        "model": DEFAULT_MODEL,
        "device": "cpu",
        "dtype": "float32",
        "model_load_seconds": 1.25,
    }


def test_second_completion_keeps_the_same_loaded_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with recording_client(monkeypatch) as (client, engine):
        payload = {"model": DEFAULT_MODEL, "prompt": "Hello"}
        client.post("/v1/completions", json=payload)
        client.post("/v1/completions", json=payload)
        health = client.get("/health")

    assert len(engine.calls) == 2
    assert health.json()["loaded"] is True
    assert health.json()["model"] == DEFAULT_MODEL


def test_omitted_generation_fields_use_process_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with recording_client(
        monkeypatch,
        {"GLASSBOX_DEVICE": "cuda", "GLASSBOX_MAX_TOKENS": "32"},
    ) as (client, engine):
        before = client.get("/health")
        response = client.post(
            "/v1/completions",
            json={"model": DEFAULT_MODEL, "prompt": "Hello"},
        )

    assert before.json()["loaded"] is False
    assert before.json()["device"] == "cuda"
    assert before.json()["dtype"] == "float16"
    assert response.status_code == 200
    assert engine.calls[0].device == "cuda"
    assert engine.calls[0].max_tokens == 32


def test_request_device_and_max_tokens_override_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with recording_client(
        monkeypatch,
        {"GLASSBOX_DEVICE": "cuda", "GLASSBOX_MAX_TOKENS": "32"},
    ) as (client, engine):
        response = client.post(
            "/v1/completions",
            json={
                "model": DEFAULT_MODEL,
                "prompt": "Hello",
                "device": "cpu",
                "max_tokens": 4,
            },
        )

    assert response.status_code == 200
    assert engine.calls[0].device == "cpu"
    assert engine.calls[0].max_tokens == 4

