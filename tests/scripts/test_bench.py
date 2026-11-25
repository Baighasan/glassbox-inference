import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script(filename: str) -> ModuleType:
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="http://127.0.0.1:8000",
    )


def _completion_payload(device: str = "cpu") -> dict[str, object]:
    return {
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "total_tokens": 6,
        },
        "glassbox_metrics": {
            "total_latency_ms": 10.0,
            "tokens_per_second": 100.0 if device == "cpu" else 250.0,
            "device": device,
            "dtype": "float32" if device == "cpu" else "float16",
            "model_load_seconds": 0.5,
            "gpu_memory_allocated_mb": None if device == "cpu" else 100.0,
            "gpu_peak_memory_allocated_mb": None if device == "cpu" else 120.0,
            "gpu_memory_reserved_mb": None if device == "cpu" else 200.0,
        },
    }


def test_bench_suite_builds_one_row_per_prompt() -> None:
    bench = _load_script("bench.py")
    prompts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "alive"})
        body = json.loads(request.content)
        prompts.append(body["prompt"])
        return httpx.Response(200, json=_completion_payload())

    with _client(handler) as client:
        rows = [
            bench._complete(client, name, prompt) for name, prompt in bench.PROMPTS
        ]

    assert [row.name for row in rows] == ["short", "medium", "chat"]
    assert prompts == [prompt for _name, prompt in bench.PROMPTS]
    assert rows[0].prompt_tokens == 4
    assert rows[0].completion_tokens == 2


def test_bench_rejects_a_failed_request() -> None:
    bench = _load_script("bench.py")

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(503, text="down")

    with _client(handler) as client:
        with pytest.raises(SystemExit):
            bench._complete(client, "short", "Hello")


def test_bench_rejects_a_response_without_usage() -> None:
    bench = _load_script("bench.py")

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json={"glassbox_metrics": {}})

    with _client(handler) as client:
        with pytest.raises(SystemExit):
            bench._complete(client, "short", "Hello")


def test_cpu_cuda_bench_leaves_warmup_out_of_the_table() -> None:
    bench = _load_script("bench_cpu_cuda.py")
    max_tokens: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/completions":
            return httpx.Response(200, json={"status": "alive"})
        body = json.loads(request.content)
        max_tokens.append(body["max_tokens"])
        return httpx.Response(200, json=_completion_payload(body["device"]))

    with _client(handler) as client:
        rows = bench._run_device(client, "cpu")

    assert max_tokens[0] == bench.WARMUP_MAX_TOKENS
    assert max_tokens[1:] == [bench.MAX_TOKENS, bench.MAX_TOKENS, bench.MAX_TOKENS]
    assert [row.name for row in rows] == ["short", "medium", "chat"]


def test_cpu_cuda_bench_rejects_a_device_mismatch() -> None:
    bench = _load_script("bench_cpu_cuda.py")

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json=_completion_payload("cuda"))

    with _client(handler) as client:
        with pytest.raises(SystemExit):
            bench._complete(client, "short", "Hello", "cpu", 32)


def test_cpu_cuda_report_includes_speedup(capsys: pytest.CaptureFixture[str]) -> None:
    bench = _load_script("bench_cpu_cuda.py")
    cpu = bench.BenchRow(
        name="short",
        prompt_tokens=4,
        completion_tokens=2,
        latency_ms=20.0,
        tokens_per_second=10.0,
        device="cpu",
        dtype="float32",
        model_load_seconds=0.5,
        gpu_memory_allocated_mb=None,
        gpu_peak_memory_allocated_mb=None,
        gpu_memory_reserved_mb=None,
    )
    cuda = bench.BenchRow(
        name="short",
        prompt_tokens=4,
        completion_tokens=2,
        latency_ms=8.0,
        tokens_per_second=20.0,
        device="cuda",
        dtype="float16",
        model_load_seconds=0.4,
        gpu_memory_allocated_mb=100.0,
        gpu_peak_memory_allocated_mb=120.0,
        gpu_memory_reserved_mb=200.0,
    )

    bench._print_report([cpu], [cuda])

    output = capsys.readouterr().out
    assert "2.00x" in output
    assert "speedup" in output
