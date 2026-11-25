"""Compare CPU and CUDA for the fixed Glassbox prompt suite.

Start the server, then run:

    uv run python scripts/bench_cpu_cuda.py

Each device gets one unmeasured warmup completion so weight movement and
CUDA startup stay out of the table. GLASSBOX_URL overrides the server.
"""

import os
import sys
from dataclasses import dataclass

import httpx

BASE_URL = os.environ.get("GLASSBOX_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL = "openai-community/gpt2"
MAX_TOKENS = 32
WARMUP_MAX_TOKENS = 1
TEMPERATURE = 0
REQUEST_TIMEOUT_SECONDS = 600.0
DEVICES = ("cpu", "cuda")

PROMPTS: tuple[tuple[str, str], ...] = (
    ("short", "The capital of France is"),
    (
        "medium",
        "Glassbox is a small local inference server. It loads one model, "
        "formats a prompt, and generates the next tokens on CPU. This "
        "benchmark sends the same request shape three times and records "
        "how long each completion takes.",
    ),
    ("chat", "User: Name three primary colors.\nAssistant:"),
)


@dataclass(frozen=True)
class BenchRow:
    name: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    tokens_per_second: float
    device: str
    dtype: str
    model_load_seconds: float | None
    gpu_memory_allocated_mb: float | None
    gpu_peak_memory_allocated_mb: float | None
    gpu_memory_reserved_mb: float | None


def main() -> None:
    """Check the server, run the suite on CPU and CUDA, and print the comparison."""
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)
    with httpx.Client(base_url=BASE_URL, timeout=timeout) as client:
        _check_health(client)
        by_device = {
            device: _run_device(client, device)
            for device in DEVICES
        }
    _print_report(by_device["cpu"], by_device["cuda"])


def _run_device(client: httpx.Client, device: str) -> list[BenchRow]:
    _complete(client, "warmup", PROMPTS[0][1], device, WARMUP_MAX_TOKENS)
    return [
        _complete(client, name, prompt, device, MAX_TOKENS)
        for name, prompt in PROMPTS
    ]


def _check_health(client: httpx.Client) -> None:
    _request(client, "GET", "/health")


def _complete(
    client: httpx.Client,
    name: str,
    prompt: str,
    device: str,
    max_tokens: int,
) -> BenchRow:
    response = _request(
        client,
        "POST",
        "/v1/completions",
        body={
            "model": MODEL,
            "prompt": prompt,
            "device": device,
            "max_tokens": max_tokens,
            "temperature": TEMPERATURE,
        },
    )
    try:
        body = response.json()
        usage = body["usage"]
        metrics = body["glassbox_metrics"]
        row = BenchRow(
            name=name,
            prompt_tokens=int(usage["prompt_tokens"]),
            completion_tokens=int(usage["completion_tokens"]),
            latency_ms=float(metrics["total_latency_ms"]),
            tokens_per_second=float(metrics["tokens_per_second"]),
            device=str(metrics["device"]),
            dtype=str(metrics["dtype"]),
            model_load_seconds=_optional_float(metrics["model_load_seconds"]),
            gpu_memory_allocated_mb=_optional_float(metrics["gpu_memory_allocated_mb"]),
            gpu_peak_memory_allocated_mb=_optional_float(
                metrics["gpu_peak_memory_allocated_mb"]
            ),
            gpu_memory_reserved_mb=_optional_float(metrics["gpu_memory_reserved_mb"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        print(f"Unexpected completion response: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    if row.device != device:
        print(
            f"Requested device {device} but the server ran on {row.device}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return row


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"expected a number or null, got {value!r}")
    return float(value)


def _request(
    client: httpx.Client,
    method: str,
    path: str,
    body: dict[str, str | int | float] | None = None,
) -> httpx.Response:
    try:
        response = client.request(method, path, json=body)
    except httpx.HTTPError as error:
        print(f"Request failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    if response.status_code != 200:
        print(
            f"Request failed: {response.status_code} {response.text}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return response


def _print_report(cpu_rows: list[BenchRow], cuda_rows: list[BenchRow]) -> None:
    print(
        f"Glassbox CPU vs CUDA  {BASE_URL}  model={MODEL}  max_tokens={MAX_TOKENS}"
    )
    print(
        f"warmup={WARMUP_MAX_TOKENS} token per device, excluded from the table"
    )
    print()

    table = [
        [
            "name",
            "prompt_tokens",
            "cpu_tokens",
            "cuda_tokens",
            "cpu_ms",
            "cuda_ms",
            "cpu_tok/s",
            "cuda_tok/s",
            "speedup",
        ]
    ]
    for cpu_row, cuda_row in zip(cpu_rows, cuda_rows, strict=True):
        if cpu_row.name != cuda_row.name:
            print(
                f"Prompt mismatch: {cpu_row.name} vs {cuda_row.name}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        speedup = cuda_row.tokens_per_second / max(cpu_row.tokens_per_second, 1e-9)
        table.append(
            [
                cpu_row.name,
                str(cpu_row.prompt_tokens),
                str(cpu_row.completion_tokens),
                str(cuda_row.completion_tokens),
                f"{cpu_row.latency_ms:.1f}",
                f"{cuda_row.latency_ms:.1f}",
                f"{cpu_row.tokens_per_second:.2f}",
                f"{cuda_row.tokens_per_second:.2f}",
                f"{speedup:.2f}x",
            ]
        )

    widths = [
        max(len(line[index]) for line in table) for index in range(len(table[0]))
    ]
    for line in table:
        print(
            "  ".join(
                cell.ljust(width) for cell, width in zip(line, widths, strict=True)
            )
        )

    print()
    print("speedup = cuda tokens/sec / cpu tokens/sec")
    print()
    _print_device_summary("cpu", cpu_rows[0], None)
    _print_device_summary("cuda", cuda_rows[0], cuda_rows)


def _print_device_summary(
    label: str,
    row: BenchRow,
    memory_rows: list[BenchRow] | None,
) -> None:
    if row.model_load_seconds is None:
        load_seconds = "none"
    else:
        load_seconds = f"{row.model_load_seconds:.2f}"
    summary = f"{label}  dtype={row.dtype}  model_load_seconds={load_seconds}"
    if memory_rows is None:
        print(summary)
        return

    allocated = _max_metric(memory_rows, "gpu_memory_allocated_mb")
    peak = _max_metric(memory_rows, "gpu_peak_memory_allocated_mb")
    reserved = _max_metric(memory_rows, "gpu_memory_reserved_mb")
    print(
        f"{summary}  gpu_allocated_mb={_format_mb(allocated)}  "
        f"gpu_peak_mb={_format_mb(peak)}  gpu_reserved_mb={_format_mb(reserved)}"
    )


def _max_metric(rows: list[BenchRow], name: str) -> float | None:
    values = [value for row in rows if (value := getattr(row, name)) is not None]
    if not values:
        return None
    return max(values)


def _format_mb(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value:.1f}"


if __name__ == "__main__":
    main()
