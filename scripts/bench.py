"""Run a fixed prompt suite against a local Glassbox server."""

import os
import sys
from dataclasses import dataclass

import httpx

BASE_URL = os.environ.get("GLASSBOX_URL", "http://127.0.0.1:8000").rstrip("/")
MODEL = "openai-community/gpt2"
MAX_TOKENS = 32
TEMPERATURE = 0
REQUEST_TIMEOUT_SECONDS = 600.0

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


def main() -> None:
    """Check the server, run the prompt suite, and print metrics."""
    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)
    with httpx.Client(base_url=BASE_URL, timeout=timeout) as client:
        _check_health(client)
        rows = [_complete(client, name, prompt) for name, prompt in PROMPTS]
    _print_report(rows)


def _check_health(client: httpx.Client) -> None:
    _request(client, "GET", "/health")


def _complete(client: httpx.Client, name: str, prompt: str) -> BenchRow:
    response = _request(
        client,
        "POST",
        "/v1/completions",
        body={
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
        },
    )
    try:
        body = response.json()
        usage = body["usage"]
        metrics = body["glassbox_metrics"]
        return BenchRow(
            name=name,
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
            latency_ms=metrics["total_latency_ms"],
            tokens_per_second=metrics["tokens_per_second"],
            device=metrics["device"],
            dtype=metrics["dtype"],
            model_load_seconds=metrics["model_load_seconds"],
        )
    except (KeyError, TypeError, ValueError) as error:
        print(f"Unexpected completion response: {error}", file=sys.stderr)
        raise SystemExit(1) from error


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


def _print_report(rows: list[BenchRow]) -> None:
    print(f"Glassbox benchmark  {BASE_URL}  model={MODEL}  max_tokens={MAX_TOKENS}")
    print()
    table = [
        [
            "name",
            "prompt_tokens",
            "completion_tokens",
            "latency_ms",
            "tokens_per_sec",
        ],
        *[
            [
                row.name,
                str(row.prompt_tokens),
                str(row.completion_tokens),
                f"{row.latency_ms:.1f}",
                f"{row.tokens_per_second:.2f}",
            ]
            for row in rows
        ],
    ]
    widths = [
        max(len(line[index]) for line in table) for index in range(len(table[0]))
    ]
    for line in table:
        print(
            "  ".join(
                cell.ljust(width) for cell, width in zip(line, widths, strict=True)
            )
        )

    first = rows[0]
    if first.model_load_seconds is None:
        load_seconds = "none"
    else:
        load_seconds = f"{first.model_load_seconds:.2f}"
    print()
    print(
        f"device={first.device}  dtype={first.dtype}  "
        f"model_load_seconds={load_seconds}"
    )


if __name__ == "__main__":
    main()
