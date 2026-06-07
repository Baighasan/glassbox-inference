# Glassbox

Glassbox is a local LLM inference stack built to learn ML inference infrastructure by progressively replacing black-box abstractions with explicit implementations.

The goal is to start with a working end-to-end model server, then peel back each layer of the stack: client, server, runtime, engine, backend, and hardware.

## Target MVP

Glassbox runs `TinyLlama/TinyLlama-1.1B-Chat-v1.0` locally on GPU through an OpenAI-compatible API using custom greedy decoding, while reporting latency, tokens/sec, token usage, dtype, device, and GPU memory metrics.

The first walking skeleton uses `gpt2` and Hugging Face `model.generate()` so the full system works before the internals are replaced.

## Setup

Glassbox requires Python 3.12 and
[`uv`](https://docs.astral.sh/uv/getting-started/installation/).

Install the project and its dependencies:

```bash
uv sync
```

Glassbox reads its runtime configuration from environment variables:

| Variable | Default | Allowed values |
| --- | --- | --- |
| `GLASSBOX_MODEL` | `gpt2` | Any non-empty model identifier |
| `GLASSBOX_DEVICE` | `cpu` | `cpu` or `cuda` |
| `GLASSBOX_MAX_TOKENS` | `128` | Integer from `1` through `128` |

For example:

```bash
export GLASSBOX_MODEL=gpt2
export GLASSBOX_DEVICE=cpu
export GLASSBOX_MAX_TOKENS=128
```

Start the local API server:

```bash
uv run glassbox
```

The server listens on `http://127.0.0.1:8000`.

Run the test suite:

```bash
uv run --with pytest pytest
```

Build the wheel and source distribution:

```bash
uv build
```

## Stack

```text
Client / curl / benchmark / harness
        |
        v
+---------------------------------------+
| Client                                |
|                                       |
| - sends completion/chat requests      |
| - receives generated text             |
| - later: benchmark and load harness   |
+-------------------+-------------------+
                    |
                    v
+---------------------------------------+
| Inference Server                      |
| FastAPI                               |
|                                       |
| - /health                             |
| - /v1/models                          |
| - /v1/completions                     |
| - /v1/chat/completions                |
| - OpenAI-compatible request handling  |
| - request validation                  |
| - response formatting                 |
| - later: streaming and queueing       |
+-------------------+-------------------+
                    |
                    v
+---------------------------------------+
| Inference Runtime                     |
|                                       |
| - model/tokenizer lifecycle           |
| - prompt formatting                   |
| - generation configuration            |
| - inference orchestration             |
| - latency, token, and memory metrics  |
| - later: prefill/decode split         |
| - later: KV cache visibility          |
+-------------------+-------------------+
                    |
                    v
+---------------------------------------+
| Inference Engine                      |
|                                       |
| v0: black-box HF generation path      |
| v1: model.forward() + greedy decoding |
| v2: explicit past_key_values usage    |
| later: custom execution pieces        |
+-------------------+-------------------+
                    |
                    v
+---------------------------------------+
| Backend                               |
|                                       |
| - v0: PyTorch execution under HF      |
| - later: explicit PyTorch operations  |
| - later: custom CUDA/Triton kernels   |
-------------------+-------------------+
                    |
                    v
+---------------------------------------+
| Hardware                              |
|                                       |
| - CPU first                           |
| - then CUDA GPU execution             |
| - target: NVIDIA Quadro T2000, 4 GB   |
+---------------------------------------+
```

## Abstraction-Breaking Plan

### 1. Client → Server: Build the API shell

Start with a working local FastAPI server that exposes OpenAI-compatible endpoints.

Goal: make Glassbox usable as an actual inference service before replacing internal model execution pieces.

### 2. Server → Runtime: Isolate inference orchestration

Move all model-specific logic out of the API routes and into a runtime layer.

The server should handle HTTP, validation, and response formatting. The runtime should handle prompt formatting, model/tokenizer lifecycle, generation configuration, and metrics.

Goal: keep serving concerns separate from inference execution concerns.

### 3. Runtime → Engine: Start with a black-box engine

Use Hugging Face `model.generate()` as the first engine implementation.

Goal: prove the full stack works end-to-end before breaking the model execution abstraction.

### 4. Runtime → Engine: Replace `model.generate()`

Replace Hugging Face generation with a custom greedy decoding loop using `model.forward(...)`.

Goal: understand token-by-token generation, logits, next-token selection, stopping conditions, and output decoding.

### 5. Engine → Backend → Hardware: Make device behavior explicit

Move from CPU execution to explicit CUDA execution with clear dtype and tensor-placement rules.

Goal: understand where the model lives, where tensors live, how fp32/fp16 affects memory, and how GPU memory pressure appears during inference.

### 6. Runtime → Engine: Add prefill/decode visibility

Measure prompt prefill separately from token-by-token decoding. Later, expose and use `past_key_values`.

Goal: understand why real inference systems separate prompt processing from incremental decoding.

### 7. Client → Server: Add serving features after single-request inference works

Add streaming, request cancellation, request queueing, and eventually batching.

Goal: move from a toy local server toward real inference-serving behavior without mixing serving logic into the runtime or engine.

### 8. Engine → Backend → Hardware: Defer lower-level execution work

After the Python stack is solid, explore quantization, custom PyTorch execution pieces, C++ extensions, CUDA/Triton kernels, or lower-level memory management.

Goal: only break into lower-level implementation once the higher-level stack boundaries are real and stable.


## MVP Milestones

1. Project skeleton with FastAPI, config, README, and minimal tests.
2. OpenAI-compatible API shell with `/health`, `/v1/models`, `/v1/completions`, and `/v1/chat/completions`.
3. `gpt2` walking skeleton on CPU using Hugging Face `model.generate()`.
4. Tiny benchmark script for a fixed prompt suite.
5. Custom greedy decoding on `gpt2` without `model.generate()`.
6. Explicit GPU path with CUDA, fp16, and GPU memory metrics.
7. TinyLlama final MVP on GPU without offloading.
8. Final writeup with demos, benchmarks, tradeoffs, and deferred roadmap.

## Deferred

These are intentionally out of scope for the first MVP:

- KV cache decoding.
- Streaming responses.
- Request batching.
- Request queueing.
- Concurrent request execution.
- Quantization.
- CPU/GPU offloading.
- Go control plane.
- C++ or CUDA backend code.
- Docker.
- Distributed inference runtime.

## More Detail

See [`docs/project-plan.md`](docs/project-plan.md) for the full project plan, design decisions, metrics, milestone definitions, and future roadmap.
