# Glassbox

Glassbox is a local LLM inference stack built to learn ML inference infrastructure by progressively replacing black-box abstractions with explicit implementations.

The goal is to start with a working end-to-end model server, then peel back each layer of the stack: client, server, engine, runtime, and hardware.

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

Glassbox reads its configuration from environment variables:

| Variable | Default | Allowed values |
| --- | --- | --- |
| `GLASSBOX_MODEL` | `openai-community/gpt2` | Any non-empty model identifier |
| `GLASSBOX_DEVICE` | `cpu` | `cpu` or `cuda` |
| `GLASSBOX_MAX_TOKENS` | `64` | Integer from `1` through `128` |

For example:

```bash
export GLASSBOX_MODEL=openai-community/gpt2
export GLASSBOX_DEVICE=cpu
export GLASSBOX_MAX_TOKENS=64
```

The server starts with no weights loaded. The `model` field on a completion selects the model. The first request for a supported model loads it, and later requests for that same model reuse the load. A different supported model replaces it. Only `openai-community/gpt2` is supported. `GLASSBOX_MODEL` does not select the loaded model. `GLASSBOX_DEVICE` is the device reported by `/health` and used when a completion omits `device`. `GLASSBOX_MAX_TOKENS` is used when a completion omits `max_tokens`. A request that sets either field overrides the environment. Request `max_tokens` must be from `1` through `128`.

Start the local API server:

```bash
uv run glassbox
```

The server listens on `http://127.0.0.1:8000`.

Run the test suite:

```bash
uv run --with pytest pytest
```

`pytest -m model` also runs the GPT-2 comparison, including the CUDA run when a GPU is available.

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
| Server                                |
| FastAPI                               |
|                                       |
| - /health                             |
| - /v1/models                          |
| - /v1/completions                     |
| - /v1/chat/completions                |
| - request validation                  |
| - response formatting                 |
| - later: streaming and a request queue|
+-------------------+-------------------+
                    |
                    v
+---------------------------------------+
| Engine                                |
|                                       |
| - tokenizer and prompt formatting     |
| - decode loop, sampling, and stop     |
| - token counts and request latency    |
| - later: scheduler and paged KV map   |
+-------------------+-------------------+
                    |
                    v
+---------------------------------------+
| Runtime                               |
|                                       |
| - load weights and device buffers     |
| - v0: model.generate() token ids      |
| - v1: forward(...) -> logits          |
| - later: KV cache and custom kernels  |
+-------------------+-------------------+
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

### 2. Server → Engine → Runtime: Walk the stack with a black box

The server calls the engine with text. The engine tokenizes, calls the runtime, and decodes. The runtime's first implementation is Hugging Face `model.generate()`.

Goal: prove the full stack works end-to-end before opening the generation loop.

### 3. Engine → Runtime: Replace `model.generate()`

The engine runs a greedy loop. The runtime's public method becomes `forward(...) -> logits` on the PyTorch module.

Goal: understand token-by-token generation, logits, next-token selection, stopping conditions, and output decoding.

### 4. Runtime → Hardware: Make device behavior explicit

Move from CPU execution to explicit CUDA execution with clear dtype and tensor-placement rules.

Goal: understand where the model lives, where tensors live, how fp32/fp16 affects memory, and how GPU memory pressure appears during inference.

### 5. Engine → Runtime: Add prefill/decode visibility

Measure prompt prefill separately from token-by-token decoding. Later, the runtime accepts `past_key_values` and the engine passes them through the loop.

Goal: understand why real inference systems separate prompt processing from incremental decoding.

### 6. Server: Add serving features after single-request inference works

Add streaming and a queue in front of the engine. Cancellation, continuous batching, and the logical KV block map belong to the engine once that queue exists.

Goal: move from a toy local server toward real inference-serving behavior while each feature stays in the layer that owns it.

### 7. Runtime: Defer lower-level execution work

After the Python stack is solid, explore quantization, custom PyTorch execution pieces, C++ extensions, CUDA/Triton kernels, or lower-level memory management.

Goal: only break into lower-level implementation once the server, engine, and runtime boundaries are real and stable.


## MVP Milestones

1. Project skeleton with FastAPI, config, README, and minimal tests.
2. OpenAI-compatible API shell with `/health`, `/v1/models`, `/v1/completions`, and `/v1/chat/completions`.
3. `gpt2` walking skeleton on CPU. The server calls the engine, and the runtime uses Hugging Face `model.generate()`.
4. Tiny benchmark script for a fixed prompt suite.
5. Custom greedy decoding on `gpt2`. The engine owns the loop, and the runtime exposes `forward(...) -> logits`.
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
- C++ or CUDA kernel code.
- Docker.
- Distributed execution.

## More Detail

See [`docs/project-plan.md`](docs/project-plan.md) for the full project plan, design decisions, metrics, milestone definitions, and future roadmap.
