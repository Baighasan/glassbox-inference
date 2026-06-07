# Glassbox Project Plan

Glassbox is a local LLM inference engine project built to learn ML infrastructure by progressively replacing black-box abstractions with explicit implementations.

One-sentence goal:

> Glassbox runs TinyLlama locally on GPU through an OpenAI-compatible API with custom greedy decoding, reporting latency, tokens/sec, and memory metrics while progressively exposing the inference stack.

## Goals

- Learn ML inference infrastructure internals.
- Build a resume-worthy project with clear technical tradeoffs.
- Produce a useful local model server as a secondary outcome.
- Start with high-level abstractions, then break them one by one.
- Keep each milestone demonstrable and end-to-end.

## Non-Goals For MVP

- Streaming responses.
- Request batching.
- Request queueing.
- Concurrent request execution.
- KV cache implementation.
- Quantization.
- CPU/GPU offloading.
- Go control plane.
- C++ or CUDA backend code.
- Docker.
- Distributed inference runtime.

## Terminology

Glassbox uses this high-level stack:

```text
Client -> Server -> Runtime -> Engine -> Backend -> Hardware
```

The boundaries are intentionally broad during the MVP. More detailed ownership, such as scheduling, streaming, KV-cache management, and lower-level execution pieces, can be refined when those features are implemented.

### Client

The caller that sends inference requests and consumes generated text.

Examples:

- `curl`.
- Benchmark harnesses.
- OpenAI-compatible SDKs or tools.

### Inference Server

The network-facing layer of the system. It accepts requests from clients, validates them, invokes the runtime, and returns responses.

Responsibilities:

- Expose `/health`.
- Expose `/v1/models`.
- Expose `/v1/completions`.
- Expose `/v1/chat/completions`.
- Validate OpenAI-style request fields.
- Format OpenAI-compatible responses.
- Handle protocol-level serving concerns as they are added later.

### Inference Runtime

The orchestration layer for inference. It manages resources, prepares requests for execution, invokes the engine, and collects request-level results.

Responsibilities:

- Manage model and tokenizer lifecycle.
- Format prompts.
- Configure generation.
- Coordinate execution through the engine.
- Collect latency, token, and memory metrics.

### Inference Engine

The model execution layer. It implements generation behavior and turns prepared inputs into generated token IDs.

Initial implementation:

- Use Hugging Face `model.generate()` as a black-box engine.

Later implementation:

- Replace `model.generate()` with explicit `model.forward(...)` calls.
- Implement token-by-token greedy decoding.
- Expose more of prefill, decode, logits processing, and KV-cache behavior.

### Backend

The computational implementation used by the engine.

Initial backend:

- Transformers model code running on PyTorch.

Later backends or backend pieces:

- More explicit PyTorch execution.
- Custom C++ extensions.
- Custom CUDA or Triton kernels.

### Prompt Formatter

The component that converts external prompt formats into model-ready text. This is part of the runtime in the MVP, but may become a distinct module as the project grows.

Examples:

- `/v1/completions`: raw prompt passes through.
- `/v1/chat/completions`: chat messages are converted into a single prompt.
- TinyLlama: use tokenizer chat template when available.

### Scheduler

A future runtime component that decides which requests run and when.

Deferred responsibilities:

- Queuing.
- Batching.
- Cancellation.
- Priorities.
- Concurrent request handling.

### Model Worker

A future separate process that owns model memory and executes inference requests.

The MVP starts as a single process. A later version may split into an API server and a model worker, potentially with a Go control plane and Python model worker.

## Architecture

```text
Client / curl / benchmark / opencode
        |
        v
FastAPI Inference Server
        |
        v
OpenAI Request Validation
        |
        v
Prompt Formatter
        |
        v
Glassbox Inference Runtime
        |
        v
Glassbox Inference Engine
        |
        v
Transformers + PyTorch Backend
        |
        v
CPU / CUDA Hardware
```

## Target Hardware

- OS: Ubuntu server.
- CPU: Intel Core i9-9880H @ 2.30 GHz.
- RAM: approximately 32 GB.
- GPU: NVIDIA Quadro T2000 with 4 GB VRAM.

## API Scope

MVP endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`

The API should mimic OpenAI closely enough for standard harnesses and clients.

Validation behavior:

- Reject `stream=true`.
- Reject `temperature != 0`.
- Reject `n > 1`.
- Reject tools/function calling fields.
- Ignore harmless unsupported fields only when clearly documented.
- Default `max_tokens` to `64`.
- Cap `max_tokens` at `128`.

Response behavior:

- Return only newly generated text, not the original prompt.
- Include OpenAI-style `usage`.
- Include custom `glassbox_metrics` during development.

## Configuration

Use environment variables for MVP configuration.

Example:

```text
GLASSBOX_MODEL=gpt2
GLASSBOX_DEVICE=cpu
GLASSBOX_MAX_TOKENS=128
```

Rules:

- Load one model per server process.
- Load the model at server startup.
- Fail startup if model loading fails.
- Require explicit device selection: `cpu` or `cuda`.
- Use `float32` on CPU.
- Use `float16` on CUDA.
- Do not support runtime model switching in MVP.

## Metrics

MVP metrics:

- `model_load_seconds`
- `prompt_tokens`
- `completion_tokens`
- `total_latency_ms`
- `tokens_per_second`
- `device`
- `dtype`
- `gpu_memory_allocated_mb`
- `gpu_memory_reserved_mb`

`/health` should include:

- loaded status
- configured model
- device
- dtype
- model load time

## Milestones

Progress:

- [x] Milestone 1: Project Skeleton
- [x] Milestone 2: OpenAI API Shell
- [ ] Milestone 3: GPT-2 Walking Skeleton
- [ ] Milestone 4: Benchmark Script
- [ ] Milestone 5: Break `model.generate()` On GPT-2
- [ ] Milestone 6: GPU Path
- [ ] Milestone 7: TinyLlama Final MVP
- [ ] Milestone 8: Final MVP Writeup

### Milestone 1: Project Skeleton - Complete

Outcome:

> A Python project structure exists with FastAPI, config loading, a README, and minimal tests scaffold.

Work:

- [x] Create Python package layout.
- [x] Create FastAPI app entry point.
- [x] Create engine/server module boundaries.
- [x] Add environment-based config.
- [x] Add README setup section.
- [x] Add minimal tests scaffold.

### Milestone 2: OpenAI API Shell - Complete

Outcome:

> The server exposes OpenAI-style endpoints with validation and mock responses.

Work:

- [x] Implement `GET /health`.
- [x] Implement `GET /v1/models`.
- [x] Implement `POST /v1/completions`.
- [x] Implement `POST /v1/chat/completions`.
- [x] Add strict validation for unsupported dangerous fields.
- [x] Add OpenAI-style response formatting.

### Milestone 3: GPT-2 Walking Skeleton

Outcome:

> Glassbox serves `gpt2` on CPU using Hugging Face `model.generate()` and returns generated text through OpenAI-compatible endpoints.

Work:

- Load `gpt2` at startup.
- Use Hugging Face tokenizer.
- Use Hugging Face `model.generate()`.
- Return only new generated text.
- Add usage stats.
- Add `glassbox_metrics`.
- Demo with `curl`.

### Milestone 4: Benchmark Script

Outcome:

> A tiny benchmark script can measure request latency and tokens/sec for a small prompt suite.

Prompt suite:

- Short prompt.
- Medium prompt.
- Chat-style prompt.

Work:

- Add `scripts/bench.py`.
- Print results to stdout.
- Do not persist benchmark results yet.

### Milestone 5: Break `model.generate()` On GPT-2

Outcome:

> Glassbox implements custom greedy decoding for `gpt2` using `model.forward(...)` and full-sequence recomputation.

Work:

- Add a generation mode that does not call `model.generate()`.
- Run the full sequence through `model.forward(...)` each step.
- Select the next token with `argmax`.
- Append the token and repeat until `max_tokens` or EOS.
- Compare custom decode against Hugging Face `generate()`.
- Document why this approach is inefficient.

This is intentionally Level B decoding:

```text
prompt tokens
-> forward full sequence
-> pick next token
-> append token
-> forward full sequence again
-> repeat
```

KV cache decoding is deferred.

### Milestone 6: GPU Path

Outcome:

> Glassbox can run `gpt2` on both CPU and CUDA, with explicit device config and GPU memory metrics.

Work:

- Support `GLASSBOX_DEVICE=cpu|cuda`.
- Move model and tensors to the selected device.
- Use `float32` on CPU.
- Use `float16` on CUDA.
- Measure GPU memory allocated and reserved.
- Benchmark `gpt2` CPU vs GPU.

### Milestone 7: TinyLlama Final MVP

Outcome:

> Glassbox runs `TinyLlama/TinyLlama-1.1B-Chat-v1.0` on GPU through an OpenAI-compatible API using custom greedy decoding.

Work:

- Load `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
- Use tokenizer chat template for chat requests.
- Run CPU correctness checks.
- Run TinyLlama on CUDA without offloading.
- Preserve custom greedy decoding.
- Record latency, tokens/sec, and memory metrics.
- Benchmark TinyLlama with the fixed prompt suite.

Strict final MVP rule:

> TinyLlama must run on GPU for the final MVP to be considered complete.

If TinyLlama does not fit in 4 GB VRAM without offloading, the final MVP is blocked until the GPU path is solved or the strict requirement is revisited.

### Milestone 8: Final MVP Writeup

Outcome:

> The project has a clear setup guide, demos, benchmark results, and a learning-oriented explanation of the abstraction layers.

Work:

- Add `curl` demo.
- Add benchmark demo.
- Attempt opencode compatibility as a stretch goal.
- Explain each abstraction layer.
- Explain what was initially black-boxed.
- Explain what was replaced.
- Include benchmark table.
- Include deferred roadmap.

## Final MVP Definition

The final MVP is complete when:

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` runs on GPU.
- The server exposes `/health`, `/v1/models`, `/v1/completions`, and `/v1/chat/completions`.
- The chat endpoint uses an OpenAI-compatible request/response shape.
- Generation uses custom greedy decoding instead of `model.generate()`.
- Responses return only newly generated text.
- Metrics include latency, tokens/sec, usage, device, dtype, and GPU memory.
- A benchmark script can run a small fixed prompt suite.
- The README explains setup, milestones, and what was learned.

## Estimated Effort

- Walking skeleton: 1 weekend.
- Custom GPT-2 greedy decode: 1 weekend.
- GPU path and benchmarks: 1 weekend.
- TinyLlama final MVP and writeup: 1 weekend.

Realistic total: 3-4 weekends, with CUDA setup and TinyLlama VRAM fit as the main risks.

## Future Roadmap

Post-MVP branches:

- Implement KV cache decoding with `past_key_values`.
- Add prefill vs decode timing.
- Add time-to-first-token metrics.
- Add streaming responses.
- Add request queueing.
- Add cancellation.
- Add batching.
- Explore quantization.
- Split API server from model worker.
- Explore Go for API/control-plane infrastructure.
- Explore C++/CUDA for lower-level backend pieces.
- Study paged/slotted KV cache management.
- Explore distributed inference runtime concepts.
