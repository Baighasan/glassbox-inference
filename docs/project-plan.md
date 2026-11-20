# Glassbox Project Plan

Glassbox is a local LLM inference stack built to learn ML infrastructure by progressively replacing black-box abstractions with explicit implementations.

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
- C++ or CUDA kernel code.
- Docker.
- Distributed execution.

## Terminology

Glassbox uses this stack:

```text
Client -> Server -> Engine -> Runtime -> Hardware
```

The server accepts text and returns text. The engine owns the tokenizer and the decode loop. The runtime owns weights, device memory, and the forward pass. Scheduling, streaming, paged KV cache, and custom kernels stay behind those same owners and are implemented after the MVP.

`model.generate()` is a walking-skeleton shortcut. It fuses the engine loop and the runtime forward, so it lives in the runtime only until Milestone 5 splits them.

### Client

The caller that sends inference requests and consumes generated text.

Examples:

- `curl`.
- Benchmark harnesses.
- OpenAI-compatible SDKs or tools.

### Server

The network-facing layer. It accepts text and generation settings, calls the engine, and formats the response. It does not tokenize.

Responsibilities:

- Expose `/health`.
- Expose `/v1/models`.
- Expose `/v1/completions`.
- Expose `/v1/chat/completions`.
- Validate OpenAI-style request fields.
- Format OpenAI-compatible responses, including `usage` and `glassbox_metrics`.
- Read loaded model, device, dtype, and load time for `/health`.

Later:

- Streaming.
- A queue in front of the engine.

### Engine

The control loop for one request. It owns the tokenizer.

Responsibilities:

- Load and use the Hugging Face tokenizer.
- Format prompts.
- Call the runtime, choose the next token, and stop at `max_tokens` or EOS.
- Count `prompt_tokens` and `completion_tokens`.
- Measure `total_latency_ms` and `tokens_per_second`.

Prompt formatting:

- `/v1/completions`: raw prompt passes through.
- `/v1/chat/completions`: chat messages are joined into a single prompt until Milestone 7.
- TinyLlama: use the tokenizer chat template.

Milestone 3 implements the runtime call as one `model.generate()`. Milestone 5 replaces that call with a greedy loop over `forward(...) -> logits`.

Later, as engine components:

- Scheduler: queueing, continuous batching, cancellation, priorities, and concurrent requests.
- Logical KV block map for paged cache management.
- Sampling beyond greedy argmax, including temperature and top-p.
- Prefill versus decode timing, and time to first token.

### Runtime

Weights, device memory, and one forward pass.

```python
logits = forward(input_ids, attention_mask)
```

Responsibilities:

- Load one model at startup and fail startup if loading fails.
- Place weights and tensors on `cpu` or `cuda`.
- Use `float32` on CPU and `float16` on CUDA.
- Report `model_load_seconds`, `device`, `dtype`, `gpu_memory_allocated_mb`, and `gpu_memory_reserved_mb`.

Implementations:

- Milestone 3: `model.generate()` returns token ids. This hides the decode loop inside Hugging Face.
- Milestone 5: the public method is `forward(...) -> logits` on the PyTorch module.
- Later: the same method accepts `past_key_values` or KV block tables.
- Later: Triton, CUDA, or GGML can replace the PyTorch forward without changing the server or the engine loop.

The MVP is one process. A later split is a server process and a worker process that hosts the runtime.

## Architecture

```text
Client / curl / benchmark / opencode
        |
        v
Server          HTTP, validation, response JSON
        |
        v
Engine          tokenize, prompt format, decode loop, sampling, stop
        |
        v
Runtime         weights, device buffers, forward -> logits
        |
        v
Hardware        CPU, then CUDA
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
- Do not switch models after startup in the MVP.

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

Ownership:

- The engine counts tokens and measures request latency and tokens/sec.
- The runtime reports load time, device, dtype, and GPU memory.
- The server attaches `usage` and `glassbox_metrics` to the response.

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

> The server calls the engine, the engine calls the runtime, and `gpt2` on CPU returns only new text through `/v1/completions`.

Work:

- Add `server`, `engine`, and `runtime` packages matching the ownership above.
- Load `gpt2` and its tokenizer at startup.
- Engine formats the completion prompt and decodes new token ids.
- Runtime call is Hugging Face `model.generate()`.
- Return only new generated text.
- Add real usage stats and `glassbox_metrics`.
- Demo with `curl`.
- Leave `/v1/chat/completions` on the mock response until Milestone 7.

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

> The engine runs greedy decoding, and the runtime's public method is `forward(...) -> logits`.

Work:

- Remove the `model.generate()` path.
- Run the full sequence through `model.forward(...)` each step.
- Engine selects the next token with `argmax`.
- Append the token and repeat until `max_tokens` or EOS.
- Compare the loop against Hugging Face `generate()` on a fixed prompt.
- Document why full-sequence recomputation is inefficient.

This is intentionally Level B decoding:

```text
prompt tokens
-> runtime.forward(full sequence) -> logits
-> engine argmax
-> append token
-> runtime.forward(full sequence) again
-> repeat
```

KV cache decoding is deferred.

### Milestone 6: GPU Path

Outcome:

> The runtime can run `gpt2` on both CPU and CUDA, with explicit device config and GPU memory metrics.

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

- Runtime loads `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.
- Engine uses the tokenizer chat template for chat requests.
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

Post-MVP work, by owner:

Server:

- Add streaming responses.
- Add a queue in front of the engine.
- Explore Go for the server.

Engine:

- Pass `past_key_values` through the decode loop.
- Add prefill versus decode timing.
- Add time-to-first-token metrics.
- Add cancellation.
- Add continuous batching.
- Add a logical KV block allocator for paged cache management.

Runtime:

- Hold KV cache tensors, then accept block tables on `forward`.
- Explore quantization.
- Explore C++/CUDA/Triton kernels.
- Split a worker process out of the server process to host the runtime.
- Explore distributed execution.
