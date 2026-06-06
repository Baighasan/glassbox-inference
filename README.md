# Glassbox

Glassbox is a local LLM inference engine built to learn ML inference infrastructure by progressively replacing black-box abstractions with explicit implementations.

The goal is to start with a working end-to-end model server, then peel back each layer: generation, decoding, device placement, model execution, memory behavior, and eventually serving/runtime concerns.

## Target MVP

Glassbox runs `TinyLlama/TinyLlama-1.1B-Chat-v1.0` locally on GPU through an OpenAI-compatible API using custom greedy decoding, while reporting latency, tokens/sec, token usage, dtype, device, and GPU memory metrics.

The first walking skeleton uses `gpt2` and Hugging Face `model.generate()` so the full system works before the internals are replaced.

## Stack

```text
Client / curl / benchmark / harness
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
+-------------------+-------------------+
                    |
                    v
+---------------------------------------+
| Prompt Layer                          |
|                                       |
| - raw prompt passthrough              |
| - chat messages -> model prompt       |
| - TinyLlama chat template             |
+-------------------+-------------------+
                    |
                    v
+---------------------------------------+
| Inference Engine                      |
|                                       |
| - generation config                   |
| - tokenizer wrapper                   |
| - model runner                        |
| - metrics collection                  |
+-------------------+-------------------+
                    |
                    v
+---------------------------------------+
| Model Runner                          |
|                                       |
| v0: Hugging Face model.generate()     |
| v1: custom greedy decode loop         |
| v2: KV cache and prefill/decode split |
+-------------------+-------------------+
                    |
                    v
+---------------------------------------+
| Runtime                               |
|                                       |
| v0: PyTorch + Transformers            |
| later: lower-level runtime pieces     |
+-------------------+-------------------+
                    |
                    v
+---------------------------------------+
| Hardware                              |
|                                       |
| CPU first                             |
| then NVIDIA Quadro T2000, 4 GB VRAM   |
+---------------------------------------+
```

## Abstraction-Breaking Plan

### 1. End-To-End Black Box

Use Hugging Face tokenizer, Hugging Face model loading, and `model.generate()` to get a complete local server working.

Goal: prove the wires are connected before optimizing or replacing internals.

### 2. Break Generation

Replace `model.generate()` with a custom greedy decoding loop using `model.forward(...)`.

Goal: understand how generated tokens are selected one step at a time.

### 3. Break Device Placement

Move from CPU execution to explicit CUDA execution with clear dtype rules.

Goal: understand model placement, tensor placement, fp32 vs fp16, and GPU memory pressure.

### 4. Break Model Realism

Move from `gpt2` to `TinyLlama/TinyLlama-1.1B-Chat-v1.0`.

Goal: work with a more realistic chat model, tokenizer behavior, and chat template.

### 5. Break Runtime Visibility

Add prefill/decode timing and eventually implement KV cache support with `past_key_values`.

Goal: understand why real inference engines separate prompt processing from token-by-token decoding.

### 6. Break Serving Simplicity

Add streaming, request queueing, cancellation, and batching after the single-request MVP works.

Goal: move from a toy local server toward real inference-serving concerns.

### 7. Break Process Boundaries

Split the API server from the model worker, potentially using a Go control plane and Python model worker.

Goal: learn how production inference systems separate routing/control-plane concerns from model execution.

### 8. Break Runtime Implementation

Explore quantization, C++ runtime pieces, CUDA kernels, and KV cache memory management later.

Goal: move from using an ML framework to understanding lower-level inference runtime design.

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
- C++ or CUDA runtime code.
- Docker.
- Distributed inference runtime.

## More Detail

See [`docs/project-plan.md`](docs/project-plan.md) for the full project plan, design decisions, metrics, milestone definitions, and future roadmap.
