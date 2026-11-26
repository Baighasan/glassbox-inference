# Glassbox

Glassbox is a local LLM inference server that runs models on CPU or CUDA through an OpenAI-compatible completions API. It is a learning project that starts with a working server and progressively opens up each layer, from HTTP requests and token generation to model execution and device memory.

The first version delegated generation to Hugging Face `model.generate()`. Today, Glassbox owns the greedy decode loop, reuses attention keys and values across decoding steps, and uses PyTorch and Transformers for the model forward pass.

## How it works

- **Server** uses FastAPI to validate requests and return generated text, token usage, latency, throughput, and device metrics.
- **Engine** formats and tokenizes the prompt, runs a full prompt prefill, then sends one new token per decode step with the request's KV cache. It selects tokens with greedy argmax and stops at an end-of-sequence token or the requested limit.
- **Runtime** loads one model at a time, places weights and inputs on CPU or CUDA, and returns logits, the updated Transformers cache, and CUDA memory metrics. Inference currently uses FP32 on both devices.

## Quick start

Glassbox requires Python 3.12 or newer and uv. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv sync
uv run glassbox
```

Send a completion request to the local server at `http://127.0.0.1:8000`.

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"openai-community/gpt2","prompt":"The future of computing is","max_tokens":32}'
```

The request's `model` field selects the model. The first request loads its weights, and subsequent requests reuse them while that model is resident. Accepted models include GPT‑2, DistilGPT‑2, GPT‑2 Medium, GPT‑Neo 125M, Qwen2.5 0.5B Instruct, and TinyLlama 1.1B Chat. [Exact model IDs](src/glassbox/config.py)

- `GET /health` reports model load status, device, dtype, and model load time.
- `GET /v1/models` currently lists GPT‑2.
- `POST /v1/chat/completions` currently returns a placeholder response. Use `/v1/completions` for inference, including with instruction-tuned models.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `GLASSBOX_DEVICE` | `cpu` | Default execution device. Use `cuda` for GPU inference. |
| `GLASSBOX_MAX_TOKENS` | `64` | Default generation limit. Accepts 1–128. |
| `GLASSBOX_LOGGING_LEVEL` | `INFO` | Glassbox logging level. Accepts `INFO` or `DEBUG`. |

A completion request can override the defaults with `device` and `max_tokens`.

## KV-cache benchmark

The benchmark uses GPT‑2 in FP32 with batch size one across six workloads spanning 32–768 prompt tokens and 32–256 generated tokens. Suite throughput is total generated tokens divided by total prefill and decode time, starting with loaded weights and tokenized inputs.

| Device | Uncached tok/s | Cached tok/s | Speedup |
| --- | ---: | ---: | ---: |
| Intel Core i9‑9880H | 1.35 | 29.95 | **22.21×** |
| NVIDIA Quadro T2000 | 6.10 | 108.20 | **17.73×** |

The largest workload retained about 72 MiB of KV tensors and increased peak CUDA allocation by about 5 MiB. The CPU result uses one trial per mode with eight PyTorch threads. The CUDA result uses the median of three trials per mode for each workload. [Benchmark runner](scripts/bench_kv_cache.py)

```bash
uv run python scripts/bench_kv_cache.py --device cpu --cpu-threads 8 --repeats 1
uv run python scripts/bench_kv_cache.py --device cuda --repeats 3
```

## Where it's going

The MVP target is to serve `TinyLlama/TinyLlama-1.1B-Chat-v1.0` end to end on a 4 GiB Quadro T2000 with all weights on the GPU. The roadmap also covers model-backed chat completions, separate prefill and decode timings, streaming, batching, and lower-level execution. [Project plan](docs/project-plan.md)

## Development

```bash
uv run pytest
uv run pytest -m model
uv build
```

The model tests compare Glassbox token IDs with Hugging Face greedy generation on CPU and, when available, CUDA.
