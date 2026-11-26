"""Benchmark GPT-2 greedy inference with and without a dynamic KV cache. (AI Generated)

Run with ``uv run python scripts/bench_kv_cache.py``. The timer covers prompt
prefill and fixed-length greedy decoding. Model loading, prompt tokenization,
and device warmup happen before timed trials.
"""

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer, Cache, PreTrainedModel

from glassbox.config import DEFAULT_MODEL, Device
from glassbox.runtime.runtime import Runtime


WORKLOADS = ((32, 32), (128, 64), (256, 128), (512, 128), (512, 256), (768, 256))
MIB = 1024**2
PROMPT = (
    "The observatory opens its doors before sunset so visitors can see how the "
    "telescopes are prepared. The staff checks the weather and chooses several "
    "objects to observe. On a clear night the group can see craters on the moon, "
    "the rings of Saturn, and distant clusters of stars. "
)


@dataclass(frozen=True)
class Trial:
    seconds: float
    peak_bytes: int | None
    cache_bytes: int
    generated_ids: tuple[int, ...]


@torch.inference_mode()
def _generate(
    model: PreTrainedModel,
    prompt_ids: torch.Tensor,
    output_tokens: int,
    use_cache: bool,
) -> tuple[torch.Tensor, Cache | None]:
    input_ids = prompt_ids
    attention_mask = torch.ones_like(prompt_ids)
    cache = None
    for _ in range(output_tokens):
        step_ids = input_ids if cache is None else input_ids[:, -1:]
        output = model(
            input_ids=step_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=use_cache,
        )
        cache = output.past_key_values
        next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del output
        input_ids = torch.cat((input_ids, next_id), dim=-1)
        attention_mask = torch.cat((attention_mask, torch.ones_like(next_id)), dim=-1)
    return input_ids[:, prompt_ids.shape[1] :], cache


def _reset_rss_peak() -> bool:
    # Linux clear_refs value 5 resets VmHWM to the process's current RSS.
    try:
        Path("/proc/self/clear_refs").write_text("5\n")
    except OSError:
        return False
    return True


def _rss_peak_bytes() -> int | None:
    try:
        lines = Path("/proc/self/status").read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) * 1024
    return None


def _measure(
    model: PreTrainedModel,
    prompt_ids: torch.Tensor,
    output_tokens: int,
    use_cache: bool,
) -> Trial:
    device = prompt_ids.device
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    rss_reset = not cuda and _reset_rss_peak()

    start = time.perf_counter()
    generated, cache = _generate(model, prompt_ids, output_tokens, use_cache)
    if cuda:
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start

    peak_bytes = torch.cuda.max_memory_allocated(device) if cuda else None
    if rss_reset:
        peak_bytes = _rss_peak_bytes()
    cache_bytes = 0
    if use_cache:
        # The last sampled token has yet to enter a forward pass.
        expected_cache_length = prompt_ids.shape[1] + output_tokens - 1
        if cache is None or cache.get_seq_length() != expected_cache_length:
            raise RuntimeError("Cached inference returned an unexpected cache length")
        cache_bytes = sum(
            tensor.numel() * tensor.element_size()
            for layer in cache.layers
            for tensor in (layer.keys, layer.values)
        )
    elif cache is not None:
        raise RuntimeError("Uncached inference returned a cache")
    return Trial(seconds, peak_bytes, cache_bytes, tuple(generated[0].tolist()))


def _format_mib(size: int | None) -> str:
    return "n/a" if size is None else f"{size / MIB:,.2f}"


def _bench_device(
    device: Device, workloads: tuple[tuple[int, int], ...], repeats: int
) -> None:
    runtime = Runtime()
    runtime.set_device(device)
    model = runtime.load(DEFAULT_MODEL).eval()
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    max_input = max(input_tokens for input_tokens, _ in workloads)
    prompt = tokenizer.encode(PROMPT, add_special_tokens=False)
    prompt = (prompt * ((max_input + len(prompt) - 1) // len(prompt)))[:max_input]
    all_ids = torch.tensor([prompt], device=device)
    for use_cache in (False, True):
        _generate(model, all_ids[:, :32], 4, use_cache)

    device_name = torch.cuda.get_device_name() if device == "cuda" else "CPU"
    memory_metric = "CUDA peak allocated" if device == "cuda" else "CPU peak RSS"
    print(f"\n{device_name} ({device}), {model.dtype}, {memory_metric}", flush=True)
    print(
        "| Input | Output | Uncached tok/s | Cached tok/s | Speedup | "
        "Uncached peak MiB | Cached peak MiB | Peak delta MiB | KV MiB |"
    )
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", flush=True)

    total_tokens = 0
    total_seconds = {False: 0.0, True: 0.0}
    for index, (input_tokens, output_tokens) in enumerate(workloads):
        print(
            f"Running {input_tokens} input / {output_tokens} output tokens on {device}",
            file=sys.stderr,
        )
        prompt_ids = all_ids[:, :input_tokens]
        trials: dict[bool, list[Trial]] = {False: [], True: []}
        for repeat in range(repeats):
            first_cached = (index + repeat) % 2 == 1
            for use_cache in (first_cached, not first_cached):
                trials[use_cache].append(_measure(model, prompt_ids, output_tokens, use_cache))

        uncached, cached = trials[False], trials[True]
        uncached_seconds = statistics.median(trial.seconds for trial in uncached)
        cached_seconds = statistics.median(trial.seconds for trial in cached)
        uncached_peak, cached_peak = (
            max((trial.peak_bytes for trial in mode if trial.peak_bytes is not None), default=None)
            for mode in (uncached, cached)
        )
        peak_delta = (
            cached_peak - uncached_peak
            if uncached_peak is not None and cached_peak is not None
            else None
        )
        print(
            f"| {input_tokens} | {output_tokens} | {output_tokens / uncached_seconds:.2f} | "
            f"{output_tokens / cached_seconds:.2f} | {uncached_seconds / cached_seconds:.2f}x | "
            f"{_format_mib(uncached_peak)} | {_format_mib(cached_peak)} | "
            f"{_format_mib(peak_delta)} | {_format_mib(cached[0].cache_bytes)} |",
            flush=True,
        )
        if any(trial.generated_ids != uncached[0].generated_ids for trial in uncached + cached):
            print(
                f"Greedy output differed across trials for {input_tokens}/{output_tokens}.",
                file=sys.stderr,
            )
        total_tokens += output_tokens
        total_seconds[False] += uncached_seconds
        total_seconds[True] += cached_seconds

    speedup = total_seconds[False] / total_seconds[True]
    print(
        f"Suite throughput ({total_tokens} generated tokens)\n"
        f"Uncached {total_tokens / total_seconds[False]:.2f} tok/s "
        f"({total_seconds[False]:.3f} s), cached {total_tokens / total_seconds[True]:.2f} tok/s "
        f"({total_seconds[True]:.3f} s), {speedup:.2f}x ({(speedup - 1) * 100:+.1f}%)",
        flush=True,
    )


def main() -> None:
    """Run the fixed batch-one workload suite on the selected devices."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--case", type=int, nargs=2, metavar=("INPUT", "OUTPUT"))
    args = parser.parse_args()
    if args.repeats < 1 or args.cpu_threads < 1:
        parser.error("--repeats and --cpu-threads must be positive")
    if args.device != "cpu" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Use --device cpu for a CPU-only run.")
    workloads = WORKLOADS
    if args.case:
        input_tokens, output_tokens = args.case
        if min(input_tokens, output_tokens) < 1 or input_tokens + output_tokens > 1024:
            parser.error("--case requires positive lengths totaling at most 1024 tokens")
        workloads = ((input_tokens, output_tokens),)

    torch.set_num_threads(args.cpu_threads)
    print(
        f"{DEFAULT_MODEL}, PyTorch {torch.__version__}, Transformers {transformers.__version__}, "
        f"{args.cpu_threads} CPU threads, {args.repeats} trial(s) per mode",
        flush=True,
    )
    if args.device in ("cpu", "both"):
        _bench_device("cpu", workloads, args.repeats)
    if args.device in ("cuda", "both"):
        _bench_device("cuda", workloads, args.repeats)


if __name__ == "__main__":
    main()
