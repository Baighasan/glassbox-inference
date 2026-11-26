# Quadro T2000 inference timings

## One projection for one new token

Measured on a TinyLlama layer with a 2,048 × 5,632 weight matrix. The compact-weight rows use custom test kernels that apply the same ideas as llama.cpp.

| Weight storage and math | Weight data read | Time per projection | Speed vs. cuBLAS FP32 |
| --- | ---: | ---: | ---: |
| Cast to FP32 at load, cuBLAS FP32 math | ~46 MB | 0.463 ms | 1× |
| FP32 storage, custom FP32 math | ~46 MB | 0.455 ms | ~1× |
| FP16 or BF16 storage, widen to FP32 in the kernel | ~23 MB | 0.229 ms | ~2× |
| Q8 storage, `dp4a` integer dot products | ~12 MB | 0.126 ms | ~3.7× |

The 16-bit path saves time by reading half as much weight data and widening each value as needed. The Q8 timing includes quantizing the input and uses approximated weights. These speedups apply to this projection. Whole-model speed also depends on attention and other work, and prompt processing handles multiple tokens at once.

## Glassbox generation with the KV cache disabled

Qwen2.5-0.5B-Instruct ships with BF16 weights. Each run generated 64 tokens, and the latency excludes model loading and weight conversion.

| Device and inference dtype | Generation time | Tokens/sec | Speed vs. CPU FP32 |
| --- | ---: | ---: | ---: |
| CPU FP32 | 10.75 s | 5.96 | 1× |
| CUDA FP32 | 8.18 s | 7.83 | ~1.31× |
| CUDA FP16 | 17.08 s | 3.75 | ~0.63× |

CUDA FP32 was ~2.1× as fast as CUDA FP16 in this run. Glassbox recomputes the growing sequence each step, so the time spent in the matrix-multiply kernels matters more here.

For the same GPT-2 small forward pass on CUDA, weights loaded directly in FP32 took 22.54 ms and weights rounded to FP16 then widened back to FP32 took 22.50 ms. Both ran in FP32, with overlapping timing ranges.
