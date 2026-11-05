# M4 Max kernel research, 2026-10-22

## Findings

The upstream maintainer states that M4 Max is excluded because the team has not thoroughly tested it, not because the hardware is incapable: [Recursive issue #1, maintainer reply](https://github.com/recursive-oss/rec/issues/1#issuecomment-6322920897). Searches of upstream issues and PRs found no M4 Max kernel patch. The adjacent [M3 Max PR #7](https://github.com/recursive-oss/rec/pull/7) is open and reuses the M4 Pro 36/32/32 profile with specialization disabled. Its [issue #6](https://github.com/recursive-oss/rec/issues/6) reports 166.5 decode tok/s at 36/32/32, falling to 96.6 at 76/64/64. This is an external contributor report with insufficient benchmark detail for an apples-to-apples comparison; it is evidence against blindly multiplying workers by GPU core count.

Read-only local observations: system_profiler reports Apple M4 Max with 40 GPU cores and 128 GB memory. MLX 0.32.0 in `.build/m4-profile-venv` reports `architecture=applegpu_g16s`, `memory_size=137438953472`, `max_recommended_working_set_size=115448725504`, `max_buffer_length=86586540032`, `resource_limit=499000`. No GPU kernels or benchmarks were run by this research agent. The architecture identity is exactly the M4 Pro identity accepted in `device.py`. The additional decoder architecture assertion in `megakernel.py:991` already permits it. MLX obtains this architecture from Metal unless MLX_METAL_GPU_ARCH overrides it: [pinned MLX device source](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/device.cpp#L585).

## Hardware expectations

| GPU | Full GPU cores | Memory bandwidth | Relevant generation |
|---|---:|---:|---|
| M4 Pro | 20 | 273 GB/s | Apple9 / g16s |
| This M4 Max | 40 | 546 GB/s | Apple9 / g16s, measured |
| M5 Pro | 20 | 307 GB/s | Apple10 / g17s |
| M5 Max | 40 | 614 GB/s | Apple10 / g17s |

Sources: [Apple M4 announcement](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/), [Apple M5 announcement](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/), [Apple GPU family tables](https://developer.apple.com/metal/limits/). The 32-core M4 Max SKU is different (410 GB/s), and should not silently inherit conclusions from this machine.

Inference, not measurement: for a workload limited entirely by DRAM traffic with equal traffic and bandwidth efficiency, M4 Max's ratio is 2.00x M4 Pro and 0.889x M5 Max. This is a conditional comparison, not an attainable-throughput prediction. Resident kernel synchronization, cache locality, expert dispatch, and CPU overhead can dominate. M5 adds GPU neural accelerators and has different occupancy management, so prefill parity is not a reasonable blanket target. [Apple's MLX article](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) illustrates the distinction: its M5/M4 tests obtain 3.33–4.06x TTFT improvement but only 1.19–1.27x generation speed improvement. These are other models on base chips, not Flint results.

Apple also documents doubled FP16/complex ALU execution, lower register-access latency, and second-generation dynamic caching in M5: [M5 GPU architecture talk](https://developer.apple.com/videos/play/tech-talks/111431/). Those differences can matter even without NAX, but they do not imply every FP32/BF16 rec instruction becomes twice as fast.

## Appropriate kernel paths

1. Adapt rec's existing g16s resident decode kernel first. It is already a genuine M4 implementation with global device synchronization, hand-tuned low-bit unpack/dot routines, attention and expert teams, BF16 rounding semantics, and reduced host dispatch. An M4 Max profile with tuned worker allocation retains those optimizations; it is not inherently a slow generic fallback. The current manifest-pinned Flint checkpoint has `mlx_export.routed_expert_bits=4`. The specialization flag only changes generated code inside the `bits == 3` branch in `mixed_bits.py`, so toggling it is a no-op for this Q4 checkpoint. Keep it disabled as a conservative default for future Q3 experiments, where the reported M4 Pro parity failure is relevant.
2. Treat mixed-expert specialization as a separate Q3 correctness investigation, not a Q4 performance knob. `mixed_bits.py:76–109` replaces runtime shared/routed dispatch with template specializations. Benchmarking `True` without scalar/packet/logit parity is not a safe performance result. Establish a minimized failure before changing arithmetic or compiler options; BF16 casts in this file are model semantics.
3. The pinned MLX already supplies appropriate non-NAX quantized GEMM paths, including split-K and gathered experts. [MLX 0.32.0 quantized dispatch](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/quantized.cpp#L771) selects qmm_nax only when supported; otherwise it uses ordinary tiled qmm kernels. [NAX gate](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/device.cpp#L943) requires macOS 26.2+ and GPU generation >=17 for Mac parts. M4's gen16 cannot acquire M5 hardware by an architecture override. MLX's existing kernels are suitable reference/baseline candidates and prefill building blocks, not drop-in replacements for a whole resident interpreter.
4. Tune prefill independently (e.g. 512/1024/2048/4096 chunk candidates at matched prompt lengths). M5-specific matrix acceleration does not transfer to M4; use the already supported simdgroup matrix and non-NAX quantized paths. Do not assume a larger chunk wins on M4 Max solely because its memory is larger.

## Residency and measurement recommendations

Apple publishes per-threadgroup limits (Apple9/Apple10: 1024 threads, 32 KiB threadgroup memory), not a reliable fixed count of concurrently resident threadgroups for this shader. Those API limits are not rec worker-count limits. Registers, threadgroup memory, live shader state, and scheduling affect actual occupancy. `megakernel.py:137–158` spins at a global barrier across all workers and has a bounded diagnostic timeout, so exceeding a viable occupancy range can cause functional failures or severe stalls.

The recommended working-set size from MLX (~107.52 GiB here) describes memory budgeting, not how many compute threadgroups can simultaneously reside. Likewise, Metal residency sets refer to resource availability, not worker occupancy.

The base megakernel contains per-phase spin/arrival instrumentation, but the optimized path explicitly requires profiling disabled (`kv_teams.py` checks `if(0)` in synchronization and requires schedule `profile=False`). It cannot simply be enabled for the optimized resident kernel. Adding supported instrumentation would be a separate implementation requiring parity checks, and instrumented performance must be kept separate from production timing. Existing synchronization diagnostics still detect worker failures. Sweep total workers independently from MLP/head workers subject to source invariants, using isolated subprocesses with watchdogs. Maintain complete logs for failed/deadlocked/parity-failed configurations. Verify both short and long contexts because KV attention changes the balance.

For deeper adaptation, measure register spills, memory stalls and occupancy in Xcode GPU capture. Apple's [compiler-occupancy guidance](https://developer.apple.com/videos/play/wwdc2023/10127/) recommends tuning maxTotalThreadsPerThreadgroup per kernel and device and explicitly notes that the maximum setting need not be optimal. Applying this may require exposing a native Metal pipeline/compile option; merely changing rec's dispatch count is a separate operation. Use these measurements to decide whether to split a phase, reduce per-thread temporary state, change tile/unroll sizes, or preserve the fused kernel.

Acceptance should require deterministic correctness/parity and diagnostic completion, then matched production decode measurements (same Flint artifact/hash, context lengths, output count, cache state, MLX version, thermal/power conditions). Report first-token/prefill separately from decode. Compare against M4-Pro-style profile running on this same M4 Max before comparing claims from different devices. To claim proximity to M5 Max requires actual matched M5 Max measurements, not bandwidth arithmetic.

## Published chart extraction and benchmark audit

The SVG explicitly states endpoint ranges and methodology: rec 131.2–22.6 tok/s, llama.cpp 88.0–8.6 tok/s; 48 GiB, 4-bit weights, five repeats, 1024 output steps, prefill excluded. README calls these 1024 output tokens. The source contains no explicit intermediate data labels. Inverting SVG's linear y mapping, `tok_s = (172.8 - y) / (172.8 - 45.051429) * 120`, recovers:

| Input context label | rec tok/s | llama.cpp tok/s |
|---|---:|---:|
| 1K | 131.2 | 88.0 |
| 2K | 128.3 | 85.9 |
| 4K | 123.0 | 83.7 |
| 8K | 113.7 | 79.5 |
| 16K | 99.0 | 71.4 |
| 32K | 82.4 | 58.9 |
| 64K | 61.7 | 42.9 |
| 128K | 39.1 | 24.2 |
| 252K | 22.6 | 8.6 |

These intermediate numbers are reconstructed from vector coordinates, not independently published raw measurements. [Pinned chart source](https://github.com/recursive-oss/rec/blob/e53663b262af4cb1ca498e596ca8b1a2d79e0800/.github/assets/throughput.svg). Local tracked-file search, GitHub code searches for benchmark/throughput, and upstream issue searches did not locate the chart benchmark implementation. Sampling settings, prompt construction, exact initial-versus-decode token convention, mean-versus-median statistic, power mode and GPU-core SKU remain undocumented. Production's defaults (temperature=1, top_p=.95, top_k=20) do not establish the benchmark's settings.

The existing `kernels/sampling.py:generate()` function uses unfiltered temperature-one sampling, takes maximum as total generated outputs, and reports `(len(tokens)-1)/decode_seconds`, excluding the initial sampled prefill token. This is a relevant convention, but there is no evidence connecting it to the chart.

The experiment harness preallocates both global KV banks and sliding caches before timing, including 65536-token bank boundaries. It records separate prefill duration, requested and actual decode counts, EOS completion, the selected sampling route, MLX environment settings, host power/thermal state, and source hashes. Detailed logit/layer/KV parity is greedy; other modes compare generated token hashes during timed runs. It uses the decode entrypoints with process-local candidate profiles and optional experimental transforms, synchronizes timing, rotates variants, and keeps production profiles unchanged. Timing excludes the first prefill-sampled token and measures --steps subsequent decode transitions, so a complete run generates steps+1 total outputs. A primitive probe pass is not full-model qualification. The optional upstream exhaustive validator broadens functional coverage but does not establish every deployment scenario.


## Full-model evidence update — 2026-10-22

The initial adaptation recommendations above have now been tested on the complete pinned Q4 checkpoint. Stock automatic QKV with a 64/64/64 worker profile passed the full-model harness and upstream exhaustive validator. Default filtered packet medians across five repetitions of 1024 timed decode transitions were **168.064 tok/s at 1K input, 162.175 at 8K, and 109.073 at 32K**. The initial prefill-sampled token is excluded from both timing and count. These are measurements of this M4 Max, not M5 comparisons or a guarantee of 160 tok/s at every context.

Static plausibility did not ensure correctness: forcing shared QKV at 48/48/48 failed first-token packet parity and remains unqualified. Shared76 passed short checks but was slower than stock76 in that screen (183.50 versus 192.05 tok/s). The detailed [kernel audit](kernel-audit.md) retains the historical reasoning with a dated outcome addendum.

The paired extended 36/32/32 control completed at 163.01 / 148.08 / 93.67 tok/s for 1K / 8K / 32K; all checks passed. The separate grammar-enabled Engine probe measured 175.34 tok/s on its shorter 81-token prompt and 512-token output workload. No matched M5 Max measurement exists, and the published M4 Pro chart's unknown protocol prevents an apples-to-apples device speedup claim. The 200.44 tok/s short64 result and 168.06 tok/s extended64 result also show timing variation whose cause was not isolated; [measurement notes](measurement-notes.md) document the observed variation without assigning a thermal or sequence-length cause. See [results.md](results.md) for final paired measurements and recommendation.
