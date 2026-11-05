# M4 Max qualification results — 2026-10-22

The existing optimized resident kernels work on this **40-core M4 Max / 128 GiB**. A **64 resident / 64 MLP / 64 vocabulary-head worker** profile is a viable candidate: sustained packet decode measured **168 tok/s at 1K input, 162 at 8K, and 109 at 32K**, with exact parity checks passing. No generic decode fallback was used. The user considers roughly 160 tok/s sufficient, so this campaign stops at a validated candidate rather than chasing the short screening peak.

The profiling campaign kept production hardware guards and kernels unchanged, injecting each experimental profile into a fresh process. The validated profile has since been integrated at the user's request; see [local build](local-build.md). The production engine verifies the 40-core SKU and rejects the untested 32-core variant. These results establish the tested workload and device, not blanket support for all sampling modes or the full 262K context limit.

## Why the repository rejects this Mac

Both `cli/engine.go` and `engine/src/rec_mlx/kernels/device.py` explicitly allow only M4 Pro, M5 Pro and M5 Max. The maintainer describes M4 Max as insufficiently tested; the hardware is not missing the instructions needed by the repository's resident decoder. MLX reports `applegpu_g16s`, matching the supported M4 Pro architecture. See [research and primary sources](research.md).

The useful adaptation is the same kind of scheduling change used for the supported chips: resize the resident, expert and output-head teams while retaining fused quantized math, attention teams, device synchronization and packet sampling. The selected 64/64/64 profile uses existing production transforms, automatic QKV scheduling, 2048-token prefill chunks and the conservative M4 specialization setting. It needs no new shader algorithm. The pinned checkpoint is Q4; the known M4 Q3 specialization workaround does not affect it.

## Sustained decoding

Values are medians, with the candidate's measured range in parentheses. The M4 Pro-shaped baseline is run on this same M4 Max, not on a separate M4 Pro.

| Input tokens | 36/32/32 baseline packet tok/s | 64/64/64 candidate packet tok/s |
| ---: | ---: | ---: |
| 1,024 | 163.01 | 168.06 (165.13–174.16) |
| 8,192 | 148.08 | 162.18 (160.31–162.61) |
| 32,768 | 93.67 | 109.07 (96.80–113.75) |

Protocol: one warmup followed by five measured repetitions, 1,024 decode transitions after the initial prefill token, temperature 1 / top-p .95 / top-k 20, packet size 8, synchronized GPU timing, 2 GiB allocation cache. Scalar and packet variants alternate order and produce identical token hashes. Both profiles passed at all three contexts; their timed token sequences also match each other. Prefill and KV capacity allocation are excluded from decode timing. All candidate samples completed the requested length without EOS truncation. Inputs repeat/truncate one fixed chat prompt; this is a synthetic performance workload.

Hardware and software: macOS 26.6.2, MLX 0.32.0, mlx-lm 0.31.3, Python 3.12.7, AC power in automatic power mode. All 12 checkpoint files were verified against the pinned revision and size/SHA-256 manifest. [Dependency versions](results/environment.txt) and raw results record identities and host state.

Raw evidence: [36-worker extended baseline](results/w36-extended-default.json), [64-worker extended run](results/w64-extended-default.json), [64-worker exhaustive validator](results/w64-extended-default.exhaustive.json), and [checkpoint verification](results/checkpoint-verification.json). Repository revision: `e53663b262af4cb1ca498e596ca8b1a2d79e0800`.

## Why 200 tok/s became about 168

The initial 128-transition screen measured 200.44 tok/s for this same profile. The longer 1,024-transition measurement was 16.2% slower. Production source, generated shaders, dependency versions, cache configuration and startup allocation figures match. There is no evidence of a kernel change or fallback between them.

The identical 1K prefill also slowed from roughly .6 seconds to .8–1.1 seconds. Sequence growth alone therefore cannot explain the difference. The machine had other desktop activity; physical temperatures, GPU clocks and GPU utilization were not captured. No thermal cause is established. Treat 200 as a short screening result, not a sustained promise. See [measurement controls](measurement-notes.md).

After both extended runs, an alternating 64/36/64/36 control repeated the original short protocol. Packet medians were **172.62 / 158.58 / 176.54 / 159.91 tok/s**, all passing correctness. The 64-worker ranking persisted in these controls, but its original 200 tok/s absolute result did not. Together with the prefill slowdown, this supports reporting the later sustained measurements rather than attributing the entire change to output length. Raw controls: [1](results/control-1-w64.json), [2](results/control-2-w36.json), [3](results/control-3-w64.json), [4](results/control-4-w36.json).

## Candidate search and correctness

Fourteen synthetic expert-team/barrier configurations passed exact arithmetic checks. Eleven full-model configurations were then screened, including intermediate expert-team sizes and a shared-tile QKV alternative. The [screening summary](results/screening-summary.json) records all outcomes.

- 64/64/64 was fastest in the initial short screen at 200.44 tok/s; the existing M4 Pro-shaped 36/32/32 profile reached 174.87. Native eager MLX measured 71.11 in that baseline screen. These are short-run measurements, not interchangeable with the sustained table.
- The M5 Max-shaped 76/64/64 schedule reached 192.05 tok/s. More workers did not consistently improve speed.
- Intermediate 48/48/48 and 56/56/56 schedules passed short checks but did not win (179.09 and 136.01 tok/s).
- Shared-QKV 48/48/48 **failed packet token parity** and was rejected before timing. Static tile-coverage checks had passed; they did not prove full-kernel correctness. Shared-QKV 76/64/64 passed the short checks but was slower than its automatic-QKV counterpart (183.50 versus 192.05). No shared-QKV candidate is selected; the failure's root cause is unresolved.

The selected stock-transform 64/64/64 candidate passed:

- Exact native/scalar layer and logit checks, then scalar/packet token, layer, logit and KV checks at 1K, 8K and 32K.
- The upstream exhaustive validator: 61 cases, 72 native checks, a 520-step sliding-window case, nine EOS cases and three cancellation/cache-reuse cases. It covers additional sampling settings and packet lengths. Its separate performance timings use a different protocol and are not used here.

Separately, all 50 existing engine tests passed under an isolated 36/32/32 M4 Max profile override. Tests and validator completed without a reported synchronization failure.

## Serving path

The [production Engine probe](results/serving64.json) with Flint grammar enabled passed at **175.34 tok/s median**, range **174.95–192.34**. It used an 81-token chat prompt, 512 output tokens, one warmup and three measured repetitions. Every response completed all 512 tokens, all tokens were accepted by the grammar, prompt-cache hits were zero, and repeated text hashes matched. The separate tool check generated and parsed `add(a=2, b=3)` successfully; no tool was executed.

This measures Engine generation, including grammar, sampling and streamed-text handling. It excludes HTTP/network and scheduler response conversion. It uses the same isolated profile; it does not enable production hardware support. Its shorter prompt and output length differ from the sustained table, so the difference between those rates does not isolate grammar overhead. It does establish that this normal grammar-enabled Engine path avoids a collapse to generic eager-decode speeds on the tested prompt.

## Comparison limits and next integration step

The published M4 Pro chart reports 131.2 / approximately 113.7 / approximately 82.4 tok/s at 1K / 8K / 32K. This machine's selected-profile measurements are higher, but the chart does not publish enough workload detail for a matched speedup claim. No matched M5 Max rec measurement was found, so proximity to M5 Max is unproven. Bandwidth ratios are not performance measurements.

The subsequent local integration registers the tested M4 Max profile and updates CLI policy together, preserving explicit failures and the resident decoder. It verifies the 40-core device and rejects the untested 32-core SKU. Kernel source and context-capacity policy are unchanged; testing through 32K input does not qualify the entire advertised context capacity. See [local build](local-build.md).

Reproduction commands and measurement semantics are in the [experiment README](README.md). The [kernel audit](kernel-audit.md) explains scheduling constraints and rejected assumptions.
