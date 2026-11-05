# M4 Max resident-kernel adaptation audit

This report combines a source audit with synthetic GPU results collected by the coordinating agent on the actual M4 Max. The audit agent ran no GPU work. Production files were not changed. Paths and line references refer to this checkout; paths beginning `kernels/` are relative to `engine/src/rec_mlx/`. The initial source/synthetic audit is retained below; the dated campaign-outcome addendum records subsequent full-model evidence and supersedes prospective statements.

## Main conclusion

The strongest starting point is already inside this repo: adapt its M4 Pro resident shader to a larger Max schedule. The known M4 numerical workaround concerns Q3 specialization and does not alter the pinned Q4 checkpoint. The M4 Pro uses the optimized resident pipeline, not a generic fallback. The M5 Max profile demonstrates how the authors scale this same source: 76 resident workers, 64 MLP workers, 64 vocabulary-head workers, and 4096-token native prefill chunks instead of M4 Pro's 36/32/32/2048. This is concrete evidence that a schedule adaptation is intended by the implementation; it is not evidence that those exact numbers are optimal or safe on M4 Max.

`engine/src/rec_mlx/kernels/device.py:15-39` is the entire chip-specific policy. No alternate M5-only instruction set or separately authored M5 kernel was found in these files. The kernels implement quantized matrix-vector work with scalar/vector arithmetic, SIMD reductions/shuffles, threadgroup memory, and device atomic synchronization. The engine hardcodes the Flint model geometry; wholesale substitution of arbitrary public GEMM kernels would require preserving quantization, rounding, attention layout, and persistent scheduling semantics.

## Pinned model nuance

Root confirmed this machine is a 40-GPU-core Apple M4 Max, `applegpu_g16s`, 128 GiB RAM. It shares the M4 Pro architecture identifier. Root also confirmed the pinned downloadable checkpoint has `routed_expert_bits=4`. The specialization flag only changes generated source inside `if bits == 3` in `mixed_bits.py`. Therefore the known M4 Q3 specialization workaround does **not** change this pinned Q4 model at all; true/false sweeps would compile identical kernels. Retaining false remains sensible for eventual mixed-Q3 support, but no Q4 performance deficit can be attributed to this flag.

## Existing optimizations to retain

- `kernels/__init__.py:11-52`: optimized build applies expert teams, faster RMS/global barriers, own-rank routing, KV teams, mixed-bit transformation, guarded fast activation, and resident packet sampling.
- `kernels/resident_optimizations.py:10-54`: reduces RMS synchronization and replaces reset-based global barrier with monotonically increasing arrival count.
- `kernels/expert_teams.py:4-17`: partitions MLP workers into six routed expert teams plus a double-width shared expert team. Supported counts are explicitly 32 or 64.
- `kernels/kv_teams.py:81-150`: four independent KV-head teams; each team handles eight Q heads plus one K and one V head. Team-local barriers avoid unnecessary whole-GPU joins.
- `kernels/rank_routing.py:17-53`: each expert team computes only its rank in top-six selection; routed scores survive scratch reuse.
- `kernels/activation_optimization.py:19-122`: exhaustive BF16 activation guard qualifies fast exp on actual hardware (65,536 gate patterns and 16 up patterns).
- `kernels/resident_block.py:60-88`, `kernels/sampling.py:17-60`: up to eight generated tokens in one persistent kernel; greedy and temperature-one kernels remain separate because even a uniform branch affects register allocation and throughput.

Production construction calls `build_decoder(..., optimized=True)` at `engine.py:90`. Prefill is intentionally native MLX (`quantized.py:122-136`), regardless of chip. Calling this native prefill a fallback would be misleading.

## Candidate matrix

Treat each candidate as an experimental profile, built in a fresh process. Specialization is irrelevant for the pinned Q4 checkpoint; retain the conservative false default for any later Q3 investigation.

| Label | Resident workers | MLP workers | Head workers | Purpose |
| --- | ---: | ---: | ---: | --- |
| Existing M4 recipe | 36 | 32 | 32 | Establish correctness and an M4 Pro-shaped baseline on the Max |
| Intermediate | 40, 48, 56, 60 | 32 | 32, then up to worker count | Measure whether attention/head scale before doubling expert teams |
| Doubled MLP | 64 | 64 | 64 | Scale routed/shared expert teams using existing transform |
| Near Max recipe | 68, 72, 76 | 64 | 64 | Find resident occupancy limit and global-sync overhead |
| M5 Max recipe control | 76 | 64 | 64 | Same schedule as repo's M5 Max, with M4 numerical workaround |

Test native prefill chunk size 2048 versus 4096 independently. A 4096 choice changes prefill throughput/memory, not resident decode. Once a schedule wins, experiment with short attention heads 4 versus 8, long heads 8 versus 4, long-context threshold, and unroll 1/2/4 only with renewed parity and occupancy validation. These options exist in `megakernel.py:1134-1195`, but the optimized transform assumes all attention/merge/router worker counts equal the resident pool.

At the time of this initial audit, all table entries were hypotheses to measure, not claimed supported configurations. Subsequent results are summarized in the campaign-outcome addendum. If root discovers a smaller GPU SKU, tune accordingly. Do not infer concurrency solely from the marketing core count.

## Constraints and hazards

1. **Residency is correctness-critical.** `sync_all` waits for every worker while retaining its registers and threadgroup memory. Launching more workers than can run concurrently may starve waiting workers and hit the 10,000,000-spin timeout. MLP handoff also spins across expert-team peers (`megakernel.py:800-819`). Diagnose any timeout and reject that candidate; do not increase timeout to hide oversubscription. Kernel instruction mix and packet wrapping can change register pressure, so a tiny barrier benchmark is insufficient to establish full-model safety.
2. **M4 compile-time specialization is known-bad in this checkout.** `device.py:38` explicitly says specialized M4 shader fails scalar/packet parity. `mixed_bits.py:76-113` turns shared/routed selection into distinct template instantiations for Q3 routed versus Q4 shared experts. Start with `specialize_mixed_experts=False`, exactly as M4 Pro. This retains custom mixed-bit math; it does not route to native inference. Whether the root cause is compiler arithmetic, register allocation, or scheduling is unknown from source. The pinned Q4 checkpoint does not enter this specialized branch.
3. **Global transforms are stateful.** `kernels/__init__.py:4` explicitly requires one engine per process. `_HEADER` is replaced and `_metal` wrapped globally. Rebuilding several candidates in one interpreter stacks transformations, corrupts anchors, and invalidates comparisons.
4. **Worker arithmetic assumes four KV teams.** `kv_teams.py:82` uses floor division with no explicit divisibility guard. Use worker counts divisible by four. An odd remainder creates invalid team mapping/synchronization.
5. **MLP supports only 32 or 64 workers today.** `expert_teams.py:5` asserts it. Supporting 48 could be a separate tuning experiment, but requires audit of team row tiling, own-rank routing, and handoffs; it is not currently a supported config knob.
6. **Routing scratch caps the unchanged layout at 80 workers.** `_LAYOUT` allocates 1280 uints; each resident worker uses 16 entries (`megakernel.py:35-43,835-838`). Do not test >80 workers without changing and verifying the layout.
7. **36 to 40 workers changes QKV algorithm.** Below 40, `kv_teams.shared_projection` distributes individual projection tiles using atomic tickets. At 40 and above, it assigns whole heads to the first ten members of each KV team; remaining members wait at that stage. Consequently gains need not be monotonic, and the M5 Max recipe leaves some groups idle during QKV.
8. **At short context, more workers may mostly idle.** Attention uses 32 partitions and usually four query heads per SIMD. KV team owns 64 tasks, while 36 workers give 72 SIMD groups/team and 76 give 152. Similarly merge has 64 tasks/KV team; larger pools do not increase useful work at every phase. More cores mainly help other stages or longer contexts until schedule/layout is retuned.
9. **Existing profiler is not final optimized-path profiling.** `kv_teams.py:85` asserts global profiling is disabled and `guard` insists `profile=False`. `FullModel.last_profile` records spin counts/arrival order, not timestamps, and its phase-label scheme should be checked against emitted barriers before use. Use GPU timing/profiling for final kernel or add purposeful instrumentation; avoid presenting unoptimized spin counts as production phase durations.
10. **Bitwise rounding is part of correctness.** `mixed_bits.py:37-58` reproduces BF16 router values, BF16 expert products, and BF16 rounding after each sum. Preserve this sequence; looser allclose tests can hide routing/token drift.
11. **Sampling modes use different kernels.** Greedy and temperature-one use persistent resident8; filtered sampling, presence penalty, grammar, or logprobs use `FilteredResidentSampler` (`sampling.py:80-98`). Benchmark the user's likely serving settings as well as peak greedy throughput.

## Acceptance checks

Use `engine/tests/validate_resident.py` for the final profile. It already compares native layer traces/logits against scalar resident decoding, scalar against resident packet outputs, exact KV contents, seeded sampling, EOS/cancellation behavior, sliding-window wrap, and 8192 context. Its 520-token run crosses the 513-token local window. Performance section alternates scalar/packet order, warms up once, and reports medians of subsequent runs.

Extend long-context checks around partition transitions: 1023/1024/1025, 8192/8193, 32768/32769, 65536/65537, plus practical larger-context case if memory/time allow. These boundaries change partition count, rounded attention path, or cache bank behavior (`megakernel.py:890-899`). Check greedy and temperature-one persistent kernels independently. A passing activation guard or isolated Q3 dot microbenchmark does not substitute for complete model/token parity.

For honest performance claims, record exact hardware SKU, MLX version, OS, power mode, model config/checkpoint identity, context, emitted token count, sampler settings, cold compilation separately, steady-state median/range, and peak memory. Compare Max schedule versus 36/32/32 on this same device. This establishes adaptation gain but is not a cross-machine M4 Pro or M5 Max result.

## Likely next optimizations if direct scaling plateaus

These are inferences from source, not measured bottlenecks:

- Vocabulary head is 150144 x 2048 Q4 and parallelized by `head_workers`; increasing 32 to 64 plausibly matters because every token streams many head weights.
- Routed expert work runs on 47 of 49 layers; doubling expert teams reduces each group's per-expert row workload but can increase idle/spinning workers and hurt residency.
- Global barriers occur between serial dependent phases. More workers do not necessarily reduce latency, particularly with redundant normalization and residual updates in every group.
- Q3 unpack work and dynamic mixed-bit branch may cost more on M4 than specialized M5 path. If it is measured as material, isolate the specialized parity failure first, then try narrower helpers/explicit arithmetic preserving the BF16 operation order, validating both scalar and packet variants.
- KV head-team work mapping may merit a Max-specific tile scheduler: existing >=40 path uses only 40 groups for QKV regardless of pool size; dynamic tiles can involve more groups but add atomics. Benchmark both on identical model workloads before changing production defaults.

## Experimental implementation and follow-up findings

Added isolated `experiments/m4_max/probe.py` and `experimental_schedule.py`; production files remain unchanged. Probe compiles actual expert-team gate/up/down, handoff, and monotonic global barrier functions with six synthetic Q4 routed experts plus shared expert at production dimensions. It checks exact activation/down parity against native MLX, then reports repeated-block timings with an explicit warning that this is a smaller, cache-friendly shader. Root reports successful exact parity/barriers on M4 Max for worker/MLP pairs 36/32, 40/32, 48/32, 64/32, 64/64, 72/64, 76/64, 80/64. Root found timings noisy (approximately 0.06–0.13 ms/block); treat these as correctness and basic barrier evidence, not inference speed or full-model residency qualification.

Intermediate MLP counts 40/48/56 are blocked by the production assert, but their tiling is structurally valid. `team=mlp_workers//8` gives5/6/7 groups per routed expert and twice that for shared. Gate/up emits pairs, down emits quads; all dimensions align, and loop bounds cover each row exactly once even though strides do not divide every width. Experimental helper performs an explicit exhaustive row-ownership check. It produces byte-identical source to the production transform for32/64. Intermediate counts require an explicit `--experimental-intermediate-mlp` flag in the probe; full model benchmark integrates same helper. At this initial-audit stage, full-model correctness/performance still needed measurement; subsequent outcomes are recorded below.

The initial static review found consistent tile and ticket accounting in the original dynamic QKV scheduler (`kv_teams.shared_projection`) when groups>=10. This was not a proof of full-kernel correctness; a subsequent shared48 full-model run failed parity, as recorded below. Each KV team executes320 valid work units (10 heads x32 four-row tiles). Each SIMD advances the atomic ticket once more to exit; next-layer base advances by320+groups*8, exactly covering valid and terminal tickets. Head normalization/publication loops remain valid for groups>=10. An isolated `--qkv-strategy shared` alternative was added to compare dynamic tile distribution with the default >=40 whole-head schedule. This is potentially useful on Max because it puts more than40 workers to work during projection; atomic overhead could negate the gain, so measure complete decode and parity.


## Verified synthetic result files

Read directly from `.build/m4-results/probe*.json`. Every row passed exact BF16 activation and down-projection parity (zero mismatches, finite results), plus the probe’s actual expert-team and global barriers. These are one synthetic seed and a smaller shader, not full-model qualification.

| Result file | Resident workers | MLP workers | Exact activation/down |
| --- | ---: | ---: | --- |
| `probe-40-32.json` | 40 | 32 | Passed |
| `probe-40-40.json` | 40 | 40 | Passed |
| `probe-48-32.json` | 48 | 32 | Passed |
| `probe-48-48.json` | 48 | 48 | Passed |
| `probe-56-56.json` | 56 | 56 | Passed |
| `probe-60-48.json` | 60 | 48 | Passed |
| `probe-64-32.json` | 64 | 32 | Passed |
| `probe-64-48.json` | 64 | 48 | Passed |
| `probe-64-64.json` | 64 | 64 | Passed |
| `probe-72-64.json` | 72 | 64 | Passed |
| `probe-76-48.json` | 76 | 48 | Passed |
| `probe-76-64.json` | 76 | 64 | Passed |
| `probe-80-64.json` | 80 | 64 | Passed |
| `probe36.json` | 36 | 32 | Passed |

The intermediate results 40/40, 48/48, 56/56, 60/48, 64/48 and 76/48 provide hardware evidence that the isolated intermediate expert-team adaptation preserves Q4 arithmetic on this device. They do not establish whether any such configuration beats the existing 32/64 scheduling choices in complete inference.

## Full-model harness correctness review

Reviewed `benchmark.py` and its `apply_experiments` overrides, together with `experimental_schedule.py`. There is no separate `experiment_config.py` in the final implementation. No serious correctness defect was found in this review. Key checks: overrides occur before optimized transform imports; every full-model invocation gets a fresh process; intermediate team helper preserves production source at32/64; static dynamic-QKV work/ticket accounting checks found no defect above40 workers, but shared48 subsequently failed full-model parity; native/scalar/packet caches are independent; parity checks compare exact layers/logits/KV; timings exclude prefill and capacity growth; non-greedy timed token chains are compared across selected variants; EOS-shortened runs record actual counts.

Scope limits remain explicit: fast sampling variants get greedy exact parity first and non-greedy token-chain checks unless the exhaustive validator is requested; a run ending on EOS can validate fewer than the requested parity steps; single-variant timing cannot independently cross-check its timed token chain. These are recorded harness limitations, not identified source defects. These review checks alone do not establish compiler behavior or residency; the later full-model observations below qualify only the tested configurations and workloads.


## Campaign outcome addendum — 2026-10-22

The initial source/synthetic audit above preceded complete-checkpoint execution. Its static coverage and ticket-accounting findings must not be read as qualification of every experimental schedule.

- **Shared48 failed and remains unqualified.** `w48-m48-h48-shared-default-short.json` records a packet-token parity failure. The failure occurs on the first generated packet token, despite passing synthetic/static checks. No compiler, synchronization or indexing root cause is established. Do not enable this configuration or relax parity to admit it.
- **Shared76 passed the short checks but was slower in that screen.** Its filtered packet median was 183.50 tok/s, versus 192.05 tok/s for stock76, using 1K input and 128 timed decode transitions. This is evidence from the short campaign, not a universal performance ordering.
- **Stock64 passed the extended full-model campaign and upstream exhaustive validator.** With 64/64/64 workers, stock automatic QKV, default filtered sampling, 1024 timed decode transitions and five measured repetitions, packet medians were 168.064 tok/s at 1K input, 162.175 at 8K and 109.073 at 32K. The full-model harness required greedy native/scalar/packet parity and matched sampled token chains; the exhaustive validator broadened coverage of sampling and stopping behavior. This establishes the recorded checks, not all contexts, models, M4 Max SKUs or deployment scenarios.
- Intermediate MLP schedules reached full-model screening too: stock-QKV 48/48/48 and 56/56/56 passed their short 1K tests. Their passing synthetic tests did not predict their complete inference performance, and neither result qualifies the separate shared-QKV path.

The 36/32/32 profile running on this same M4 Max is the relevant adaptation control; its completed extended medians were 163.01 / 148.08 / 93.67 tok/s at 1K / 8K / 32K, with all checks passing. The separate grammar-enabled Engine probe passed at 175.34 tok/s on an 81-token prompt and 512-token responses, including a successful generated tool-call check. There is no matched M5 Max measurement. Do not present the bandwidth ratio as measured M5 proximity, or the published M4 Pro chart as a matched benchmark. Short and sustained absolute timings varied, and GPU utilization and physical temperature were not measured; see [measurement notes](measurement-notes.md).

Final paired measurements and the resulting recommendation belong in [results.md](results.md). Production support remains unchanged; the campaign tests process-local experimental profiles.
