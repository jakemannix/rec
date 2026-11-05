---
type: feature
feature: m4-max-profiling
last_updated: 2026-10-22
status: active
files:
  - experiments/m4_max/benchmark.py
  - experiments/m4_max/probe.py
  - experiments/m4_max/experimental_schedule.py
  - experiments/m4_max/serving_probe.py
  - engine/src/rec_mlx/kernels/device.py
  - engine/tests/test_device.py
  - cli/engine.go
  - cli/rec_test.go
tags:
  - doctrack/type/feature
  - doctrack/status/active
  - doctrack/audience/claude
---

# M4 Max profiling

Related: [[../_project|Project index]]. The experiment documentation in `experiments/m4_max/README.md` and its result artifacts contain the current measurements and reproduction commands.

## Scope and decisions

The user requested real kernel adaptation and performance evaluation, explicitly rejecting a chip-allowlist-only change or silent slow fallback. Production checks stay intact during qualification. Experimental profile injection occurs inside a fresh subprocess because decoder construction mutates global shader factories. Each candidate must pass exact native/scalar/packet checks before timings are accepted; GPU synchronization failures are failures, never grounds for increasing spin limits.

Local hardware: 40-core Apple M4 Max, 128 GiB, MLX architecture `applegpu_g16s`. The existing M4 Pro shader is the relevant starting point. M5 Max's worker allocation is a comparison candidate, not a hardware-derived default. The pinned checkpoint is Q4; `specialize_mixed_experts` affects Q3 only and cannot improve this checkpoint.

`benchmark.py` measures synchronized native MLX, scalar resident and packet decode with explicit sampling mode, warmup, repeated timings, prefill reporting, power information and source hashes. It rejects token divergence and records mismatch diagnostics. Detailed logits/layer/KV checks use greedy sampling; the optional upstream exhaustive validator covers additional sampling, cancellation, EOS and sliding-window cases.

`probe.py` exercises actual expert-team arithmetic/handoffs using a small synthetic bank. Its exact parity result is useful; its latency is not full-model throughput or proof of full-model residency. `experimental_schedule.py` enables independently audited intermediate MLP team sizes in the experiment only. The optional shared-QKV strategy exercises tile distribution beyond the stock 40-worker threshold.

## Measurement boundaries

The README's M4 Pro chart uses five runs and 1024 outputs, prefill excluded, but does not publish exact prompt, sampling or aggregation details. Same-device candidate comparisons are stronger evidence than direct cross-machine chart ratios. No matched M5 Max run is available. Never present bandwidth ratios as measured token speed.

## Current qualification

The selected 64/64/64 profile passed full-model exact parity at 1K, 8K and 32K, plus the upstream 61-case resident validator (sampling, EOS, cancellation/cache reuse and sliding-window coverage). Extended default-sampling packet medians are 168.06, 162.18 and 109.07 tok/s respectively. The earlier 200.44 tok/s screen used only 128 transitions; identical shaders and slower identical prefill indicate uncontrolled host variation in addition to workload differences. No thermal cause was established. The user considers about 160 tok/s sufficient; further peak tuning was not pursued.

All 50 existing engine tests passed separately with the 36/32/32 override. Fourteen synthetic schedules passed, but full-model shared-QKV 48/48/48 failed first-token parity and was rejected. Shared-QKV 76/64/64 was slower than its stock counterpart. Keep automatic QKV and existing 64-worker expert transforms for the selected candidate. Source-level tile accounting is not proof of full-kernel correctness.

`serving_probe.py` exercises production Engine grammar/sampling using the experimental profile and records actual generated counts, grammar acceptance, decode metrics and optional tool-call validation. It excludes HTTP and scheduler response conversion. See `experiments/m4_max/results.md` for the completed campaign and archived evidence.

The serving probe passed at 175.34 tok/s median (81-token prompt, 512-token responses, one warmup plus three measured repetitions), with complete grammar acceptance, cold-prefix assertions and a successful generated tool-call check. Extended 36/32/32 baseline medians were 163.01 / 148.08 / 93.67 tok/s. Alternating short 64/36/64/36 controls yielded 172.62 / 158.58 / 176.54 / 159.91, retaining the 64-worker ranking while confirming absolute timing drift. This does not identify a thermal cause or isolate grammar overhead.

The profiling campaign originally kept production selection unchanged. At the user's subsequent request, the validated profile was promoted into `device.py`, and the CLI chip allowlist now admits M4 Max. A cached, bounded `system_profiler` query verifies the 40-core GPU because MLX's device name does not distinguish the 32-core SKU. Unknown or unqualified core counts fail explicitly. Other chip profiles and shader transforms remain unchanged.

`test_device.py` checks all four supported profiles, unsupported identity/SKU rejection, core-count parsing/caching and detection failures. The source engine suite now passes 62 tests without profile injection. Local packaging uses the repository's standalone Python 3.12.12 build process and the unchanged pinned Flint model manifest. See `experiments/m4_max/local-build.md` for executable paths and launch commands.

Local build completed as `.build/rec-m4-max`, version `0.1.0-m4max-local`. Full `make test` passed. The packaged binary loaded the complete Flint checkpoint and passed 512-token authenticated HTTP generation, tool-call generation, streaming, invalid-key rejection and clean shutdown. Model files were hard-linked from the verified profiling checkpoint into the standard cache, without another download. Binary checksum and embedded device source hash were verified. The test server is stopped. Validation evidence is archived in `experiments/m4_max/results/local-build-verification.json`.

The user subsequently requested normal PATH-based usage. Installed the verified binary into `~/.local/share/rec/releases/0.1.0-m4max-local/rec`, with the installer's `.managed` hash marker, and linked `~/.local/bin/rec` to it. No previous `rec` existed on PATH, and `~/.local/bin` was already present; no shell configuration change was needed. The installed copy is independent of the build directory. Use `rec serve --context 32768`, then `rec launch pi` in another terminal.

Qualification covers this 40-core SKU and the tested context/sampling workload; it does not establish 32-core SKU support, the full 262K context limit or matched M5 Max performance.
