---
project: rec
type: index
doctrack_version: "3.0.0"
last_updated: 2026-10-22
tags:
  - doctrack/type/index
  - doctrack/status/active
  - doctrack/audience/claude
---

# rec

This vault currently documents the M4 Max profiling work only. It is not a full codebase inventory.

## Features

- [[features/m4-max-profiling|M4 Max profiling]]: isolated device profiles, correctness probes and performance measurement.

## File registry

| Source file | Documentation |
| --- | --- |
| `experiments/m4_max/benchmark.py` | [[features/m4-max-profiling]] |
| `experiments/m4_max/probe.py` | [[features/m4-max-profiling]] |
| `experiments/m4_max/experimental_schedule.py` | [[features/m4-max-profiling]] |
| `experiments/m4_max/serving_probe.py` | [[features/m4-max-profiling]] |
| `engine/src/rec_mlx/kernels/device.py` | [[features/m4-max-profiling]] |
| `engine/tests/test_device.py` | [[features/m4-max-profiling]] |
| `cli/engine.go` | [[features/m4-max-profiling]] |
| `cli/rec_test.go` | [[features/m4-max-profiling]] |

The connected Obsidian/Doctrack servers target a different repository. Read and update this local vault directly until a server targets this project.
