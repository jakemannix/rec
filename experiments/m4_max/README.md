# M4 Max kernel experiments

Measure whether rec's existing optimized resident kernels can be tuned for M4 Max, with correctness checks and explicit failures. The campaign kept production selection unchanged; the validated 64/64/64 profile has since been integrated into the [local build](local-build.md). Full-model validation passes through the tested 32K context workload; see [results and qualification limits](results.md). A primitive probe or one short prompt does not establish supported inference.

The research machine is a 40-GPU-core Apple M4 Max with 128 GiB unified memory. MLX 0.32.0 reports `applegpu_g16s`, the same architecture key used by the supported M4 Pro profile. These findings do not qualify the 32-core M4 Max SKU.

## Setup

Run from the repository root. The local experiment uses Python 3.12.7 in `.build/m4-profile-venv`. Install the project with its validation dependencies to use the pinned versions and build the native grammar extension:

```sh
uv venv --python 3.12.7 .build/m4-profile-venv
uv pip install --python .build/m4-profile-venv/bin/python -e './engine[validation]'
```

The scripts import `engine/src` directly. Download the checkpoint specified by `packaging/flint-int4-mlx.json` into `.build/m4-profile-model`. The manifest pins Hugging Face revision `3b93f9053b530488405ae9fca51d4900d6ce21f0` and every file's size and SHA-256. This command reuses verified files and verifies downloads before renaming them:

```sh
.build/m4-profile-venv/bin/python - <<'PY'
import hashlib
import json
import urllib.request
from pathlib import Path

manifest = json.loads(Path('packaging/flint-int4-mlx.json').read_text())
destination = Path('.build/m4-profile-model')
destination.mkdir(parents=True, exist_ok=True)

def valid(path, item):
    if not path.exists() or path.stat().st_size != item['size']:
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256']

for item in manifest['files']:
    path = destination / item['name']
    if valid(path, item):
        continue
    partial = path.with_suffix(path.suffix + '.download')
    print('Downloading', item['name'], flush=True)
    urllib.request.urlretrieve(item['url'], partial)
    assert valid(partial, item), f'Checkpoint verification failed: {partial}'
    partial.replace(path)
PY
```

The complete checkpoint is approximately 22.6 GB. The current checkpoint uses **four-bit routed experts**. `--specialize-mixed-experts` affects only the three-bit branch and is not a tuning knob for this checkpoint.

## Run

Run GPU experiments sequentially, with other GPU workloads idle and stable power/thermal conditions. Each full benchmark starts a fresh subprocess because kernel transforms mutate module factories. Existing result files are not overwritten.

Start with the M4 Pro team configuration as a baseline on this M4 Max:

```sh
.build/m4-profile-venv/bin/python experiments/m4_max/benchmark.py \
  --model .build/m4-profile-model \
  --output .build/m4-results/baseline-36-32-32.json \
  --workers 36 --mlp-workers 32 --head-workers 32 \
  --contexts 1024,8192,32768 --steps 128 --repeats 3 \
  --variants native,scalar,packet --sampling greedy --timeout 1800
```

Change one team allocation at a time and give each run a new output path. Stock expert-team schedules support `--mlp-workers 32` or `64`; total workers must fit all teams. More workers need not improve throughput or synchronization. Compare each candidate with the same baseline workload before extending its context coverage.

Two optional flags enable isolated prototypes inside the benchmark subprocess:

- `--experimental-intermediate-mlp` permits `--mlp-workers 40`, `48` or `56` through an experimental expert-team transform. Static row-coverage checks do not establish full-model correctness or residency.
- `--qkv-strategy shared` forces the shared tiled QKV projection instead of stock `auto` scheduling. This tests whether the stock strategy switch at larger worker pools causes a performance penalty.

For example, append `--workers 56 --mlp-workers 48 --experimental-intermediate-mlp` to a command using a new result path; test `--qkv-strategy shared` separately before combining changes. These flags patch only the child process and record transformed-source identities. They do not enable production support, and require the same full-model parity and throughput checks as other candidates.

For a candidate that passes short full-model checks, broaden validation and measurement:

```sh
.build/m4-profile-venv/bin/python experiments/m4_max/benchmark.py \
  --model .build/m4-profile-model \
  --output .build/m4-results/candidate-extended.json \
  --workers 64 --mlp-workers 64 --head-workers 64 \
  --contexts 1024,8192,32768 \
  --steps 1024 --repeats 5 --parity-steps 16 \
  --variants scalar,packet --sampling default \
  --exhaustive-validation --timeout 3600
```

Repeat selected configurations with `--sampling unfiltered` (temperature 1, top-p 1, top-k 0) and `--sampling default` (temperature 1, top-p .95, top-k 20). Prefill chunk tuning uses `--prefill-chunk-size`; assess its separately recorded prefill time. Longer native reference runs and the exhaustive validator can take substantial time; the parent records timeout/failure status and terminates its process group.

To measure the production Engine with Flint grammar, run the separate serving probe in a fresh process after all other GPU experiments finish:

```sh
.build/m4-profile-venv/bin/python experiments/m4_max/serving_probe.py \
  --model .build/m4-profile-model \
  --output .build/m4-results/serving-64.json --tool-check
```

It uses the selected 64/64/64 profile, one warmup and three measured 512-token responses with cold prefill. It records actual counts, Engine decode metrics and grammar acceptance, and optionally checks a generated tool call without executing it. It excludes HTTP/network and scheduler response conversion. Run under an external timeout when automating it.

`probe.py` tests real Q4 expert-team arithmetic with synthetic weights and barriers. Its small, cache-friendly workload cannot establish full-model residency, decoding correctness or tokens/second. Use its results only to shortlist full-checkpoint candidates.

## Measurement and limits

- `native`, `scalar`, and `packet` use the native MLX model, rec scalar resident decode, and rec packet decode respectively. The selected optimized path is required; there is no silent fallback.
- Before timing, greedy parity compares native/scalar layer states and logits, then packet/scalar outputs and KV state. Non-greedy timed runs compare token hashes; their detailed parity coverage is narrower unless the upstream exhaustive validator is also run.
- Timing synchronizes the GPU, excludes prefill and KV capacity allocation, warms each variant, and rotates variant order. Prefill time is recorded separately. Startup uses no allocation cache; decode defaults to a 2 GiB cache. `MLX_METAL_FAST_SYNCH` defaults to 1, while explicit environment overrides are preserved and recorded.
- `--steps` counts decode transitions **after** the initial token sampled from prefill. A complete 1024-step run generates 1025 total output tokens. EOS is respected, and shortened samples report actual count and completion status. Do not present a shortened run as a complete requested-length measurement.
- Contexts repeat and truncate a fixed tokenized chat prompt. This is a synthetic workload, not a claim about realistic prompt distributions. Repetitions report median, range and standard deviation, with dependency versions, source/kernel hashes and host state.
- The upstream M4 Pro chart documents five repetitions and 1024 outputs with prefill excluded, but not sampling settings, prompt construction, first-token accounting or aggregation statistic. Its reconstructed values are comparison context, not a matched control. No published M5 Max rec benchmark was found in this investigation.
- The optimized kernel rejects global phase profiling. Enabling the base kernel's spin/arrival counters requires a separate supported instrumentation change; current error diagnostics remain active. Threadgroup API limits do not prove simultaneous worker residency.

## Research

[Detailed findings and reconstructed chart values](research.md) include source links and adaptation priorities. Key evidence:

- [Recursive maintainer: M4 Max has not been thoroughly tested](https://github.com/recursive-oss/rec/issues/1#issuecomment-6322920897).
- [Apple M4 Pro/Max specifications](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/): 273/546 GB/s for the full configurations.
- [Apple M5 Pro/Max specifications](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/): M5 Max reaches 614 GB/s and adds GPU Neural Accelerators.
- [Apple's MLX measurements](https://machinelearning.apple.com/research/exploring-llms-mlx-m5) distinguish compute-heavy prefill from bandwidth-heavy generation; these are other models, not rec performance promises.
- [Pinned MLX NAX hardware gate](https://github.com/ml-explore/mlx/blob/v0.32.0/mlx/backend/metal/device.cpp#L943): the M4's generation 16 does not qualify for the M5 accelerator path.

## Results

The selected 64/64/64 profile measured **168 / 162 / 109 tok/s** at **1K / 8K / 32K** input contexts in the extended campaign. It passed exact parity checks and the upstream 61-case validator. The initial 200 tok/s short screen is not a sustained throughput promise.

The separate production Engine probe with Flint grammar measured **175 tok/s median** on an 81-token prompt and 512-token response. All grammar acceptance checks and the tool-call check passed. This excludes HTTP/network overhead and uses a different workload from the extended benchmark.

See [the results report](results.md) for the baseline comparison, rejected candidates, serving measurement and limitations. Raw measurements are archived in [`results/`](results/); detailed host-variation notes are in [measurement-notes.md](measurement-notes.md).
