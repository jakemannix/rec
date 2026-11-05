"""Probe actual Q4 expert-team math and barriers without loading Flint weights.

Run one candidate per process. This smaller kernel cannot qualify full-model
residency or predict token throughput; use benchmark.py for those questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engine" / "src"))


SOURCE = r"""
    uint tid=thread_index_in_threadgroup,lane=thread_index_in_simdgroup;
    uint group=threadgroup_position_in_grid.x,phase=0;
    device atomic_uint* control=(device atomic_uint*)private_control;
    threadgroup uint ready[1];
    threadgroup uint d[64];
    threadgroup bfloat16_t n[2048];
    device uint* ids=routing+group*16;
    if(tid<64)d[tid]=metadata[tid];
    if(tid<6)ids[tid]=tid;
    if(tid==0) {
        diagnostic[group]=0;
        atomic_store_explicit((device atomic_uint*)(routing+group*16+14),0u,memory_order_relaxed);
    }
    for(uint j=0;j<8;++j)n[tid+j*256]=input[tid+j*256];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #define SYNC() if(!sync_all(control,tid,phase,ready,profile,group)){if(tid==0)diagnostic[group]=atomic_load_explicit(control+2,memory_order_relaxed);return;}
    SYNC();
    for(uint step=0;step<__STEPS__;step++) {
        if(group<__TEAM_MLP_WORKERS__) {
            expert_team_gateup(bank,d,n,ids,silu_table,activation,lane,simdgroup_index_in_threadgroup,group);
            if(!expert_team_handoff(routing,control,tid,group,step,phase,ready)) {
                if(tid==0)diagnostic[group]=atomic_load_explicit(control+2,memory_order_relaxed);
                return;
            }
            expert_team_down(bank,d,activation,ids,down,lane,simdgroup_index_in_threadgroup,group);
        }
        SYNC();
    }
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=36)
    parser.add_argument("--mlp-workers", type=int, choices=(32, 40, 48, 56, 64), default=32)
    parser.add_argument("--experimental-intermediate-mlp", action="store_true")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.mlp_workers <= args.workers <= 80 or args.workers % 4:
        parser.error("workers must be a multiple of 4, fit MLP workers, and be <=80")
    if args.steps < 1 or args.repeats < 1:
        parser.error("steps and repeats must be positive")
    if args.mlp_workers not in (32, 64) and not args.experimental_intermediate_mlp:
        parser.error("MLP 40/48/56 requires --experimental-intermediate-mlp")

    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.activations import swiglu
    from rec_mlx.kernels import expert_teams, megakernel, resident_optimizations

    mx.random.seed(args.seed)
    mx.set_cache_limit(256 * 1024**2)
    metadata = [0] * 64
    pieces = []
    matrices = {}
    offset = 0
    # Six synthetic routed experts use the actual 512x2048 / 2048x512
    # geometry. Shared expert is twice as wide. Only selected experts are
    # allocated: timings therefore represent a small, cache-friendly probe.
    for name, slot, shape in (
        ("gate", 30, (6, 512, 2048)),
        ("up", 35, (6, 512, 2048)),
        ("down", 40, (6, 2048, 512)),
        ("shared_gate", 45, (1024, 2048)),
        ("shared_up", 50, (1024, 2048)),
        ("shared_down", 55, (2048, 1024)),
    ):
        weights = (mx.random.normal(shape) * (shape[-1] ** -0.5)).astype(mx.bfloat16)
        quantized = mx.quantize(weights, group_size=64, bits=4)
        mx.eval(*quantized)
        matrices[name] = quantized
        locations = []
        for tensor in quantized:
            padding = (-offset) % 256
            if padding:
                pieces.append(mx.zeros((padding,), mx.uint8))
                offset += padding
            locations.append(offset)
            pieces.append(tensor.reshape(-1).view(mx.uint8))
            offset += tensor.nbytes
        metadata[slot : slot + 5] = [*locations, shape[-2], shape[-1]]
    bank = mx.concatenate(pieces)
    descriptor = mx.array(metadata, mx.uint32)
    x = mx.random.normal((2048,)).astype(mx.bfloat16)
    gates = (mx.arange(65536, dtype=mx.uint32) << 16).view(mx.float32).astype(mx.bfloat16)
    silu_table = swiglu(gates, mx.ones_like(gates))
    mx.eval(bank, descriptor, x, silu_table)

    def matmul(name: str, value, expert: int | None = None):
        packed = matrices[name]
        if expert is not None:
            packed = tuple(item[expert] for item in packed)
        return mx.quantized_matmul(value, *packed, transpose=True, group_size=64, bits=4)

    expected_activation, expected_down = [], []
    for expert in range(6):
        gate = matmul("gate", x[None], expert)
        up = matmul("up", x[None], expert)
        act = nn.silu(gate) * up
        expected_activation.append(act.reshape(-1))
        expected_down.append(matmul("down", act, expert).reshape(-1))
    gate = matmul("shared_gate", x[None])
    up = matmul("shared_up", x[None])
    act = nn.silu(gate) * up
    expected_activation.append(act.reshape(-1))
    expected_down.append(matmul("shared_down", act).reshape(-1))
    expected_activation = mx.concatenate(expected_activation)
    expected_down = mx.concatenate(expected_down)
    mx.eval(expected_activation, expected_down)

    team_transform = expert_teams.transform
    if args.experimental_intermediate_mlp:
        from experimental_schedule import expert_teams_transform

        team_transform = expert_teams_transform
    header = resident_optimizations.transform(team_transform(megakernel._HEADER, args.mlp_workers))
    substitutions = {
        "WORKERS": args.workers,
        "GROUPS": args.workers * 8,
        "MLP_GROUPS": args.mlp_workers * 8,
        "ATTENTION_GROUPS": args.workers * 8,
        "MERGE_GROUPS": args.workers * 8,
        "ROUTER_GROUPS": args.workers * 8,
        "TEAM_MLP_WORKERS": args.mlp_workers,
        "PROFILE": 0,
        "UNROLL": 1,
        "STEPS": args.steps,
    }
    source = SOURCE
    for name, value in substitutions.items():
        header = header.replace(f"__{name}__", str(value))
        source = source.replace(f"__{name}__", str(value))
    kernel = megakernel._metal(
        "m4_max_expert_team_probe",
        input_names=["bank", "metadata", "input", "silu_table", "private_control"],
        output_names=["activation", "down", "routing", "diagnostic", "profile"],
        header=header,
        source=source,
        compile_options={"math_mode": "safe"},
    )

    def run():
        # A CPU-uploaded fresh control buffer matches production isolation.
        result = kernel(
            inputs=[bank, descriptor, x, silu_table, mx.array([0] * 1024, mx.uint32)],
            grid=(args.workers * 256, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[(4096,), (7 * 2048,), (1280,), (args.workers,), (1,)],
            output_dtypes=[mx.bfloat16, mx.bfloat16, mx.uint32, mx.uint32, mx.uint32],
        )
        mx.eval(*result)
        diagnostics = result[3].tolist()
        if any(diagnostics):
            raise RuntimeError(f"Expert-team/global barrier failed: {diagnostics}")
        return result

    result = {
        "kind": "synthetic_q4_expert_team_probe",
        "device": mx.device_info(),
        "configuration": {key: value for key, value in vars(args).items() if key != "output"},
        "bank_bytes": bank.nbytes,
        "shader_sha256": hashlib.sha256((header + source).encode()).hexdigest(),
        "limitation": (
            "Tests actual expert-team math/barriers in a smaller shader; "
            "cannot establish full-model residency or token throughput."
        ),
    }
    try:
        actual = run()
        parity = {}
        for name, candidate, reference in (
            ("activation", actual[0], expected_activation),
            ("down", actual[1], expected_down),
        ):
            parity[name] = {
                "exact": bool(mx.array_equal(candidate, reference).item()),
                "mismatched_elements": int(mx.sum(candidate != reference).item()),
                "max_abs_error": float(
                    mx.max(
                        mx.abs(candidate.astype(mx.float32) - reference.astype(mx.float32))
                    ).item()
                ),
                "finite": bool(mx.all(mx.isfinite(candidate)).item()),
            }
        result["parity"] = parity
        if not all(row["exact"] for row in parity.values()):
            raise RuntimeError("Synthetic expert math differs from native MLX reference")
        run()
        elapsed = []
        for _ in range(args.repeats):
            mx.synchronize()
            start = time.perf_counter()
            run()
            mx.synchronize()
            elapsed.append(time.perf_counter() - start)
        result["status"] = "passed"
        result["probe_seconds"] = elapsed
        result["median_ms_per_expert_block"] = 1000 * statistics.median(elapsed) / args.steps
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
