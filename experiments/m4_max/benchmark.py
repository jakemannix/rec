#!/usr/bin/env python3
"""Isolated, fail-closed M4 Max full-checkpoint correctness and throughput probe.

The parent imports no MLX. Every invocation creates one fresh child with a local
DeviceProfile override, leaving production checks and source files untouched.
"""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=36)
    p.add_argument("--mlp-workers", type=int, choices=(32, 40, 48, 56, 64), default=32)
    p.add_argument(
        "--experimental-intermediate-mlp",
        action="store_true",
        help="Opt into unaudited-on-GPU 40/48/56-worker expert schedules",
    )
    p.add_argument(
        "--qkv-strategy",
        choices=("auto", "shared"),
        default="auto",
        help="Use stock scheduling or force the shared tiled QKV projection",
    )
    p.add_argument("--head-workers", type=int, default=32)
    p.add_argument("--prefill-chunk-size", type=int, default=2048)
    p.add_argument("--specialize-mixed-experts", action="store_true")
    p.add_argument("--contexts", default="1024,8192,32768")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--parity-steps", type=int, default=8)
    p.add_argument("--packet-size", type=int, choices=range(1, 9), default=8)
    p.add_argument("--variants", default="native,scalar,packet")
    p.add_argument(
        "--sampling",
        choices=("greedy", "unfiltered", "default"),
        default="greedy",
        help="Timed sampling: greedy; temp1/top_p1; or production temp1/top_p.95/top_k20",
    )
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--cache-limit-mib", type=int, default=2048)
    p.add_argument(
        "--exhaustive-validation",
        action="store_true",
        help="Run the repository's full resident validator before this benchmark",
    )
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    a.contexts = [int(x) for x in a.contexts.split(",")]
    a.variants = a.variants.split(",")
    if not (32 <= a.workers <= 80 and a.workers % 4 == 0):
        p.error("workers must be a multiple of four in [32,80]")
    if not (1 <= a.head_workers <= a.workers and a.mlp_workers <= a.workers):
        p.error("head/MLP workers must fit the resident pool")
    if a.mlp_workers not in (32, 64) and not a.experimental_intermediate_mlp:
        p.error("MLP40/48/56 require --experimental-intermediate-mlp")
    if min(a.steps, a.repeats, a.parity_steps, a.prefill_chunk_size, *a.contexts) <= 0:
        p.error("steps, repeats, parity-steps, chunk size and contexts must be positive")
    if a.warmups < 1 or a.timeout <= 0:
        p.error("at least one warmup and a positive timeout are required")
    if a.cache_limit_mib < 0:
        p.error("cache-limit-mib must be nonnegative")
    if not a.variants or set(a.variants) - {"native", "scalar", "packet"}:
        p.error("variants must be a comma-separated subset of native,scalar,packet")
    a.model, a.output = a.model.resolve(), a.output.resolve()
    return a


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


def event(name, **fields):
    print(json.dumps(dict(event=name, **fields), default=str), flush=True)


def host_state():
    result = {}
    for command in (
        ["pmset", "-g", "batt"],
        ["pmset", "-g", "custom"],
        ["pmset", "-g", "therm"],
        ["sw_vers"],
    ):
        try:
            completed = subprocess.run(command, text=True, capture_output=True, timeout=5)
            result[" ".join(command)] = dict(
                returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[" ".join(command)] = dict(error=repr(exc))
    return result


def apply_experiments(mlp_workers, intermediate, qkv_strategy):
    """Patch only this process; retain exact transformed-source identities."""
    from rec_mlx.kernels import expert_teams, kv_teams

    metadata = dict(qkv_strategy=qkv_strategy, intermediate_mlp=mlp_workers not in (32, 64))
    if mlp_workers not in (32, 64):
        if not intermediate:
            raise ValueError("Intermediate expert schedules require explicit opt-in")
        from experimental_schedule import expert_teams_transform, validate_team_tiles

        validate_team_tiles(mlp_workers)
        metadata["expert_team_tile_validation"] = "passed_static_row_coverage"
        metadata["expert_schedule_helper_sha256"] = hashlib.sha256(
            Path(inspect.getfile(expert_teams_transform)).read_bytes()
        ).hexdigest()
        metadata["expert_team_original_sha256"] = hashlib.sha256(
            inspect.getsource(expert_teams.transform).encode()
        ).hexdigest()
        metadata["expert_team_experiment_sha256"] = hashlib.sha256(
            inspect.getsource(expert_teams_transform).encode()
        ).hexdigest()
        expert_teams.transform = expert_teams_transform
    if qkv_strategy == "shared":
        original = inspect.getsource(kv_teams.transform)
        anchor = "    if groups >= 10:\n"
        if original.count(anchor) != 1:
            raise RuntimeError("QKV strategy override expected exactly one stock branch anchor")
        source = original.replace(anchor, "    if False:  # experimental shared QKV strategy\n")
        scope = dict(kv_teams.__dict__)
        exec(compile(source, "<experimental_shared_qkv>", "exec"), scope)
        kv_teams.transform = scope["transform"]
        metadata["qkv_original_sha256"] = hashlib.sha256(original.encode()).hexdigest()
        metadata["qkv_experiment_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    return metadata


def parent(args):
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing result: {args.output}")
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child"]
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    timed_out = False
    try:
        code = process.wait(timeout=args.timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        timed_out = time.monotonic() - started >= args.timeout
        os.killpg(process.pid, signal.SIGKILL)
        code = process.wait()
    result = json.loads(args.output.read_text()) if args.output.exists() else {}
    result["subprocess"] = dict(
        returncode=code,
        timeout_seconds=args.timeout,
        timed_out=timed_out,
        elapsed_seconds=time.monotonic() - started,
    )
    if code:
        result["status"] = "timeout" if timed_out else "failed"
    save(args.output, result)
    return 1 if code else 0


def child(args):
    result = dict(status="started", arguments=vars(args), parity=[], performance=[])
    save(args.output, result)
    try:
        os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")
        sys.path.insert(0, str(ROOT / "engine" / "src"))
        import mlx.core as mx
        import rec_mlx.kernels.device as device
        from mlx_lm.sample_utils import make_sampler
        from rec_mlx.kernels import build_decoder, megakernel
        from rec_mlx.quantized import QATModel
        from transformers import AutoTokenizer

        mx.set_cache_limit(0)
        mx.clear_cache()
        mx.reset_peak_memory()
        mx.random.seed(42)
        info = mx.device_info()
        if info["device_name"] != "Apple M4 Max":
            raise RuntimeError(f"This isolated experiment requires Apple M4 Max: {info}")
        profile = device.DeviceProfile(
            chip="M4 Max experimental",
            architecture=info["architecture"],
            workers=args.workers,
            mlp_workers=args.mlp_workers,
            head_workers=args.head_workers,
            prefill_chunk_size=args.prefill_chunk_size,
            specialize_mixed_experts=args.specialize_mixed_experts,
        )
        device.profile_for_device = lambda actual_info: profile
        experiment_metadata = apply_experiments(
            args.mlp_workers, args.experimental_intermediate_mlp, args.qkv_strategy
        )
        kernels = []
        factory = megakernel._metal

        def capture(name, **options):
            kernels.append(
                dict(
                    name=name,
                    source_sha256=hashlib.sha256(
                        (options.get("header", "") + options["source"]).encode()
                    ).hexdigest(),
                )
            )
            return factory(name, **options)

        megakernel._metal = capture
        result.update(
            device=info,
            profile=dataclasses.asdict(profile),
            kernels=kernels,
            experiments=experiment_metadata,
            platform=platform.platform(),
            python=sys.version,
            environment={
                name: value
                for name, value in os.environ.items()
                if name.startswith("MLX_") or name in ("METAL_DEVICE_WRAPPER_TYPE",)
            },
            host_state_before=host_state(),
            source_sha256={
                str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((ROOT / "engine" / "src" / "rec_mlx").rglob("*.py"))
            },
            harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            dependencies={
                name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "transformers")
            },
            git_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            config_sha256=hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
            execution=dict(
                optimized=True,
                fallback_allowed=False,
                sampling=args.sampling,
                scalar="FullModel.forward",
                packet={
                    "greedy": "ResidentSampler.greedy_kernel",
                    "unfiltered": "ResidentSampler.kernel",
                    "default": "FilteredResidentSampler.block",
                }[args.sampling],
                native="QATModel.native",
                packet_size=args.packet_size,
                strict_bitwise_parity=True,
                parity_sampling="greedy",
                prefill_excluded=True,
                eos_respected=True,
                synthetic_prompt=True,
            ),
        )
        if args.exhaustive_validation:
            # Its factory transforms are global; execute in a separate fresh process.
            script = ROOT / "engine" / "tests" / "validate_resident.py"
            code = (
                "import sys,runpy;sys.path.insert(0,sys.argv[1]);"
                f"sys.path.insert(0,{str(Path(__file__).resolve().parent)!r});"
                "from benchmark import apply_experiments;"
                f"apply_experiments({args.mlp_workers!r},"
                f"{args.experimental_intermediate_mlp!r},{args.qkv_strategy!r});"
                "from rec_mlx.kernels import device;"
                f"p=device.DeviceProfile(**{dataclasses.asdict(profile)!r});"
                "device.profile_for_device=lambda info:p;"
                "s=sys.argv[2];sys.argv=[s,'--model',sys.argv[3],'--output',sys.argv[4]];"
                "runpy.run_path(s,run_name='__main__')"
            )
            validation_path = args.output.with_suffix(".exhaustive.json")
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    code,
                    str(ROOT / "engine" / "src"),
                    str(script),
                    str(args.model),
                    str(validation_path),
                ],
                check=True,
            )
            result["exhaustive_validation"] = str(validation_path)
        event("load_model", profile=dataclasses.asdict(profile))
        started = time.monotonic()
        model = QATModel(args.model)
        engine, resident = build_decoder(model, optimized=True)
        mx.eval(model.native.parameters())
        mx.set_cache_limit(args.cache_limit_mib * 1024**2)
        result["allocation_cache_limit_bytes"] = args.cache_limit_mib * 1024**2
        result["startup_memory"] = dict(
            active_bytes=mx.get_active_memory(), peak_bytes=mx.get_peak_memory()
        )
        result["load_and_compile_setup_seconds"] = time.monotonic() - started
        result["schedule"] = engine.schedule
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True, fix_mistral_regex=False
        )
        prompt = tokenizer.apply_chat_template(
            [
                dict(
                    role="user",
                    content="Write a complete Python program using openpyxl to "
                    "create an Excel sales report, with sample data, formulas, formatting and "
                    "a bar chart. Explain and test the code.",
                )
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        captured = {}
        greedy_kernel = resident.greedy_kernel

        def capture_greedy(**kwargs):
            out = greedy_kernel(**kwargs)
            captured["logits"], captured["trace"] = out[:2]
            return out

        def compare(label, expected, actual):
            if bool(mx.array_equal(expected, actual).item()):
                return
            delta = mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32))
            largest = float(mx.max(delta).item())
            detail = dict(
                check=label,
                expected_shape=list(expected.shape),
                actual_shape=list(actual.shape),
                different_elements=int(mx.sum(expected != actual).item()),
                max_absolute_difference=largest if largest < float("inf") else str(largest),
                expected_argmax=int(mx.argmax(expected.reshape(-1)).item()),
                actual_argmax=int(mx.argmax(actual.reshape(-1)).item()),
            )
            result["mismatch"] = detail
            raise AssertionError(json.dumps(detail))

        def packet_failure_diagnostics(expected_logits, expected_trace):
            """Inspect only the first packet step, before diverging tokens compound errors."""
            actual_logits, actual_trace = captured["logits"][0], captured["trace"][0]

            def finite(value):
                value = float(value)
                return value if -float("inf") < value < float("inf") else str(value)

            def first_difference(expected, actual):
                mismatch = expected.reshape(-1) != actual.reshape(-1)
                count = int(mx.sum(mismatch).item())
                if not count:
                    return dict(different_elements=0)
                index = int(mx.argmax(mismatch.astype(mx.int32)).item())
                return dict(
                    different_elements=count,
                    first_flat_index=index,
                    expected=finite(expected.reshape(-1)[index].item()),
                    actual=finite(actual.reshape(-1)[index].item()),
                    max_absolute_difference=finite(
                        mx.max(
                            mx.abs(expected.astype(mx.float32) - actual.astype(mx.float32))
                        ).item()
                    ),
                )

            row_counts = mx.sum(expected_trace != actual_trace, axis=1).tolist()
            row_errors = mx.max(
                mx.abs(expected_trace.astype(mx.float32) - actual_trace.astype(mx.float32)), axis=1
            ).tolist()
            first_row = next((index for index, count in enumerate(row_counts) if count), None)
            return dict(
                packet_offset=0,
                trace_index_convention="row0=embedding; rowN=output after transformer layer N-1",
                first_divergent_trace_row=first_row,
                first_divergent_transformer_layer=(first_row - 1 if first_row else None),
                trace=first_difference(expected_trace, actual_trace),
                divergent_trace_rows=[
                    dict(
                        row=index,
                        different_elements=count,
                        max_absolute_difference=finite(row_errors[index]),
                    )
                    for index, count in enumerate(row_counts)
                    if count
                ],
                logits=first_difference(expected_logits, actual_logits),
                expected_logit_argmax=int(mx.argmax(expected_logits).item()),
                actual_logit_argmax=int(mx.argmax(actual_logits).item()),
            )

        def prepare(ids, count):
            session = engine.make_session(max_new_tokens=count + 1)
            logits = session.prefill(ids)
            return session, logits

        def native_cache(session, capacity):
            session._state["kv"].grow(capacity)
            session._state["capacity"] = session._state["kv"].capacity
            caches = session.prompt_cache.layers
            for row, cache in zip(engine.metadata[:49], caches, strict=True):
                if not row[1]:
                    cache.grow(capacity)
            mx.eval([entry.state for entry in caches])
            return caches

        def parity(ids):
            event("parity", context=len(ids), steps=args.parity_steps)
            scalar, a = prepare(ids, args.parity_steps)
            packet, b = prepare(ids, args.parity_steps)
            native, c = prepare(ids, args.parity_steps)
            compare("prefill_scalar_packet", a, b)
            compare("prefill_scalar_native", a, c)
            caches = native_cache(native, len(ids) + args.parity_steps + 1)
            pending = int(mx.argmax(a).item())
            steps, packets = 0, 0
            resident.greedy_kernel = capture_greedy
            try:
                while steps < args.parity_steps and pending not in model.eos_tokens:
                    first = pending
                    count = min(args.packet_size, args.parity_steps - steps)
                    expected, logits, traces = [], [], []
                    for _ in range(count):
                        value, trace = engine.forward(scalar._state, pending)
                        h = model.native.model.embed_tokens(mx.array([[pending]], mx.int32))
                        native_trace = [h.reshape(-1)]
                        for layer, cache in zip(model.native.layers, caches, strict=True):
                            h = layer(h, None, cache)
                            native_trace.append(h.reshape(-1))
                        native_logits = model.native.lm_head(model.native.model.norm(h))
                        compare("scalar_native_layers", mx.stack(native_trace), trace)
                        compare("scalar_native_logits", native_logits.reshape(-1), value)
                        pending = int(mx.argmax(value).item())
                        expected.append(pending)
                        logits.append(value)
                        traces.append(trace)
                        if pending in model.eos_tokens:
                            break
                    actual = resident.block(packet._state, first, count, 43 + steps, temperature=0)
                    if actual != expected:
                        result["mismatch"] = dict(
                            check="packet_tokens", expected=expected, actual=actual, position=steps
                        )
                        try:
                            result["mismatch"]["first_step_diagnostics"] = (
                                packet_failure_diagnostics(logits[0], traces[0])
                            )
                        except Exception as diagnostic_error:
                            result["mismatch"]["diagnostic_error"] = repr(diagnostic_error)
                        raise AssertionError(str(result["mismatch"]))
                    for index in range(len(expected)):
                        compare("packet_scalar_logits", logits[index], captured["logits"][index])
                        compare("packet_scalar_layers", traces[index], captured["trace"][index])
                    for index, (x, y) in enumerate(
                        zip(scalar._state["kv"].banks, packet._state["kv"].banks, strict=True)
                    ):
                        compare(f"packet_scalar_kv_{index}", x, y)
                    if scalar._state["position"] != packet._state["position"]:
                        raise AssertionError("Scalar/packet position mismatch")
                    steps += len(actual)
                    packets += 1
            finally:
                resident.greedy_kernel = greedy_kernel
                captured.clear()
            if steps == 0:
                raise AssertionError("Prompt ended immediately; no decode correctness validated")
            return dict(
                context=len(ids),
                steps=steps,
                packets=packets,
                passed=True,
                checks=[
                    "native_scalar_layers",
                    "native_scalar_logits",
                    "packet_tokens",
                    "packet_scalar_layers",
                    "packet_scalar_logits",
                    "packet_scalar_kv",
                ],
            )

        def measure(ids, variant):
            mx.synchronize()
            prefill_started = time.perf_counter()
            session, logits = prepare(ids, args.steps)
            mx.eval(logits)
            mx.synchronize()
            prefill_seconds = time.perf_counter() - prefill_started
            caches = (
                native_cache(session, len(ids) + args.steps + 1) if variant == "native" else None
            )
            # Grow outside timed region: repeated context-allocation stalls must not
            # masquerade as kernel performance. This is recorded in metadata below.
            if variant != "native":
                engine.ensure_capacity(session._state, args.steps)
                mx.eval(session._state["kv"].banks)
            settings = {
                "greedy": dict(temperature=0, top_p=1, top_k=0),
                "unfiltered": dict(temperature=1, top_p=1, top_k=0),
                "default": dict(temperature=1, top_p=0.95, top_k=20),
            }[args.sampling]
            sampler = make_sampler(
                temp=settings["temperature"], top_p=settings["top_p"], top_k=settings["top_k"]
            )

            def pick(value, index):
                flat = value.reshape(-1).astype(mx.float32)
                if args.sampling == "greedy":
                    return int(mx.argmax(flat).item())
                if args.sampling == "unfiltered":
                    noise = mx.random.gumbel(flat.shape, key=mx.random.key(42 + index))
                    return int(mx.argmax(flat + noise).item())
                return int(sampler(flat - mx.logsumexp(flat)).item())

            mx.random.seed(42)
            pending = pick(logits, 0)
            count, token_ids = 0, []
            mx.synchronize()
            started = time.perf_counter()
            while count < args.steps and pending not in model.eos_tokens:
                if variant == "packet":
                    tokens = resident.block(
                        session._state,
                        pending,
                        min(args.packet_size, args.steps - count),
                        43 + count,
                        **settings,
                    )
                else:
                    value = (
                        model.native(mx.array([[pending]], mx.int32), cache=caches)
                        if variant == "native"
                        else session.forward_token(pending)
                    )
                    tokens = [pick(value, count + 1)]
                count += len(tokens)
                pending = tokens[-1]
                token_ids.extend(tokens)
            mx.synchronize()
            elapsed = time.perf_counter() - started
            if not count:
                raise AssertionError("No decode tokens produced")
            return dict(
                seconds=elapsed,
                tokens=count,
                requested_steps=args.steps,
                completed_requested_steps=count == args.steps,
                tokens_per_second=count / elapsed,
                prefill_seconds=prefill_seconds,
                prefill_tokens_per_second=len(ids) / prefill_seconds,
                stopped_on_eos=pending in model.eos_tokens,
                token_sha256=hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
            )

        result["execution"]["decode_capacity_preallocated"] = True
        result["execution"]["token_count_convention"] = (
            "steps counts timed forward transitions after the first token sampled from prefill; "
            "the first prefill token is excluded from both elapsed time and token count"
        )
        save(args.output, result)
        for context in args.contexts:
            ids = (prompt * ((context + len(prompt) - 1) // len(prompt)))[:context]
            result["parity"].append(parity(ids))
            save(args.output, result)
            mx.clear_cache()
            timings = {variant: [] for variant in args.variants}
            result["in_progress_context"] = dict(context=context, timings=timings)
            # Rotate execution order to reduce systematic thermal/order bias.
            for repeat in range(args.warmups + args.repeats):
                shift = repeat % len(args.variants)
                order = args.variants[shift:] + args.variants[:shift]
                expected_hash = None
                for variant in order:
                    event(
                        "benchmark",
                        context=context,
                        variant=variant,
                        repeat=repeat,
                        warmup=repeat < args.warmups,
                    )
                    sample = measure(ids, variant)
                    if expected_hash is None:
                        expected_hash = sample["token_sha256"]
                    elif sample["token_sha256"] != expected_hash:
                        raise AssertionError("Timed native/scalar/packet generation diverged")
                    if repeat >= args.warmups:
                        timings[variant].append(sample)
                    event("sample", context=context, variant=variant, **sample)
                    save(args.output, result)
            row = dict(context=context, variants={})
            for variant, samples in timings.items():
                speeds = [sample["tokens_per_second"] for sample in samples]
                row["variants"][variant] = dict(
                    samples=samples,
                    median_tokens_per_second=statistics.median(speeds),
                    min_tokens_per_second=min(speeds),
                    max_tokens_per_second=max(speeds),
                    stdev_tokens_per_second=statistics.stdev(speeds) if len(speeds) > 1 else 0,
                )
            result["performance"].append(row)
            save(args.output, result)
            mx.clear_cache()
        result["status"] = "passed"
        result.pop("in_progress_context", None)
        result["host_state_after"] = host_state()
        result["final_memory"] = dict(
            active_bytes=mx.get_active_memory(), peak_bytes=mx.get_peak_memory()
        )
    except Exception as exc:
        result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        event("failed", error=repr(exc))
    save(args.output, result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    args = arguments()
    sys.exit(child(args) if args.child else parent(args))
