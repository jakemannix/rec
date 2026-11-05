#!/usr/bin/env python3
"""Measure production Engine generation with Flint grammar; excludes HTTP/network.

Run as its own process under an external timeout. No production files are changed.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPT = (
    "Write a complete, well-documented Python program using openpyxl to create an Excel "
    "sales report. Include realistic sample data for twelve months and five product lines, "
    "formulas for revenue, costs and margins, styled headers, currency formats, conditional "
    "formatting, totals, a bar chart, a summary worksheet, validation and unit tests. "
    "Provide the full implementation and explain how to run it. Do not abbreviate the code."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--tool-check", action="store_true")
    args = parser.parse_args()
    if min(args.repeats, args.max_tokens) < 1:
        parser.error("repeats and max-tokens must be positive")
    args.model, args.output = args.model.resolve(), args.output.resolve()
    args.manifest = (args.manifest or args.model / "flint-int4.json").resolve()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status="started", scope="Production Engine path; excludes HTTP/network and "
                  "scheduler response conversion", samples=[], grammar_applied=True,
                  grammar_api="engine.constraints.tools(None)", cold_prefill=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, default=str) + "\n")
        temporary.replace(args.output)

    save()
    try:
        os.environ.setdefault("MLX_METAL_FAST_SYNCH", "1")
        sys.path.insert(0, str(ROOT / "engine" / "src"))
        import mlx.core as mx
        from rec_mlx.kernels import device, megakernel

        info = mx.device_info()
        if info["device_name"] != "Apple M4 Max":
            raise RuntimeError(f"Expected Apple M4 Max, received {info}")
        profile = device.DeviceProfile(chip="M4 Max experimental", architecture=info["architecture"],
                                       workers=64, mlp_workers=64, head_workers=64,
                                       prefill_chunk_size=2048, specialize_mixed_experts=False)
        device.profile_for_device = lambda actual_info: profile
        kernels, factory = [], megakernel._metal

        def capture(name, **options):
            kernels.append(dict(name=name, source_sha256=hashlib.sha256(
                (options.get("header", "") + options["source"]).encode()).hexdigest()))
            return factory(name, **options)

        megakernel._metal = capture
        # Import only after the profile override: Engine binds this function at import time.
        from rec_mlx.engine import Engine, Sampling

        engine = Engine(args.model, args.manifest, context_limit=262144)
        sampling = Sampling(temperature=1, top_p=0.95, top_k=20, max_tokens=args.max_tokens)
        constraint = engine.constraints.tools(None)
        ids = engine.prompt([dict(role="user", content=PROMPT)])
        result.update(device=info, profile=dataclasses.asdict(profile), kernels=kernels,
                      sampling=dataclasses.asdict(sampling), startup=engine.startup,
                      config_sha256=hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
                      manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                      harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
                      prompt_tokens=len(ids), mlx_environment={key: value for key, value in
                      os.environ.items() if key.startswith("MLX_")})
        save()

        def run(prompt_ids, grammar):
            engine._prompt_cache = None
            engine._prefix_preparation = None
            mx.synchronize()
            started = time.perf_counter()
            terminal, fragments = None, []
            for event in engine.generate_tokens(prompt_ids, sampling, grammar=grammar):
                if "metrics" in event:
                    terminal = event
                else:
                    fragments.append(event.get("text", ""))
            mx.synchronize()
            if terminal is None:
                raise RuntimeError("Engine returned no terminal metrics")
            metrics = terminal["metrics"]
            if metrics["decode_strategy"] != "resident_sampling":
                raise AssertionError("Expected optimized resident sampling")
            if metrics["cached_prompt_tokens"] != 0:
                raise AssertionError("Expected cold prefill")
            if metrics.get("grammar_accepted_tokens") != metrics["generated_tokens"]:
                raise AssertionError("Grammar did not accept every generated token")
            return dict(metrics=metrics, finish_reason=terminal["finish_reason"],
                        wall_seconds=time.perf_counter() - started,
                        completed_token_budget=metrics["generated_tokens"] == args.max_tokens,
                        text_sha256=hashlib.sha256("".join(fragments).encode()).hexdigest())

        for repeat in range(args.repeats + 1):
            sample = run(ids, constraint)
            sample.update(repeat=repeat, warmup=repeat == 0)
            result["samples"].append(sample)
            save()
            print(json.dumps(dict(repeat=repeat, warmup=repeat == 0, **sample["metrics"])), flush=True)
        speeds = [sample["metrics"]["decode_tokens_per_second"] for sample in result["samples"][1:]]
        result["summary"] = dict(median_decode_tokens_per_second=statistics.median(speeds),
                                 min_decode_tokens_per_second=min(speeds),
                                 max_decode_tokens_per_second=max(speeds))
        if args.tool_check:
            from rec_mlx.chat import parse_tool_call

            tools = [dict(type="function", function=dict(name="add", description="Add two integers",
                     parameters=dict(type="object", properties={"a": {"type": "integer"},
                     "b": {"type": "integer"}}, required=["a", "b"], additionalProperties=False)))]
            prompt_ids = engine.prompt([dict(role="user", content=
                "Call the add tool with a=2 and b=3. Do not calculate the result yourself.")], tools=tools)
            tool_result = run(prompt_ids, engine.constraints.tools(tools))
            raw = engine.tokenizer.decode(tool_result["metrics"]["token_ids"], skip_special_tokens=False)
            calls = [parse_tool_call(body, tools) for body in re.findall(
                r"<tool_call>(.*?)</tool_call>", raw, re.S)]
            expected = dict(name="add", arguments={"a": 2, "b": 3})
            result["tool_check"] = dict(result=tool_result, calls=calls, passed=expected in calls)
            if expected not in calls:
                raise AssertionError("Expected add(a=2,b=3) tool call was not produced")
        result["status"] = "passed"
    except Exception as exc:
        result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        print(json.dumps(dict(status="failed", error=repr(exc))), flush=True)
    save()
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
