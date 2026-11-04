"""Probe microbatch sizes on a single TPU; preserve each stage's effective batch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import math
from statistics import median

from llm5.launch import check_runtime, command, load_manifest
from llm5.plan import generate


def prepare_plan(manifest_path, device_count, microbatches=None):
    return generate(Path(manifest_path).parent, "data/packed-512k", "models/tokenizer",
                    "runs/checkpoints", "m2-bf16-512k-v1", microbatches, device_count)


def is_oom(log):
    text = log.lower()
    return any(s in text for s in ("resource_exhausted", "out of memory", "ran out of memory",
                                   "failed to allocate", "cannot allocate memory"))


def candidates(effective_batch, length=1024, max_micro_tokens=131072, device_count=1):
    return [2**i for i in range(effective_batch.bit_length())
            if effective_batch % 2**i == 0 and 2**i % device_count == 0
            and 2**i * length <= max_micro_tokens*device_count]


def steady_step_seconds(path, skip_steps):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    times = [float(r["perf/step_time_seconds"]) for r in rows
             if r.get("step", -1) >= skip_steps and "perf/step_time_seconds" in r]
    if len(times) < 3 or any(not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError(f"Insufficient valid steady-state timing metrics: {path}")
    return median(times)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="runs/plan/manifest.json")
    p.add_argument("--out", default="runs/microbatches.json")
    p.add_argument("--log-dir", default="runs/probes")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--skip-steps", type=int, default=4)
    p.add_argument("--max-micro-tokens", type=int, default=131072)
    p.add_argument("--prepare-plan", action="store_true",
                   help="Generate the default 512K plan before tuning and apply selected microbatches afterwards")
    args = p.parse_args()
    if args.skip_steps < 1 or args.steps < args.skip_steps + 3:
        p.error("Reserve at least three measured steps after warmup")
    os.environ.setdefault("JAX_PLATFORMS", "tpu")
    device_count = check_runtime()
    if args.prepare_plan:
        prepare_plan(args.manifest, device_count)
    manifest = load_manifest(args.manifest)
    if device_count != manifest["device_count"]:
        raise RuntimeError(f"Regenerate plan with --device-count {device_count}")
    output, log_dir = Path(args.out), Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    chosen = {}
    report = {}
    for s in manifest["stages"]:
        results = []
        for micro in candidates(s["effective_batch"], s["length"], args.max_micro_tokens, device_count):
            probe_id = f"{s['name']}-{micro}-{time.time_ns()}"
            metrics_path = log_dir / f"{probe_id}.jsonl"
            overrides = [f"per_device_batch_size={float(micro/device_count)}",
                         f"gradient_accumulation_steps={s['effective_batch'] // micro}",
                         f"steps={args.steps}", "dataset_type=synthetic", "eval_interval=-1",
                         "load_full_state_path=", "enable_checkpointing=false",
                         "save_checkpoint_on_start=false", "save_checkpoint_on_completion=false",
                         f"base_output_directory={log_dir.as_posix()}",
                         f"run_name=probe-{probe_id}", f"metrics_file={metrics_path.as_posix()}",
                         "max_inflight_computations=1", "profile_cleanly=true",
                         "async_checkpointing=false", "log_period=1"]
            log_path = log_dir / f"{s['name']}-micro{micro}.log"
            print(f"Probing {s['name']} micro={micro} GA={s['effective_batch'] // micro}", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(command(s["config"], overrides), stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                seconds = steady_step_seconds(metrics_path, args.skip_steps)
                results.append(dict(microbatch=micro, accumulation=s["effective_batch"]//micro,
                                    median_step_seconds=seconds, loss_tokens_per_second=s["tokens_per_step"]/seconds))
                print(f"Median step: {seconds:.4f}s; {s['tokens_per_step']/seconds:,.0f} loss tokens/s", flush=True)
            elif not is_oom(log_path.read_text(encoding="utf-8", errors="replace")):
                raise RuntimeError(f"Probe failed for a reason other than OOM; inspect {log_path}")
        if not results:
            raise RuntimeError(f"No fitting microbatch for {s['name']}; inspect {log_dir}.")
        best = min(results, key=lambda r: r["median_step_seconds"])
        chosen[s["name"]] = best["microbatch"]
        report[s["name"]] = dict(selected=best, measurements=results)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.prepare_plan:
        prepare_plan(args.manifest, device_count, chosen)
        print(f"Saved {output}; applied selected microbatches to {Path(args.manifest).parent}")
    else:
        print(f"Saved {output}; regenerate the plan with --microbatches {output}")


if __name__ == "__main__":
    main()
