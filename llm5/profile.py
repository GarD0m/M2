"""Capture one short, isolated trace with a CLI-selected configuration."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from llm5 import launch, streaming
from llm5.plan import generate, read_json
from llm5.train import configure_tpu, run_training_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--microbatch", type=int, default=32, help="Global number of sequences")
    parser.add_argument("--ga", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=1, help="Optimizer steps to skip before tracing")
    parser.add_argument("--profile-steps", type=int, default=4, help="Optimizer steps to trace")
    parser.add_argument("--math", type=float, default=0.05, help="Math data fraction")
    parser.add_argument("--code", type=float, default=0.15, help="Code data fraction")
    args = parser.parse_args(argv)
    if args.length < 2 or args.microbatch < 1 or args.ga < 1:
        parser.error("length must be at least 2; microbatch and GA must be positive")
    if args.warmup_steps < 0 or args.profile_steps < 1:
        parser.error("warmup steps must be nonnegative and profile steps must be positive")
    if not (0 <= args.math <= 1 and 0 <= args.code <= 1 - args.math):
        parser.error("math and code fractions must be nonnegative and sum to at most 1")
    configure_tpu()
    streaming.configure(train_weights=[1 - args.math - args.code, args.math, args.code])
    device_count = launch.check_runtime()
    if args.microbatch % device_count:
        parser.error("global microbatch must be divisible by the TPU device count")
    # A fresh directory prevents MaxText from restoring or overwriting a training run.
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    steps = args.warmup_steps + args.profile_steps + 1
    budget = (args.length - 1) * args.microbatch * args.ga * max(100, steps + 1)
    curriculum = {"seed": 42, "warmup_steps": 64, "target_tokens": budget,
                  "stages": [{"name": "test", "length": args.length, "budget": budget,
                              "microbatch": args.microbatch, "accumulation": args.ga,
                              "math": args.math, "code": args.code}]}
    with TemporaryDirectory(prefix="llm5-profile-") as temporary:
        manifest = generate(temporary, "models/tokenizer", "runs/profiles",
                            run_id, device_count=device_count, curriculum_config=curriculum)
        stage = manifest["stages"][0]
        config_path = Path(stage["config"])
        cfg = read_json(config_path)
        cfg.update(steps=steps, profiler="xplane", profile_cleanly=True,
                   skip_first_n_steps_for_profiler=args.warmup_steps, profiler_steps=args.profile_steps,
                   enable_checkpointing=False, save_checkpoint_on_completion=False,
                   eval_interval=-1, load_full_state_path="")
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        launch.check_data(stage, cfg, validate_tokenizer=True)
        print(f"Profiling: length={stage['length']}, "
              f"global microbatch={stage['microbatch']}, GA={stage['accumulation']}, "
              f"warmup={args.warmup_steps}, profile steps={args.profile_steps}", flush=True)
        print(f"TensorBoard logdir: {cfg['base_output_directory']}/{cfg['run_name']}/tensorboard", flush=True)
        run_training_config(config_path)


if __name__ == "__main__":
    main()
