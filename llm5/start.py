"""Stream data and train directly from the user's curriculum.json."""
import argparse
from pathlib import Path
from llm5 import streaming, launch
from llm5.plan import generate, read_json
from llm5.hub import CheckpointHub
from llm5.train import configure_tpu


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-dir", default="runs/plan-512k")
    p.add_argument("--tokenizer", default="models/tokenizer")
    p.add_argument("--output-root", default="runs/checkpoints")
    p.add_argument("--run-prefix", default="m2-deepseek4-1pass-16e-512k-v1")
    p.add_argument("--tokenizer-batch-size", type=int, default=256)
    p.add_argument("--tokenizer-threads", type=int)
    p.add_argument("--hub-repo", default="tim123xyz/Test")
    p.add_argument("--profile-config", type=Path, help="Enable JAX tracing using this JSON file")
    args = p.parse_args(argv)
    profile = None
    if args.profile_config is not None:
        try:
            profile = read_json(args.profile_config)
        except (OSError, ValueError) as exc:
            p.error(f"--profile-config: {exc}")
        minimums = {"skip_first_n_steps_for_profiler": 0, "profiler_steps": 1}
        if not isinstance(profile, dict) or profile.keys() != minimums.keys():
            p.error(f"--profile-config must contain exactly: {', '.join(minimums)}")
        for key, minimum in minimums.items():
            if type(profile[key]) is not int or profile[key] < minimum:
                p.error(f"--profile-config: {key} must be an integer >= {minimum}")
    configure_tpu()
    streaming.configure(args.tokenizer_batch_size, args.tokenizer_threads)
    device_count = launch.check_runtime()
    manifest = generate(args.plan_dir, args.tokenizer,
                        args.output_root, args.run_prefix, device_count=device_count)
    for stage in manifest["stages"]:
        print(f"{stage['name']}: global microbatch={stage['microbatch']}, "
              f"GA={stage['accumulation']}, steps={stage['num_steps']}", flush=True)
    with CheckpointHub(args.hub_repo, manifest) as hub:
        resume = hub.restore_latest()
        launch.main(["--manifest", str(Path(args.plan_dir) / "manifest.json"), "--execute"],
                    checkpoint_hub=hub, resume=resume, profile=profile)


if __name__ == "__main__":
    main()
