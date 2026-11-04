"""Stream data and train directly from the user's curriculum.json."""
import argparse
from pathlib import Path
from llm5 import streaming, launch
from llm5.plan import generate
from llm5.hub import CheckpointHub


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-dir", default="runs/plan-512k")
    p.add_argument("--tokenizer", default="models/tokenizer")
    p.add_argument("--output-root", default="runs/checkpoints")
    p.add_argument("--run-prefix", default="m2-bf16-512k-v1")
    p.add_argument("--tokenizer-batch-size", type=int, default=256)
    p.add_argument("--tokenizer-threads", type=int)
    p.add_argument("--hub-repo", default="tim123xyz/Test")
    args = p.parse_args(argv)
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
                    checkpoint_hub=hub, resume=resume)


if __name__ == "__main__":
    main()
