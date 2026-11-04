"""Prepare data and train directly from the user's curriculum.json."""
import argparse
from pathlib import Path
from llm5 import data, launch
from llm5.plan import generate


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-dir", default="runs/plan-512k")
    p.add_argument("--data-root", default="data/packed-512k")
    p.add_argument("--tokenizer", default="models/tokenizer")
    p.add_argument("--output-root", default="runs/checkpoints")
    p.add_argument("--run-prefix", default="m2-bf16-512k-v1")
    args = p.parse_args(argv)
    device_count = launch.check_runtime()
    manifest = generate(args.plan_dir, args.data_root, args.tokenizer,
                        args.output_root, args.run_prefix, device_count=device_count)
    for stage in manifest["stages"]:
        print(f"{stage['name']}: global microbatch={stage['microbatch']}, "
              f"GA={stage['accumulation']}, steps={stage['num_steps']}", flush=True)
    if not all((Path(args.data_root) / s["name"] / "metadata.json").exists()
               for s in manifest["stages"]):
        print("Preparing actual training data from UltraData L3", flush=True)
        data.prepare(args.data_root, args.tokenizer)
    launch.main(["--manifest", str(Path(args.plan_dir) / "manifest.json"), "--execute"])


if __name__ == "__main__":
    main()
