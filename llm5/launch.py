"""Validate and run the native MaxText sequence length curriculum."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

from llm5.plan import MAXTEXT_VERSION, ROOT, digest, read_json
from llm5.streaming import load_tokenizer
from llm5.train import run_training_config


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, file in [("curriculum_sha256", "curriculum.json"),
                      ("sources_sha256", "sources.json"), ("model_sha256", "model.json"),
                      ("base_sha256", "maxtext-base.json")]:
        if manifest[key] != digest(ROOT / "configs" / file):
            raise ValueError(f"{file} changed; regenerate the plan")
    if manifest["maxtext_version"] != MAXTEXT_VERSION:
        raise ValueError("Unexpected MaxText revision")
    return manifest


def check_data(stage, cfg, validate_tokenizer):
    if cfg["max_target_length"] != stage["length"]:
        raise ValueError("Streaming sequence length differs from the plan")
    if validate_tokenizer:
        load_tokenizer(cfg["tokenizer_path"], cfg["vocab_size"])


def check_runtime():
    from importlib.metadata import version
    import jax
    if version("maxtext") != MAXTEXT_VERSION:
        raise RuntimeError(f"Install maxtext=={MAXTEXT_VERSION} using requirements.txt")
    devices = jax.devices()
    if not devices or any(d.platform != "tpu" for d in devices):
        raise RuntimeError("Select a TPU runtime in Colab")
    return len(devices)


def checkpoint_items(cfg, step):
    return f"{cfg['base_output_directory'].rstrip('/')}/{cfg['run_name']}/checkpoints/{step}/items"


def command(config, overrides=()):
    return [sys.executable, "-m", "llm5.train", str(config), *overrides]


def main(argv=None, checkpoint_hub=None, resume=None, profile=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="runs/plan/manifest.json")
    p.add_argument("--check-data", action="store_true")
    p.add_argument("--execute", action="store_true", help="Start training on the current TPU VM")
    p.add_argument("--hub-repo", default="tim123xyz/Test")
    args = p.parse_args(argv)
    manifest = load_manifest(args.manifest)
    if args.execute and checkpoint_hub is None:
        from llm5.hub import CheckpointHub
        with CheckpointHub(args.hub_repo, manifest) as hub:
            return main(argv, checkpoint_hub=hub, resume=hub.restore_latest(), profile=profile)
    if args.execute:
        os.environ.setdefault("JAX_PLATFORMS", "tpu")
        # MaxText initializes distributed JAX before its first device query.
        # Subsequent stages reuse that runtime in this process.
    for index, s in enumerate(manifest["stages"]):
        if resume is not None and index < resume["index"]:
            print(f"{s['name']} already completed before restored checkpoint", flush=True)
            continue
        cfg = read_json(s["config"])
        if args.check_data or args.execute:
            check_data(s, cfg, validate_tokenizer=args.check_data)
        resuming_stage = resume is not None and resume["index"] == index
        overrides = ()
        if profile is not None:
            first_step = resume["step"] if resuming_stage else s["start_step"]
            skip = profile["skip_first_n_steps_for_profiler"]
            if first_step + skip < s["end_step"]:
                overrides = ("profiler=xplane", "profile_cleanly=true",
                             f"skip_first_n_steps_for_profiler={skip}",
                             f"profiler_steps={profile['profiler_steps']}")
            elif first_step < s["end_step"]:
                print(f"{s['name']}: too few steps left after profiler warmup; skipping trace", flush=True)
        cmd = command(s["config"], overrides)
        print(shlex.join(cmd), flush=True)
        if not args.execute:
            continue
        from etils import epath
        if epath.Path(checkpoint_items(cfg, s["end_step"])).exists():
            print(f"{s['name']} complete; keeping its saved full state", flush=True)
            continue
        if resuming_stage:
            # Native auto-restore loads this stage's items AND streaming cursor.
            # A fresh runtime need not download the preceding stage as well.
            cfg["load_full_state_path"] = ""
            Path(s["config"]).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        if cfg["load_full_state_path"] and not epath.Path(cfg["load_full_state_path"]).exists():
            raise FileNotFoundError(f"Previous stage full checkpoint missing: {cfg['load_full_state_path']}")
        run_training_config(s["config"], start_step=s["start_step"], checkpoint_hub=checkpoint_hub,
                            overrides=overrides)
        checkpoint_hub.sync_logs()
        checkpoint_hub.wait()
        if not epath.Path(checkpoint_items(cfg, s["end_step"])).exists():
            raise RuntimeError("Training exited without the required final full-state checkpoint")


if __name__ == "__main__":
    main()
