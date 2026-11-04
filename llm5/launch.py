"""Validate and run the native MaxText sequence length curriculum."""
import argparse
import glob
import json
import os
from pathlib import Path
import shlex
import sys

import pyarrow.parquet as pq
from llm5.plan import MAXTEXT_VERSION, ROOT, digest, read_json
from llm5.train import run_training_config


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, file in [("curriculum_sha256", "curriculum.json"),
                      ("sources_sha256", "sources.json"), ("model_sha256", "model.json"),
                      ("base_sha256", "maxtext-base.json")]:
        if manifest[key] != digest(ROOT / "configs" / file):
            raise ValueError(f"{file} changed; regenerate plan and, where relevant, packed data")
    if manifest["maxtext_version"] != MAXTEXT_VERSION:
        raise ValueError("Unexpected MaxText revision")
    return manifest


def check_data(stage):
    cfg = read_json(stage["config"])
    tokenizer = Path(cfg["tokenizer_path"])
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tokenizer / "tokenizer.json"))
    if tok.get_vocab_size() != cfg["vocab_size"] or tok.token_to_id("<|pad|>") != 0:
        raise ValueError("Tokenizer vocabulary or PAD ID mismatch")
    train_files = sorted(glob.glob(cfg["grain_train_files"]))
    eval_files = sorted(glob.glob(cfg["grain_eval_files"]))
    if not train_files or not eval_files:
        raise FileNotFoundError(f"Missing packed train/eval data for {stage['name']}")
    meta = json.loads((Path(train_files[0]).parent / "metadata.json").read_text())
    if meta["tokenizer_sha256"] != digest(tokenizer / "tokenizer.json"):
        raise ValueError("Packed data belongs to a different tokenizer")
    for key, file in [("curriculum_sha256", "curriculum.json"), ("sources_sha256", "sources.json")]:
        if meta[key] != digest(ROOT / "configs" / file):
            raise ValueError(f"Packed data predates {file}; prepare data in a new directory")
    if (meta["stage"]["length"], meta["stage"]["sequences"]) != (stage["length"], stage["sequences"]):
        raise ValueError("Packed data length or sequence budget mismatch")
    count = sum(pq.ParquetFile(f).metadata.num_rows for f in train_files)
    if count != stage["sequences"]:
        raise ValueError(f"Expected {stage['sequences']} sequences; found {count}")
    eval_count = sum(pq.ParquetFile(f).metadata.num_rows for f in eval_files)
    if eval_count < cfg["eval_steps"]:
        raise ValueError("Too few eval records")
    for file in (train_files[0], train_files[-1], eval_files[0]):
        table = pq.ParquetFile(file).read_row_group(0, columns=["text"])
        for row in table.column("text").to_pylist():
            if len(row) != stage["length"] or min(row) <= 0 or max(row) >= cfg["vocab_size"]:
                raise ValueError(f"Invalid token length or range in {file}")


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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="runs/plan/manifest.json")
    p.add_argument("--check-data", action="store_true")
    p.add_argument("--execute", action="store_true", help="Start training on the current TPU VM")
    args = p.parse_args()
    manifest = load_manifest(args.manifest)
    if args.execute:
        os.environ.setdefault("JAX_PLATFORMS", "tpu")
        # MaxText initializes distributed JAX before its first device query.
        # Subsequent stages reuse that runtime in this process.
    for s in manifest["stages"]:
        if args.check_data or args.execute:
            check_data(s)
        cfg = read_json(s["config"])
        cmd = command(s["config"])
        print(shlex.join(cmd), flush=True)
        if not args.execute:
            continue
        from etils import epath
        if epath.Path(checkpoint_items(cfg, s["end_step"])).exists():
            print(f"{s['name']} complete; keeping its saved full state", flush=True)
            continue
        if cfg["load_full_state_path"] and not epath.Path(cfg["load_full_state_path"]).exists():
            raise FileNotFoundError(f"Previous stage full checkpoint missing: {cfg['load_full_state_path']}")
        run_training_config(s["config"])
        if not epath.Path(checkpoint_items(cfg, s["end_step"])).exists():
            raise RuntimeError("Training exited without the required final full-state checkpoint")


if __name__ == "__main__":
    main()
