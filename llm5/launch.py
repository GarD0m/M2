"""Validate and run the native MaxText sequence length curriculum."""
import argparse
import glob
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pyarrow.parquet as pq
from llm5.plan import MAXTEXT_COMMIT, ROOT, digest, read_yaml


def load_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, file in [("curriculum_sha256", "curriculum.yml"),
                      ("sources_sha256", "sources.yml"), ("model_sha256", "model.yml")]:
        if manifest[key] != digest(ROOT / "configs" / file):
            raise ValueError(f"{file} changed; regenerate plan and, where relevant, packed data")
    if manifest["maxtext_commit"] != MAXTEXT_COMMIT:
        raise ValueError("Unexpected MaxText revision")
    return manifest


def check_data(stage):
    cfg = read_yaml(stage["config"])
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
    for key, file in [("curriculum_sha256", "curriculum.yml"), ("sources_sha256", "sources.yml")]:
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
    import maxtext
    import jax
    package = Path(maxtext.__file__).resolve()
    checkout = next((p for p in package.parents if (p / ".git").exists()), None)
    if checkout is None:
        raise RuntimeError("Install the pinned editable MaxText checkout using scripts/setup_tpu.sh")
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"], text=True)
    if commit != MAXTEXT_COMMIT or dirty.strip():
        raise RuntimeError("MaxText checkout differs from the tested source revision")
    devices = jax.devices()
    if len(devices) != 1 or jax.process_count() != 1 or devices[0].platform != "tpu":
        raise RuntimeError(f"Expected one local TPU device; found {devices}")
    if "v6" not in devices[0].device_kind.lower():
        raise RuntimeError(f"Expected TPU v6e, found {devices[0].device_kind}")


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
        check_runtime()
    for s in manifest["stages"]:
        if args.check_data or args.execute:
            check_data(s)
        cfg = read_yaml(s["config"])
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
        subprocess.run(cmd, check=True)
        if not epath.Path(checkpoint_items(cfg, s["end_step"])).exists():
            raise RuntimeError("Training exited without the required final full-state checkpoint")


if __name__ == "__main__":
    main()
