"""Generate cumulative-step MaxText configs from a length curriculum."""
import argparse
import hashlib
import json
from pathlib import Path
import yaml

MAXTEXT_COMMIT = "ce608b918c4fdfbfdb62f5a7976d2e15e4fdd720"
ROOT = Path(__file__).resolve().parents[1]


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curriculum(config, microbatches=None, device_count=1):
    if device_count < 1:
        raise ValueError("device_count must be positive")
    if sum(s["budget"] for s in config["stages"]) != config["target_tokens"]:
        raise ValueError("Stage budgets must sum to target_tokens")
    stages, end = [], 0
    for spec in config["stages"]:
        s = dict(spec)
        effective = s["microbatch"] * s["accumulation"]
        s["microbatch"] = (microbatches or {}).get(s["name"], max(s["microbatch"], device_count))
        if s["microbatch"] < device_count or effective % s["microbatch"] or s["microbatch"] % device_count:
            raise ValueError("Global microbatch must divide effective batch and be divisible by device_count")
        s["device_count"] = device_count
        s["per_device_microbatch"] = s["microbatch"] // device_count
        s["accumulation"] = effective // s["microbatch"]
        if not (0 <= s["math"] <= 1 and 0 <= s["code"] <= 1 - s["math"]):
            raise ValueError("Invalid domain mixture")
        s["general"] = 1 - s["math"] - s["code"]
        s["effective_batch"] = effective
        # Native Grain ShiftData masks the final target in each length-L record.
        s["tokens_per_step"] = effective * (s["length"] - 1)
        s["num_steps"] = max(1, round(s["budget"] / s["tokens_per_step"]))
        s["start_step"] = end
        end += s["num_steps"]
        s["end_step"] = end
        s["sequences"] = s["num_steps"] * effective
        s["loss_tokens"] = s["num_steps"] * s["tokens_per_step"]
        stages.append(s)
    if not 0 < config["warmup_steps"] < stages[0]["num_steps"]:
        raise ValueError("Warmup must finish in the first stage")
    return stages


def parameter_count(model):
    d, ff, v, n = (model[k] for k in
                    ("base_emb_dim", "base_mlp_dim", "vocab_size", "base_num_decoder_layers"))
    q = model["base_num_query_heads"] * model["head_dim"]
    kv = model["base_num_kv_heads"] * model["head_dim"]
    # QK RMS norms are per head_dim, plus two per-layer RMS norms and final norm.
    return v*d + n*(2*d*q + 2*d*kv + 3*d*ff + 2*d + 2*model["head_dim"]) + d


def memory_components(model, stage):
    """Named tensor sizes only; excludes optimizer workspace, remat and XLA temporaries."""
    params = parameter_count(model)
    tokens = stage["per_device_microbatch"] * stage["length"]
    q = model["base_num_query_heads"] * model["head_dim"]
    kv = model["base_num_kv_heads"] * model["head_dim"]
    gib = 1024**3
    return {
        "bf16_weights_gib": params*2/gib,
        "fp32_gradient_buffer_gib": params*4/gib,
        "saved_qkv_gib": model["base_num_decoder_layers"]*tokens*(q+2*kv)*2/gib,
        "layer_inputs_gib": model["base_num_decoder_layers"]*tokens*model["base_emb_dim"]*2/gib,
        "one_fp32_logits_tile_gib": tokens*model["vocab_size"]*4/model["num_vocab_tiling"]/gib,
    }


def generate(out, data_root, tokenizer, output_root, run_prefix, microbatches=None, device_count=1):
    raw = read_yaml(ROOT / "configs/curriculum.yml")
    stages = curriculum(raw, microbatches, device_count)
    model = read_yaml(ROOT / "configs/model.yml")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_root, tokenizer = Path(data_root).resolve(), Path(tokenizer).resolve()
    if not output_root.startswith("gs://"):
        output_root = Path(output_root).resolve().as_posix()
    total_steps = stages[-1]["end_step"]
    for i, s in enumerate(stages):
        s["memory_components"] = memory_components(model, s)
        cfg = dict(model)
        cfg["base_config"] = "base.yml"  # resolved in installed MaxText configs
        cfg.update(
            run_name=f"{run_prefix}-{s['name']}", base_output_directory=output_root,
            max_target_length=s["length"], per_device_batch_size=float(s["per_device_microbatch"]),
            ici_data_parallelism=device_count,
            gradient_accumulation_steps=s["accumulation"], steps=s["end_step"],
            learning_rate_schedule_steps=total_steps,
            # Upstream casts fraction*total_steps to int; guard float roundoff.
            warmup_steps_fraction=(raw["warmup_steps"] + 0.01) / total_steps,
            tokenizer_path=str(tokenizer), data_shuffle_seed=raw["seed"], init_weights_seed=raw["seed"],
            grain_train_files=str(data_root / s["name"] / "train-*.parquet"),
            grain_eval_files=str(data_root / s["name"] / "eval-*.parquet"),
            load_full_state_path="",
        )
        if i:
            prev = stages[i-1]
            # Explicit full-state restore loads items, not the previous length's iterator.
            cfg["load_full_state_path"] = (
                f"{output_root.rstrip('/')}/{run_prefix}-{prev['name']}/checkpoints/{prev['end_step']}/items"
            )
        path = out / f"{s['name']}.yml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        s["config"] = str(path)
    manifest = dict(maxtext_commit=MAXTEXT_COMMIT, stages=stages, device_count=device_count,
                    warmup_steps=raw["warmup_steps"], parameters=parameter_count(model),
                    approximate_loss_tokens=sum(s["loss_tokens"] for s in stages),
                    curriculum_sha256=digest(ROOT / "configs/curriculum.yml"),
                    sources_sha256=digest(ROOT / "configs/sources.yml"),
                    model_sha256=digest(ROOT / "configs/model.yml"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/plan")
    p.add_argument("--data-root", default="data/packed")
    p.add_argument("--tokenizer", default="artifacts/tokenizer")
    p.add_argument("--output-root", default="runs/checkpoints")
    p.add_argument("--run-prefix", default="llm5")
    p.add_argument("--microbatches", help="JSON file written by llm5.tune")
    p.add_argument("--device-count", type=int, default=1)
    args = p.parse_args()
    values = vars(args)
    values["microbatches"] = json.loads(Path(args.microbatches).read_text()) if args.microbatches else None
    m = generate(**values)
    print(f"Parameters: {m['parameters']:,}; loss tokens: ~{m['approximate_loss_tokens']:,}")
    for s in m["stages"]:
        print(f"{s['name']:>3}  micro={s['microbatch']} GA={s['accumulation']} "
              f"steps={s['num_steps']:,} end={s['end_step']:,} math={s['math']:.0%} code={s['code']:.0%}")


if __name__ == "__main__":
    main()
