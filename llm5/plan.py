"""Generate cumulative-step MaxText configs from a length curriculum."""
import argparse
import hashlib
import json
import math
from pathlib import Path

MAXTEXT_VERSION = "0.2.5"
ROOT = Path(__file__).resolve().parents[1]
CSA_HEAD_CHUNK_BUDGET = 128 * 1024**2
LOGITS_TILE_ALIGNMENT = 128


def aligned_logits_tiles(tokens, limit):
    """Choose the largest tile count within the limit that divides aligned tokens."""
    if not tokens or tokens % LOGITS_TILE_ALIGNMENT:
        return math.gcd(tokens, limit)
    blocks = tokens // LOGITS_TILE_ALIGNMENT
    return next(tiles for tiles in range(min(limit, blocks), 0, -1) if blocks % tiles == 0)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curriculum(config, device_count=1):
    if device_count < 1:
        raise ValueError("device_count must be positive")
    if sum(s["budget"] for s in config["stages"]) != config["target_tokens"]:
        raise ValueError("Stage budgets must sum to target_tokens")
    stages, end = [], 0
    for spec in config["stages"]:
        s = dict(spec)
        effective = s["microbatch"] * s["accumulation"]
        if s["microbatch"] < device_count or s["microbatch"] % device_count:
            raise ValueError("Global microbatch must be positive and divisible by device_count")
        if s["accumulation"] < 1:
            raise ValueError("Gradient accumulation must be positive")
        s["device_count"] = device_count
        s["per_device_microbatch"] = s["microbatch"] // device_count
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
    if model["decoder_block"] == "deepseek4":
        # Compressed attention, indexers, MoE and mHC are counted by MaxText
        # from the actual parameter tree at initialization.
        return None
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
    tiles = aligned_logits_tiles(tokens, model["num_vocab_tiling"])
    q = model["base_num_query_heads"] * model["head_dim"]
    kv = model["base_num_kv_heads"] * model["head_dim"]
    gib = 1024**3
    if params is None:
        return {"one_fp32_logits_tile_gib": tokens*model["vocab_size"]*4/tiles/gib}
    return {
        "bf16_weights_gib": params*2/gib,
        "fp32_gradient_buffer_gib": params*4/gib,
        "saved_qkv_gib": model["base_num_decoder_layers"]*tokens*(q+2*kv)*2/gib,
        "layer_inputs_gib": model["base_num_decoder_layers"]*tokens*model["base_emb_dim"]*2/gib,
        "one_fp32_logits_tile_gib": tokens*model["vocab_size"]*4/tiles/gib,
    }


def generate(out, tokenizer, output_root, run_prefix, device_count=1, curriculum_config=None):
    raw = read_json(ROOT / "configs/curriculum.json") if curriculum_config is None else curriculum_config
    stages = curriculum(raw, device_count)
    model = read_json(ROOT / "configs/model.json")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = Path(tokenizer).resolve()
    if not output_root.startswith("gs://"):
        output_root = Path(output_root).resolve().as_posix()
    total_steps = stages[-1]["end_step"]
    for i, s in enumerate(stages):
        cfg = dict(
            use_tokamax_gmm=True,
            use_gmm_v2=True,
            use_gmm_v2_heuristic_tiling=True,
            moe_use_direct_token_gather=True,
        )
        cfg.update(model)
        if model["decoder_block"] == "deepseek4":
            length = s["length"]
            batch = max(s["per_device_microbatch"], math.ceil(model["eval_per_device_batch_size"]))
            csa_fp32_bytes_per_head = batch * length * (length // 4) * 4
            head_limit = max(1, CSA_HEAD_CHUNK_BUDGET // csa_fp32_bytes_per_head)
            head_divisor = math.gcd(model["base_num_query_heads"], model["indexer_n_heads"])
            head_chunk = max(
                heads for heads in range(1, min(head_limit, head_divisor) + 1)
                if head_divisor % heads == 0
            )
            cfg.update(
                use_csa_streamindex_kernel=True,
                csa_qk_head_chunk_size=head_chunk,
                remat_policy="custom",
                decoder_layer_input="remat",
                query_proj="device",
                key_proj="device",
                value_proj="device",
                kv_proj="device",
                qkv_proj="device",
                moe_routing_maps="device",
                router_topk_matmul_vjp=True,
                sinkhorn_iterations=10,
                optimizer_memory_host_offload=False,
                parameter_memory_host_offload=False,
                indexer_topk=32,
                # Keep the loss in FP32, but store the logits projection in BF16.
                logits_dot_in_fp32=False,
                num_vocab_tiling=16,
            )
        s["memory_components"] = memory_components(cfg, s)
        cfg["base_config"] = str(ROOT / "configs/maxtext-base.json")
        cfg.update(
            run_name=f"{run_prefix}-{s['name']}", base_output_directory=output_root,
            max_target_length=s["length"], per_device_batch_size=float(s["per_device_microbatch"]),
            ici_data_parallelism=device_count,
            gradient_accumulation_steps=s["accumulation"], steps=s["end_step"],
            learning_rate_schedule_steps=total_steps,
            # Upstream casts fraction*total_steps to int; guard float roundoff.
            warmup_steps_fraction=(raw["warmup_steps"] + 0.01) / total_steps,
            tokenizer_path=str(tokenizer), data_shuffle_seed=raw["seed"], init_weights_seed=raw["seed"],
            load_full_state_path="",
        )
        if i:
            prev = stages[i-1]
            # Explicit full-state restore loads items, not the previous length's iterator.
            cfg["load_full_state_path"] = (
                f"{output_root.rstrip('/')}/{run_prefix}-{prev['name']}/checkpoints/{prev['end_step']}/items"
            )
        path = out / f"{s['name']}.json"
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        s["config"] = str(path)
    manifest = dict(maxtext_version=MAXTEXT_VERSION, stages=stages, device_count=device_count, input_pipeline="hf-stream-v1",
                    warmup_steps=raw["warmup_steps"], parameters=parameter_count(model),
                    approximate_loss_tokens=sum(s["loss_tokens"] for s in stages),
                    curriculum_sha256=(digest(ROOT / "configs/curriculum.json") if curriculum_config is None
                                       else hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()),
                    sources_sha256=digest(ROOT / "configs/sources.json"),
                    model_sha256=digest(ROOT / "configs/model.json"),
                    base_sha256=digest(ROOT / "configs/maxtext-base.json"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/plan")
    p.add_argument("--tokenizer", default="models/tokenizer")
    p.add_argument("--output-root", default="runs/checkpoints")
    p.add_argument("--run-prefix", default="llm5")
    p.add_argument("--device-count", type=int, default=1)
    args = p.parse_args()
    values = vars(args)
    m = generate(**values)
    count = f"{m['parameters']:,}" if m["parameters"] is not None else "counted at model initialization"
    print(f"Parameters: {count}; loss tokens: ~{m['approximate_loss_tokens']:,}")
    for s in m["stages"]:
        print(f"{s['name']:>3}  micro={s['microbatch']} GA={s['accumulation']} "
              f"steps={s['num_steps']:,} end={s['end_step']:,} math={s['math']:.0%} code={s['code']:.0%}")


if __name__ == "__main__":
    main()
