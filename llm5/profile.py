"""Capture one short, isolated trace with a CLI-selected configuration."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from llm5 import launch, streaming
from llm5.mhc import use_fused_pre_backward
from llm5.plan import aligned_logits_tiles, generate, read_json, use_short_context_tuning
from llm5.train import POLAR_EXPRESS_COEFFICIENTS, configure_tpu, run_training_config

MATH_FRACTION = 0.05
CODE_FRACTION = 0.15


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--microbatch", type=int, default=32, help="Global number of sequences")
    parser.add_argument("--ga", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=1, help="Optimizer steps to skip before tracing")
    parser.add_argument("--profile-steps", type=int, default=1, help="Optimizer steps to trace")
    args = parser.parse_args(argv)
    if args.length < 2 or args.microbatch < 1 or args.ga < 1:
        parser.error("length must be at least 2; microbatch and GA must be positive")
    if args.warmup_steps < 0 or args.profile_steps < 1:
        parser.error("warmup steps must be nonnegative; profile steps must be positive")
    # XLA reads its flags when the backend starts. Do this before querying devices.
    import jax
    from jax._src import xla_bridge
    if xla_bridge.backends_are_initialized():
        parser.error("JAX is already initialized; restart the kernel before profiling")
    flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_dump_to" in flags:
        parser.error("remove the existing --xla_dump_to from XLA_FLAGS before profiling")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    hlo_dir = f"runs/profiles/{run_id}-test/hlo"
    os.environ["XLA_FLAGS"] = (f"{flags} --xla_dump_to={hlo_dir} "
                               "--xla_dump_hlo_as_text --xla_dump_hlo_as_proto").strip()
    # A cache hit can bypass compilation and leave no HLO dump.
    jax.config.update("jax_enable_compilation_cache", False)
    configure_tpu()
    streaming.configure(train_weights=[1 - MATH_FRACTION - CODE_FRACTION, MATH_FRACTION, CODE_FRACTION])
    device_count = launch.check_runtime()
    device_kind = jax.devices()[0].device_kind
    if args.microbatch % device_count:
        parser.error("global microbatch must be divisible by the TPU device count")
    # A fresh directory prevents MaxText from restoring or overwriting a training run.
    # MaxText flushes the trace at the end of the last profiled step.
    steps = args.warmup_steps + args.profile_steps
    budget = (args.length - 1) * args.microbatch * args.ga * max(100, steps + 1)
    curriculum = {"seed": 42, "warmup_steps": 64, "target_tokens": budget,
                  "stages": [{"name": "test", "length": args.length, "budget": budget,
                              "microbatch": args.microbatch, "accumulation": args.ga,
                              "math": MATH_FRACTION, "code": CODE_FRACTION}]}
    with TemporaryDirectory(prefix="llm5-profile-") as temporary:
        manifest = generate(temporary, "models/tokenizer", "runs/profiles",
                            run_id, device_count=device_count, curriculum_config=curriculum,
                            device_kind=device_kind)
        stage = manifest["stages"][0]
        config_path = Path(stage["config"])
        cfg = read_json(config_path)
        if cfg["indexer_topk"] >= args.length // 4:
            parser.error("length // 4 must exceed the model's CSA top-k")
        cfg.update(steps=steps, profiler="xplane", profile_cleanly=True,
                   skip_first_n_steps_for_profiler=args.warmup_steps, profiler_steps=args.profile_steps,
                   enable_checkpointing=False, save_checkpoint_on_completion=False,
                   eval_interval=-1, load_full_state_path="")
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        launch.check_data(stage, cfg, validate_tokenizer=True)
        output_dir = Path(cfg["base_output_directory"]) / cfg["run_name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        tile_limit = cfg["num_vocab_tiling"]
        effective_tiles = aligned_logits_tiles(args.length * stage["per_device_microbatch"], tile_limit)
        short_context = use_short_context_tuning(args.length, stage["per_device_microbatch"], device_count)
        mhc_shape = (args.microbatch, args.length, cfg["mhc_expansion_rate"], cfg["base_emb_dim"])
        metadata = {"length": args.length, "global_microbatch": args.microbatch, "ga": args.ga,
                    "device_count": device_count, "device_kind": device_kind, "warmup_steps": args.warmup_steps,
                    "profile_steps": args.profile_steps, "vocab_tiles": tile_limit,
                    "effective_vocab_tiles": effective_tiles,
                    "csa_head_chunk": cfg["csa_qk_head_chunk_size"],
                    "native_csa_mask": False,
                    "indexer_topk": cfg["indexer_topk"],
                    "csa_indexer_saved": short_context and cfg["use_csa_streamindex_kernel"],
                    "prefuse_moe_weights": cfg["prefuse_moe_weights"],
                    "moe_up_projections_saved": cfg.get("moe_mlpwi_0") == cfg.get("moe_mlpwi_1") == "device",
                    "moe_sorted_input_saved": cfg.get("moe_x_sorted") == "device",
                    "muon_polar_express_steps": len(POLAR_EXPRESS_COEFFICIENTS),
                    "splash_context_saved": cfg.get("context") == "device",
                    "mhc_lite": cfg["enable_mhc_lite"],
                    "mhc_lite_pallas": cfg["use_mhc_pallas_kernel"],
                    "mhc_bwd_feature_tile": cfg.get("mhc_pallas_kernel_bwd_feature_block_size", 1024),
                    "moe_up_weight_grad_tile": (None if cfg["use_gmm_v2_heuristic_tiling"]
                                                else cfg["wi_tile_drhs_mlp_dim"]),
                    "mhc_pre_saved": short_context and cfg["use_mhc_pallas_kernel"],
                    "mhc_pre_bwd_fused": (device_count == 1 and cfg["use_mhc_pallas_kernel"]
                                         and use_fused_pre_backward(mhc_shape, cfg.get("mhc_pallas_kernel_bwd_block_size", 128))),
                    "logits_dot_in_fp32": cfg["logits_dot_in_fp32"],
                    "math": MATH_FRACTION, "code": CODE_FRACTION}
        (output_dir / "profile-config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Profiling: {device_kind}, length={stage['length']}, "
              f"global microbatch={stage['microbatch']}, GA={stage['accumulation']}, "
              f"warmup={args.warmup_steps}, profile steps={args.profile_steps}, "
              f"vocab tiles={effective_tiles}/{tile_limit}, CSA head chunk={cfg['csa_qk_head_chunk_size']}, "
              f"indexer top-k={cfg['indexer_topk']}, mHC-lite Pallas={cfg['use_mhc_pallas_kernel']}, "
              f"fused MoE gate/up={cfg['prefuse_moe_weights']}", flush=True)
        print(f"TensorBoard logdir: {cfg['base_output_directory']}/{cfg['run_name']}/tensorboard", flush=True)
        print(f"HLO dump: {Path(hlo_dir).resolve()}", flush=True)
        run_training_config(config_path, record_metrics=False)


if __name__ == "__main__":
    main()
