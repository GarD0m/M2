"""Capture one short, isolated trace with a CLI-selected configuration."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from llm5 import launch, streaming
from llm5.plan import aligned_logits_tiles, generate, read_json
from llm5.train import configure_tpu, run_training_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--microbatch", type=int, default=32, help="Global number of sequences")
    parser.add_argument("--ga", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", type=int, default=1, help="Optimizer steps to skip before tracing")
    parser.add_argument("--profile-steps", type=int, default=4, help="Optimizer steps to trace")
    parser.add_argument("--vocab-tiles", type=int, default=16, help="Maximum number of vocabulary loss tiles")
    parser.add_argument("--csa-head-chunk", type=int, help="CSA teacher heads per chunk (default: automatic)")
    parser.add_argument("--native-csa-mask", action="store_true", help="Use MaxText's CSA validity gather as a baseline")
    parser.add_argument("--indexer-topk", type=int, default=32, help="CSA compressed blocks selected per token")
    parser.add_argument("--moe-save", choices=("none", "sorted-input", "up-projections"), default="none",
                        help="Keep selected MoE activations on device across the backward pass")
    parser.add_argument("--splash-bwd-kv-block", type=int, choices=(128, 256, 512),
                        help="Tokamax Splash backward KV block and compute block size")
    parser.add_argument("--mhc-lite", action="store_true", help="Test mHC-lite with its fused TPU Pallas kernel")
    parser.add_argument("--fp32-logits-dot", action="store_true", help="Use FP32 instead of BF16 logits projection output")
    parser.add_argument("--aggressive", action="store_true", help="Enable fused mHC-lite alongside the default optimizations")
    parser.add_argument("--math", type=float, default=0.05, help="Math data fraction")
    parser.add_argument("--code", type=float, default=0.15, help="Code data fraction")
    args = parser.parse_args(argv)
    args.mhc_lite = args.mhc_lite or args.aggressive
    if args.length < 2 or args.microbatch < 1 or args.ga < 1:
        parser.error("length must be at least 2; microbatch and GA must be positive")
    if args.warmup_steps < 0 or args.profile_steps < 1 or args.vocab_tiles < 1:
        parser.error("warmup steps must be nonnegative; profile steps and vocab tiles must be positive")
    if args.csa_head_chunk is not None and args.csa_head_chunk < 1:
        parser.error("CSA head chunk must be positive")
    if args.indexer_topk < 1 or args.indexer_topk >= args.length // 4:
        parser.error("indexer top-k must be positive and less than length // 4")
    if not (0 <= args.math <= 1 and 0 <= args.code <= 1 - args.math):
        parser.error("math and code fractions must be nonnegative and sum to at most 1")
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
    streaming.configure(train_weights=[1 - args.math - args.code, args.math, args.code])
    device_count = launch.check_runtime()
    if args.microbatch % device_count:
        parser.error("global microbatch must be divisible by the TPU device count")
    # A fresh directory prevents MaxText from restoring or overwriting a training run.
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
                   num_vocab_tiling=args.vocab_tiles,
                   indexer_topk=args.indexer_topk,
                   logits_dot_in_fp32=args.fp32_logits_dot,
                   enable_mhc_lite=args.mhc_lite, use_mhc_pallas_kernel=args.mhc_lite,
                   skip_first_n_steps_for_profiler=args.warmup_steps, profiler_steps=args.profile_steps,
                   enable_checkpointing=False, save_checkpoint_on_completion=False,
                   eval_interval=-1, load_full_state_path="")
        if args.moe_save == "sorted-input":
            cfg["moe_x_sorted"] = "device"
        elif args.moe_save == "up-projections":
            cfg.update(moe_mlpwi_0="device", moe_mlpwi_1="device")
        if args.splash_bwd_kv_block is not None:
            cfg.update(sa_block_kv_dkv=args.splash_bwd_kv_block,
                       sa_block_kv_dkv_compute=args.splash_bwd_kv_block)
        if args.csa_head_chunk is not None:
            heads = (cfg["base_num_query_heads"], cfg["indexer_n_heads"])
            if any(n % args.csa_head_chunk for n in heads):
                parser.error("CSA head chunk must divide both query and indexer head counts")
            cfg["csa_qk_head_chunk_size"] = args.csa_head_chunk
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        launch.check_data(stage, cfg, validate_tokenizer=True)
        output_dir = Path(cfg["base_output_directory"]) / cfg["run_name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        effective_tiles = aligned_logits_tiles(args.length * stage["per_device_microbatch"], args.vocab_tiles)
        metadata = {"length": args.length, "global_microbatch": args.microbatch, "ga": args.ga,
                    "device_count": device_count, "warmup_steps": args.warmup_steps,
                    "profile_steps": args.profile_steps, "vocab_tiles": args.vocab_tiles,
                    "effective_vocab_tiles": effective_tiles,
                    "csa_head_chunk": cfg["csa_qk_head_chunk_size"],
                    "native_csa_mask": args.native_csa_mask,
                    "indexer_topk": args.indexer_topk,
                    "moe_save": args.moe_save,
                    "splash_bwd_kv_block": cfg.get("sa_block_kv_dkv", 512),
                    "mhc_lite_pallas": args.mhc_lite,
                    "logits_dot_in_fp32": args.fp32_logits_dot,
                    "math": args.math, "code": args.code}
        (output_dir / "profile-config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Profiling: length={stage['length']}, "
              f"global microbatch={stage['microbatch']}, GA={stage['accumulation']}, "
              f"warmup={args.warmup_steps}, profile steps={args.profile_steps}, "
              f"vocab tiles={effective_tiles}/{args.vocab_tiles}, CSA head chunk={cfg['csa_qk_head_chunk_size']}, "
              f"indexer top-k={args.indexer_topk}, mHC-lite Pallas={args.mhc_lite}, "
              f"FP32 logits dot={args.fp32_logits_dot}", flush=True)
        print(f"TensorBoard logdir: {cfg['base_output_directory']}/{cfg['run_name']}/tensorboard", flush=True)
        print(f"HLO dump: {Path(hlo_dir).resolve()}", flush=True)
        run_training_config(config_path, record_metrics=False, native_csa_mask=args.native_csa_mask)


if __name__ == "__main__":
    main()
