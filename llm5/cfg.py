from math import gcd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOK = ROOT / "models/tokenizer"
RUN = ROOT / "runs"
REPO = "tim123xyz/Test"
NAME = "m2-32"
EXPERTS = 32

MODEL = {
    "model_name": 'deepseek4-tiny',
    "override_model_config": True,
    "decoder_block": 'deepseek4',
    "base_num_decoder_layers": 2,
    "base_emb_dim": 2048,
    "base_mlp_dim": 2048,
    "base_moe_mlp_dim": 2048,
    "num_experts": EXPERTS,
    "num_experts_per_tok": 3,
    "prefuse_moe_weights": True,
    "float32_weight_sum": False,
    "shared_experts": 0,
    "routed_score_func": 'sqrtsoftplus',
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.5,
    "vocab_size": 65536,
    "base_num_query_heads": 16,
    "head_dim": 128,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 512,
    "o_lora_rank": 128,
    "enable_mhc_lite": True,
    "first_num_hash_layers": 0,
    "compress_ratios": [128, 4],
    "indexer_head_dim": 64,
    "indexer_n_heads": 8,
    "indexer_sparse_training": True,
    "indexer_loss_scaling_factor": 1.0,
    "sliding_window_size": 128,
    "use_qk_norm": True,
    "max_position_embeddings": 32768,
    "logits_via_embedding": True,
    "normalize_embedding_logits": False,
    "scan_layers": True,
    "attention": 'flash',
    "use_tokamax_splash": True,
    "sa_q_layout": 'SEQ_MINOR',
    "sa_k_layout": 'SEQ_MINOR',
    "sa_v_layout": 'SEQ_MINOR',
    "eval_sa_q_layout": 'SEQ_MINOR',
    "eval_sa_k_layout": 'SEQ_MINOR',
    "eval_sa_v_layout": 'SEQ_MINOR',
    "sa_use_fused_bwd_kernel": True,
    "fused_mlp": True,
    "dtype": 'bfloat16',
    "weight_dtype": 'bfloat16',
    "grad_dtype": 'float32',
    "mu_dtype": 'float32',
    "opt_type": 'muon',
    "muon_consistent_rms": 0.2,
    "muon_weight_decay": 0.1,
    "learning_rate": 0.001,
    "lr_schedule_type": 'wsd',
    "wsd_decay_steps_fraction": 0.0,
    "learning_rate_final_fraction": 1.0,
    "tokenizer_type": 'huggingface',
    "dataset_type": 'grain',
    "grain_file_type": 'parquet',
    "hf_path": 'openbmb/Ultra-FineWeb-L3',
    "tokenize_train_data": False,
    "tokenize_eval_data": False,
    "packing": False,
    "add_bos": False,
    "add_eos": False,
    "enable_data_shuffling": False,
    "grain_worker_count": 0,
    "grain_worker_count_eval": 0,
    "grain_num_threads": 1,
    "grain_prefetch_buffer_size": 2,
    "enable_checkpointing": True,
    "async_checkpointing": False,
    "skip_jax_distributed_system": True,
    "save_checkpoint_on_completion": True,
    "checkpoint_period": 100,
    "save_checkpoint_on_start": False,
    "max_num_checkpoints_to_keep": 3,
    "log_period": 10,
    "eval_interval": 500,
    "eval_steps": 8,
    "eval_per_device_batch_size": 1.0,
    "ici_fsdp_parallelism": 1,
    "dcn_data_parallelism": 1,
    "use_tokamax_gmm": True,
    "use_gmm_v2": True,
    "use_gmm_v2_heuristic_tiling": True,
    "moe_use_direct_token_gather": True,
    "use_csa_streamindex_kernel": True,
    "remat_policy": 'custom',
    "decoder_layer_input": 'remat',
    "query_proj": 'device',
    "key_proj": 'device',
    "value_proj": 'device',
    "kv_proj": 'device',
    "qkv_proj": 'device',
    "moe_routing_maps": 'device',
    "router_topk_matmul_vjp": True,
    "indexer_topk": 32,
    "logits_dot_in_fp32": False,
    "num_vocab_tiling": 16,
}

DATA = {
    "general": (
        "openbmb/Ultra-FineWeb-L3",
        "bc3b1ba986fcaef6871b9790a413b16267c2de0f",
        (
            ('Ultra-FineWeb-L3-en-QA-Synthetic', 'train', 0.25),
            ('Ultra-FineWeb-L3-en-Multi-Style-Synthetic', 'train', 0.25),
            ('Ultra-FineWeb-L3-zh-QA-Synthetic', 'train', 0.25),
            ('Ultra-FineWeb-L3-zh-Multi-Style-Synthetic', 'train', 0.25),
        ),
    ),
    "math": (
        "openbmb/UltraData-Math",
        "fe10db8efd35597fd7fcff8ff576b5ec4ea5ff87",
        (
            ('UltraData-Math-L3-Conversation-Synthetic', 'train', 0.25),
            ('UltraData-Math-L3-Multi-Style-Synthetic', 'train', 0.25),
            ('UltraData-Math-L3-QA-Synthetic', 'train', 0.25),
            ('UltraData-Math-L3-Textbook-Exercise-Synthetic', 'train', 0.25),
        ),
    ),
    "code": (
        "openbmb/UltraData-Code",
        "8ce985cfd4522832c8b986e05968c87fffa0c66a",
        (
            ('UltraData-Code-L3', 'py', 1),
            ('UltraData-Code-L3', 'cpp', 1),
            ('UltraData-Code-L3', 'java', 1),
            ('UltraData-Code-L3', 'js', 1),
            ('UltraData-Code-L3', 'go', 1),
            ('UltraData-Code-L3', 'rust', 1),
            ('UltraData-Code-L3', 'cs', 1),
            ('UltraData-Code-L3', 'php', 1),
            ('UltraData-Code-L3', 'r', 1),
            ('UltraData-Code-L3', 'rb', 1),
            ('UltraData-Code-L3', 'sh', 1),
        ),
    ),
}

STAGES = (
    (1024, 2000000000, 32, 0.05, 0.15),
    (2048, 2000000000, 16, 0.07, 0.17),
    (4096, 2000000000, 8, 0.09, 0.19),
    (8192, 2000000000, 4, 0.11, 0.21),
    (16384, 1000000000, 2, 0.13, 0.23),
    (32768, 1000000000, 1, 0.15, 0.25),
)


def short(length, batch):
    return length <= 1024 and length * batch <= 32768


def tiles(tokens, limit):
    if tokens < 128 or tokens % 128:
        return gcd(tokens, limit)
    blocks = tokens // 128
    return next(n for n in range(min(blocks, limit), 0, -1) if blocks % n == 0)


def configs(devices, root=RUN / "train", prefix=NAME, specs=STAGES):
    stages = []
    end = 0
    for length, budget, batch, _, _ in specs:
        local = max(1, min(batch, 32768 // length))
        micro = local * devices
        ga = max(1, round(batch * 16 / micro))
        end += max(1, round(budget / ((length - 1) * micro * ga)))
        stages.append((length, local, ga, end))

    previous = ""
    for length, local, ga, end in stages:
        tokens = local * length
        ep_limit = max(2, min(8, 32768 // tokens))
        ep = max(n for n in range(1, ep_limit + 1) if devices % n == 0 and EXPERTS % n == 0)
        cfg = dict(
            MODEL,
            run_name=f"{prefix}-{length // 1024}k",
            base_output_directory=root.as_posix(),
            max_target_length=length,
            per_device_batch_size=float(local),
            ici_data_parallelism=devices // ep,
            ici_expert_parallelism=ep,
            gradient_accumulation_steps=ga,
            steps=end,
            learning_rate_schedule_steps=stages[-1][-1],
            warmup_steps_fraction=64.01 / stages[-1][-1],
            tokenizer_path=str(TOK),
            data_shuffle_seed=42,
            init_weights_seed=42,
            load_full_state_path=previous,
            use_mhc_pallas_kernel=devices == 1,
            optimizer_memory_host_offload=devices == 1,
            num_vocab_tiling=16 if devices == 1 else tiles(tokens, max(1, min(16, tokens // 2048))),
        )
        head_bytes = max(local, 1) * length * (length // 4) * 4
        heads = min(8, max(1, (128 * 1024**2) // head_bytes))
        cfg["csa_qk_head_chunk_size"] = 1 << (heads.bit_length() - 1)
        if short(length, local):
            cfg["context"] = "device"
            if tokens <= 16384:
                cfg.update(moe_mlpwi_0="device", moe_mlpwi_1="device", moe_x_sorted="device")
        if devices == 1 and length == 1024:
            cfg.update(
                use_gmm_v2_heuristic_tiling=False,
                wi_tile_fwd_batch_seq=128, wi_tile_fwd_embed_dim=2048, wi_tile_fwd_mlp_dim=4096,
                wi_tile_dlhs_batch_seq=128, wi_tile_dlhs_embed_dim=2048, wi_tile_dlhs_mlp_dim=4096,
                wi_tile_drhs_batch_seq=256, wi_tile_drhs_embed_dim=2048, wi_tile_drhs_mlp_dim=2048,
                wo_tile_fwd_batch_seq=128, wo_tile_fwd_embed_dim=2048, wo_tile_fwd_mlp_dim=2048,
                wo_tile_dlhs_batch_seq=128, wo_tile_dlhs_embed_dim=2048, wo_tile_dlhs_mlp_dim=2048,
                wo_tile_drhs_batch_seq=256, wo_tile_drhs_embed_dim=2048, wo_tile_drhs_mlp_dim=2048,
                mhc_pallas_kernel_bwd_feature_block_size=2048,
            )
        previous = (root / cfg["run_name"] / "checkpoints" / str(end) / "items").as_posix()
        yield cfg
