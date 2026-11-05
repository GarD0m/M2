"""Native MaxText entry point with fused-QKV Muon dimension support."""
from functools import wraps
from importlib import import_module
from pathlib import Path
from contextlib import nullcontext
import warnings
import math
import json
import os
from tqdm.auto import tqdm

_TRAINING_PROGRESS = None


def configure_tpu():
    os.environ["JAX_PLATFORMS"] = "tpu"


def load_maxtext_conversion_configs():
    from transformers.configuration_utils import PreTrainedConfig

    module_name = "maxtext.checkpoint_conversion.utils.hf_model_configs"
    original = PreTrainedConfig.__post_init__

    @wraps(original)
    def initialize(config, **kwargs):
        cls = type(config)
        if (
            cls.__module__ == module_name
            and cls.__name__ in ("DeepseekV32Config", "DeepseekV4Config")
            and not hasattr(config, "rope_parameters")
            and kwargs.get("rope_scaling")
        ):
            config.rope_parameters = dict(kwargs.pop("rope_scaling"))
        return original(config, **kwargs)

    # These two legacy configs are constructed at import time.
    PreTrainedConfig.__post_init__ = initialize
    try:
        import_module(module_name)
    finally:
        PreTrainedConfig.__post_init__ = original


def install_checkpoint_manager_compat(module=None):
    if module is None:
        from maxtext.common import checkpointing as module
    native = module.create_orbax_checkpoint_manager
    if getattr(native, "_llm5_orbax_v1", False):
        return

    @wraps(native)
    def create(checkpoint_dir, enable_checkpointing, use_async, save_interval_steps,
               dataset_type=None, *args, **kwargs):
        # MaxText 0.2.5 accepts the old positional argument but ignores it.
        # The actual save and restore still use config.dataset_type.
        return native(checkpoint_dir, enable_checkpointing, use_async, save_interval_steps,
                      None, *args, **kwargs)

    create._llm5_orbax_v1 = True
    module.create_orbax_checkpoint_manager = create


def update_training_progress(metrics, step, metric_type):
    state = _TRAINING_PROGRESS
    if state is None or metric_type != "train":
        return
    progress = state["bar"]
    if not state["initialized"]:
        resumed = min(progress.total, max(0, int(step)-state["start_step"]))
        progress.reset()
        progress.n = progress.initial = progress.last_print_n = resumed
        state["initialized"] = True
    scalars = metrics.get("scalar", {})
    values = {}
    if "learning/loss" in scalars:
        values["loss"] = f"{float(scalars['learning/loss']):.4f}"
    if "learning/current_learning_rate" in scalars:
        values["lr"] = f"{float(scalars['learning/current_learning_rate']):.2e}"
    progress.set_postfix(values, refresh=False)
    completed = min(progress.total, max(0, int(step)+1-state["start_step"]))
    progress.update(max(0, completed-progress.n))

PERPLEXITY_SOURCES = (
    ("learning/lm_loss", "learning/perplexity", "learning/log_perplexity"),
    ("eval/avg_loss", "eval/avg_perplexity", "eval/avg_log_perplexity"),
)


def prepare_perplexity_metrics(metrics):
    scalars = metrics.get("scalar", {})
    if not any(source in scalars for source, _, _ in PERPLEXITY_SOURCES):
        return metrics
    scalars = dict(scalars)
    for source, perplexity, log_perplexity in PERPLEXITY_SOURCES:
        if source in scalars:
            loss = float(scalars[source])
            scalars[log_perplexity] = loss
            if math.isfinite(loss):
                try:
                    scalars[perplexity] = math.exp(loss)
                except OverflowError:
                    scalars[perplexity] = math.inf
    return {**metrics, "scalar": scalars}


def serializable_perplexity_metrics(metrics, float32=False):
    """Keep log perplexity when the derived exponential exceeds writer range."""
    scalars = metrics.get("scalar", {})
    limit = float.fromhex("0x1.fffffep+127") if float32 else float.fromhex("0x1.fffffffffffffp+1023")
    omit = {p for _, p, lp in PERPLEXITY_SOURCES if p in scalars and lp in scalars
            and math.isfinite(float(scalars[lp]))
            and (not math.isfinite(float(scalars[p])) or abs(float(scalars[p])) > limit)}
    if not omit:
        return metrics
    return {**metrics, "scalar": {key: val for key, val in scalars.items() if key not in omit}}


def nonfinite_metrics(metrics):
    import numpy as np
    from collections.abc import Mapping
    bad = {}
    def visit(name, value):
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(f"{name}/{key}" if name else key, item)
            return
        try:
            array = np.asarray(value)
            finite = np.isfinite(array)
        except TypeError:
            return
        if not finite.all():
            bad[name] = repr(array.reshape(-1)[~finite.reshape(-1)][:3].tolist())
    for group in ("scalar", "scalars"):
        visit("", metrics.get(group, {}))
    return bad


def install_metric_diagnostics(module=None):
    if module is None:
        from maxtext.common import metric_logger as module
    original = module.MetricLogger.write_metrics
    if getattr(original, "_llm5_finite_check_version", 0) == 3:
        return
    if getattr(original, "_llm5_finite_check", False):
        original = original.__wrapped__

    @wraps(original)
    def write_metrics(self, metrics, step, metric_type="train"):
        metrics = prepare_perplexity_metrics(metrics or {})
        # Exponential range overflow is represented by the finite log metric.
        bad = nonfinite_metrics(serializable_perplexity_metrics(metrics))
        if bad:
            detail = "; ".join(f"{key}={value}" for key, value in bad.items())
            message = f"Non-finite metrics at step={step}, phase={metric_type}: {detail}"
            critical = {"learning/loss", "learning/lm_loss", "learning/grad_norm", "learning/raw_grad_norm",
                        "learning/param_norm", "eval/avg_loss", "eval/loss"}
            if critical.intersection(bad):
                raise FloatingPointError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        result = original(self, metrics, step, metric_type)
        update_training_progress(metrics, step, metric_type)
        return result

    write_metrics._llm5_finite_check = True
    write_metrics._llm5_finite_check_version = 3
    module.MetricLogger.write_metrics = write_metrics
    for method_name, use_float32 in (("write_metrics_to_tensorboard", True), ("write_metrics_locally", False)):
        native = getattr(module.MetricLogger, method_name, None)
        if native is None or getattr(native, "_llm5_perplexity_range", False):
            continue
        def wrap_writer(writer, float32):
            @wraps(writer)
            def write(self, metrics, *args, **kwargs):
                return writer(self, serializable_perplexity_metrics(metrics, float32), *args, **kwargs)
            write._llm5_perplexity_range = True
            return write
        setattr(module.MetricLogger, method_name, wrap_writer(native, use_float32))


def install_json_config_support(module=None):
    if module is None:
        from maxtext.configs import pyconfig as module
    initialize = module._initialize_pydantic
    if not getattr(initialize, "_llm5_json_model_overrides", False):
        @wraps(initialize)
        def initialize_json(argv=None, *args, **kwargs):
            if not kwargs.get("base_config") and argv and len(argv) >= 2 and Path(argv[1]).suffix == ".json":
                config = json.loads(Path(module.resolve_config_path(argv[1])).read_text(encoding="utf-8"))
                if config.get("override_model_config"):
                    from omegaconf import OmegaConf
                    cli = OmegaConf.from_cli(argv[2:])
                    # MaxText's preset only yields to CLI/kwargs overrides.
                    # Promote explicit JSON values, preserving CLI and env precedence.
                    explicit = {
                        key: value for key, value in config.items()
                        if key != "base_config" and key not in cli
                        and module.yaml_key_to_env_key(key) not in os.environ
                    }
                    kwargs = {**explicit, **kwargs}
            return initialize(argv, *args, **kwargs)

        initialize_json._llm5_json_model_overrides = True
        module._initialize_pydantic = initialize_json

    original = module._resolve_or_infer_config
    if getattr(original, "_llm5_json_support", False):
        return

    @wraps(original)
    def resolve(argv=None, **kwargs):
        if not kwargs.get("base_config") and argv and len(argv) >= 2 and Path(argv[1]).suffix == ".json":
            return module.resolve_config_path(argv[1]), argv[2:]
        return original(argv, **kwargs)

    resolve._llm5_json_support = True
    module._resolve_or_infer_config = resolve


def install_fused_qkv_muon_support(module=None):
    if module is None:
        from maxtext.utils import muon_utils as module
    # Fused QKV has shape [D, Hq+2*Hkv, head_dim]. Muon must reduce
    # over D and flatten both output axes, rather than batch over heads.
    if "qkv_proj" not in module.ATTENTION_QKV_NAMES:
        module.ATTENTION_QKV_NAMES += ("qkv_proj",)


def install_mhc_token_tiling():
    import jax
    import jax.numpy as jnp
    from maxtext.layers import mhc

    native_head = mhc.DeepSeek4HyperHead.__call__
    if getattr(native_head, "_llm5_token_tiling", False):
        return

    def apply_tiles(function, inputs):
        batch, length, *features = inputs.shape
        tile = 64
        if length <= tile or length % tile:
            return function(inputs)
        chunks = inputs.reshape(batch, length // tile, tile, *features).swapaxes(0, 1)
        # Recompute each tile in backward instead of stacking its FP32 activations.
        outputs = jax.lax.map(jax.checkpoint(function, prevent_cse=False), chunks)
        return outputs.swapaxes(0, 1).reshape(batch, length, *outputs.shape[3:])

    @wraps(native_head)
    def hyper_head(self, inputs):
        if self.dtype != jnp.bfloat16:
            return native_head(self, inputs)

        def collapse(chunk):
            batch, length, streams, width = chunk.shape
            chunk = chunk.astype(jnp.bfloat16)
            flat = chunk.reshape(batch, length, streams * width)
            precision = jax.lax.Precision(self.config.matmul_precision)
            variance = jnp.mean(jnp.square(flat.astype(jnp.float32)), axis=-1)
            weights = jnp.asarray(self.hc_fn[...], jnp.bfloat16)
            mixes = jnp.einsum("bsd,dk->bsk", flat, weights, precision=precision,
                              preferred_element_type=jnp.float32)
            # Apply the scalar RMS correction after projection so the large operand stays BF16.
            mixes = mixes * jax.lax.rsqrt(variance[..., None] + self.eps)
            scale = jnp.asarray(self.hc_scale[...], jnp.float32)
            bias = jnp.asarray(self.hc_base[...], jnp.float32)
            gates = (jax.nn.sigmoid(mixes * scale + bias) + self.eps).astype(jnp.bfloat16)
            output = jnp.einsum("bsk,bskd->bsd", gates, chunk, precision=precision,
                               preferred_element_type=jnp.float32)
            return output.astype(self.dtype)

        return apply_tiles(collapse, inputs)

    class MhcRMSNorm(mhc.RMSNorm):
        def __call__(self, inputs, out_sharding=None):
            normalize = super().__call__
            if out_sharding is not None:
                return normalize(inputs, out_sharding)
            return apply_tiles(normalize, inputs)

    hyper_head._llm5_token_tiling = True
    mhc.DeepSeek4HyperHead.__call__ = hyper_head
    # This alias is used only for mHC's per-token stream normalization.
    mhc.RMSNorm = MhcRMSNorm


def install_deepseek4_loop():
    from types import SimpleNamespace
    from flax import nnx
    from maxtext.common.common_types import MODEL_MODE_TRAIN
    from maxtext.layers import moe, nnx_decoders
    from maxtext.models import deepseek4

    install_mhc_token_tiling()
    decoder = nnx_decoders.NNXDecoder
    native = decoder._apply_deepseek4_scanned_blocks
    if getattr(native, "_llm5_loop3", False):
        return

    class RoutedOnly(nnx.Module):
        def __init__(self, config, **kwargs):
            self.MoeBlock_0 = moe.RoutedMoE(
                config=config,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                intermediate_dim=config.moe_mlp_dim,
                **kwargs,
            )

        def __call__(self, inputs, input_ids=None):
            return self.MoeBlock_0(inputs=inputs, input_ids=input_ids)

    def create_mlp(config, **kwargs):
        if config.shared_experts == 0:
            return RoutedOnly(config, **kwargs)
        return moe.RoutedAndSharedMoE(config, **kwargs)

    # Only V4 uses this factory. Keep the native expert parameter path for Muon.
    deepseek4.moe = SimpleNamespace(RoutedAndSharedMoE=create_mlp)

    @wraps(native)
    def apply(self, y, decoder_segment_ids, decoder_positions, deterministic, model_mode, *args, **kwargs):
        if self.config.num_decoder_layers != 2 or self.config.first_num_hash_layers != 0:
            raise ValueError("The shared loop requires one HCA+CSA scan block and no prefix layers")
        if model_mode != MODEL_MODE_TRAIN:
            raise ValueError("The shared decoder loop currently supports training and eval in train mode")
        # Each call runs the native scan with the same two layer parameter sets.
        for _ in range(3):
            y = native(self, y, decoder_segment_ids, decoder_positions, deterministic, model_mode, *args, **kwargs)
        return y

    apply._llm5_loop3 = True
    decoder._apply_deepseek4_scanned_blocks = apply


def main():
    configure_tpu()
    load_maxtext_conversion_configs()
    install_checkpoint_manager_compat()
    from llm5.streaming import install_streaming_input
    install_streaming_input()
    install_json_config_support()
    install_fused_qkv_muon_support()
    install_metric_diagnostics()
    install_deepseek4_loop()
    from absl import app
    from maxtext.trainers.pre_train import train
    app.run(train.main)


def run_training_config(config_path, start_step=0, checkpoint_hub=None):
    """Run MaxText directly in the caller's process; preserve native exceptions."""
    global _TRAINING_PROGRESS
    configure_tpu()
    load_maxtext_conversion_configs()
    install_checkpoint_manager_compat()
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    previous = _TRAINING_PROGRESS
    with tqdm(total=config["steps"]-start_step, desc=config["run_name"], unit="step", mininterval=1.0,
              position=0, leave=False, dynamic_ncols=True) as progress:
        _TRAINING_PROGRESS = {"bar": progress, "start_step": start_step, "initialized": False}
        try:
            from llm5.streaming import install_streaming_input
            install_streaming_input()
            install_json_config_support()
            install_fused_qkv_muon_support()
            install_metric_diagnostics()
            install_deepseek4_loop()
            from absl import flags
            from maxtext.trainers.pre_train import train
            if not flags.FLAGS.is_parsed():
                flags.FLAGS(["llm5"])
            with checkpoint_hub.capture() if checkpoint_hub is not None else nullcontext():
                return train.main(["llm5.train", str(config_path)])
        finally:
            _TRAINING_PROGRESS = previous


if __name__ == "__main__":
    main()
