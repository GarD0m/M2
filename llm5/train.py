"""Native MaxText entry point with fused-QKV Muon dimension support."""
from functools import partial, wraps
from importlib import import_module
from pathlib import Path
from contextlib import nullcontext
import warnings
import math
import json
import os
from tqdm.auto import tqdm

_TRAINING_PROGRESS = None
TENSORBOARD_SCALARS = {
    "learning/loss",
    "learning/current_learning_rate",
    "eval/avg_loss",
    "perf/per_device_tokens_per_sec",
}


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
    from jax import device_get

    if module is None:
        from maxtext.common import metric_logger as module
    original = module.MetricLogger.write_metrics
    if getattr(original, "_llm5_finite_check_version", 0) == 4:
        return
    if getattr(original, "_llm5_finite_check", False):
        original = original.__wrapped__

    @wraps(original)
    def write_metrics(self, metrics, step, metric_type="train"):
        metrics = dict(metrics or {})
        metrics.update(device_get({key: metrics[key] for key in ("scalar", "scalars") if key in metrics}))
        metrics = prepare_perplexity_metrics(metrics)
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
    write_metrics._llm5_finite_check_version = 4
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

    tensorboard = module.MetricLogger.write_metrics_to_tensorboard

    @wraps(tensorboard)
    def write_tensorboard(self, metrics, step, metric_type):
        if metric_type == "running_eval":
            return
        scalars = {key: value for key, value in metrics.get("scalar", {}).items()
                   if key in TENSORBOARD_SCALARS}
        return tensorboard(self, {"scalar": scalars}, step, metric_type)

    def record_setup(self, params):
        # Throughput reporting still needs these values in memory.
        self.metadata[module.MetadataKey.PER_DEVICE_TFLOPS], _, _ = (
            module.maxtext_utils.calculate_tflops_training_per_device(self.config)
        )
        self.metadata[module.MetadataKey.PER_DEVICE_TOKENS] = (
            module.maxtext_utils.calculate_tokens_training_per_device(self.config)
        )
        count = module.max_utils.calculate_num_params_from_pytree(params)
        module.max_logging.log(f"number parameters: {count / 1e9:.3f} billion")

    module.MetricLogger.write_metrics_to_tensorboard = write_tensorboard
    module.MetricLogger.write_setup_info_to_tensorboard = record_setup


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


def install_mhc_low_memory():
    import jax
    import jax.numpy as jnp
    from maxtext.layers import mhc, normalizations

    native_head = mhc.DeepSeek4HyperHead.__call__
    if getattr(native_head, "_llm5_mhc_low_memory", False):
        return
    if getattr(native_head, "_llm5_token_tiling", False):
        native_head = native_head.__wrapped__

    def row_dot(x, y):
        return jnp.einsum("bsd,bsd->bs", x, y, preferred_element_type=jnp.float32)[..., None]

    def normalize(x, epsilon):
        inverse_rms = jax.lax.rsqrt(row_dot(x, x) / x.shape[-1] + epsilon)
        return x * inverse_rms.astype(x.dtype), inverse_rms

    @partial(jax.custom_vjp, nondiff_argnums=(2,))
    def rms_norm(x, scale, epsilon):
        normalized, _ = normalize(x, epsilon)
        return normalized * scale

    def rms_forward(x, scale, epsilon):
        normalized, inverse_rms = normalize(x, epsilon)
        return normalized * scale, (x, scale, inverse_rms)

    def rms_backward(epsilon, residuals, gradient):
        x, scale, inverse_rms = residuals
        inverse_bf16 = inverse_rms.astype(x.dtype)
        normalized = x * inverse_bf16
        scaled_gradient = gradient * scale
        correction = (row_dot(normalized, scaled_gradient) / x.shape[-1]).astype(x.dtype)
        dx = (scaled_gradient - normalized * correction) * inverse_bf16
        dscale = jnp.einsum("bsd,bsd->d", gradient, normalized, preferred_element_type=jnp.float32)
        return dx, dscale.astype(scale.dtype)

    rms_norm.defvjp(rms_forward, rms_backward)

    class MhcRMSNorm(normalizations.RMSNorm):
        def __call__(self, inputs, out_sharding=None):
            if (self.dtype != jnp.bfloat16 or inputs.dtype != jnp.bfloat16
                    or inputs.ndim != 3 or out_sharding is not None
                    or not self.with_scale or self.scale_offset != 0
                    or self.parameter_memory_host_offload):
                return super().__call__(inputs, out_sharding)
            scale = self.scale.get_value().astype(jnp.bfloat16)
            return rms_norm(inputs, scale, self.epsilon)

    @wraps(native_head)
    def hyper_head(self, inputs):
        if self.dtype != jnp.bfloat16:
            return native_head(self, inputs)

        def collapse(chunk):
            batch, length, streams, width = chunk.shape
            chunk = chunk.astype(jnp.bfloat16)
            flat = chunk.reshape(batch, length, streams * width)
            precision = jax.lax.Precision(self.config.matmul_precision)
            variance = row_dot(flat, flat) / flat.shape[-1]
            weights = jnp.asarray(self.hc_fn[...], jnp.bfloat16)
            mixes = jnp.einsum("bsd,dk->bsk", flat, weights, precision=precision,
                              preferred_element_type=jnp.float32)
            # Apply the scalar RMS correction after projection so the large operand stays BF16.
            mixes = mixes * jax.lax.rsqrt(variance + self.eps)
            scale = jnp.asarray(self.hc_scale[...], jnp.float32)
            bias = jnp.asarray(self.hc_base[...], jnp.float32)
            gates = (jax.nn.sigmoid(mixes * scale + bias) + self.eps).astype(jnp.bfloat16)
            output = jnp.einsum("bsk,bskd->bsd", gates, chunk, precision=precision,
                               preferred_element_type=jnp.float32)
            return output.astype(self.dtype)

        return jax.checkpoint(collapse)(inputs)

    hyper_head._llm5_mhc_low_memory = True
    mhc.DeepSeek4HyperHead.__call__ = hyper_head
    mhc.RMSNorm = MhcRMSNorm


def install_deepseek4_support():
    from types import SimpleNamespace
    from flax import nnx
    from maxtext.layers import moe, nnx_decoders
    from maxtext.models import deepseek4

    install_mhc_low_memory()
    install_moe_route_reuse(moe)
    decoder = nnx_decoders.NNXDecoder
    native = decoder._apply_deepseek4_scanned_blocks
    if getattr(native, "_llm5_loop3_remat", False) or getattr(native, "_llm5_loop3", False):
        decoder._apply_deepseek4_scanned_blocks = native.__wrapped__
    if getattr(deepseek4.moe.RoutedAndSharedMoE, "_llm5_routed_only", False):
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
    create_mlp._llm5_routed_only = True
    deepseek4.moe = SimpleNamespace(RoutedAndSharedMoE=create_mlp)


def install_moe_route_reuse(moe):
    import jax
    import jax.numpy as jnp
    from jax.ad_checkpoint import checkpoint_name

    native = moe._route_activations
    if getattr(native, "_llm5_reuse_permutation", False):
        return

    def route_forward(inputs, selected_experts):
        top_k = selected_experts.size // inputs.shape[0]
        permutation = checkpoint_name(jnp.argsort(selected_experts.reshape(-1)), "moe_routing_maps")
        return inputs[permutation // top_k], permutation.reshape(inputs.shape[0], top_k)

    @jax.custom_vjp
    def route(inputs, selected_experts):
        return route_forward(inputs, selected_experts)[0]

    def route_backward(permutation, gradient):
        ordered = gradient[jnp.argsort(permutation.reshape(-1))]
        return ordered.reshape(permutation.shape + gradient.shape[1:]).sum(axis=1), None

    route.defvjp(route_forward, route_backward)

    @wraps(native)
    def route_activations(inputs, selected_experts):
        if not isinstance(inputs, jax.Array) or inputs.shape[0] == 0:
            return native(inputs, selected_experts)
        return route(inputs, selected_experts)

    route_activations._llm5_reuse_permutation = True
    moe._route_activations = route_activations


def install_aligned_logits_tiles(trainer):
    from maxtext.configs.pyconfig import HyperParameters

    native = trainer.vocab_tiling_nnx_loss
    if getattr(native, "_llm5_hyperparameter_logits_tiles", False):
        return
    if getattr(native, "_llm5_aligned_logits_tiles", False):
        native = native.__wrapped__

    @wraps(native)
    def loss(model, hidden_states, data, config, is_train):
        tokens = math.prod(hidden_states.shape[:-1])
        if tokens % 128 == 0:
            tile_limit = config.num_vocab_tiling
            if not is_train and config.decoder_block == "deepseek4":
                eval_ratio = min(1, config.eval_per_device_batch_size / config.per_device_batch_size)
                tile_limit = max(1, math.ceil(tile_limit * eval_ratio))
            tiles = math.gcd(tile_limit, tokens // 128)
            if tiles != config.num_vocab_tiling:
                update = {"num_vocab_tiling": tiles}
                if isinstance(config, HyperParameters):
                    config = HyperParameters(config._pydantic_config.model_copy(update=update))
                else:
                    config = config.model_copy(update=update)
        return native(model, hidden_states, data, config, is_train)

    loss._llm5_hyperparameter_logits_tiles = True
    trainer.vocab_tiling_nnx_loss = loss


def load_trainer():
    configure_tpu()
    load_maxtext_conversion_configs()
    install_checkpoint_manager_compat()
    from llm5.streaming import install_streaming_input
    install_streaming_input()
    install_json_config_support()
    install_fused_qkv_muon_support()
    install_metric_diagnostics()
    install_deepseek4_support()
    from maxtext.trainers.pre_train import train
    install_aligned_logits_tiles(train)
    return train


def main():
    from absl import app

    train = load_trainer()
    app.run(train.main)


def run_training_config(config_path, start_step=0, checkpoint_hub=None, overrides=()):
    """Run MaxText directly in the caller's process; preserve native exceptions."""
    global _TRAINING_PROGRESS
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    previous = _TRAINING_PROGRESS
    with tqdm(total=config["steps"]-start_step, desc=config["run_name"], unit="step", mininterval=1.0,
              position=0, leave=False, dynamic_ncols=True) as progress:
        _TRAINING_PROGRESS = {"bar": progress, "start_step": start_step, "initialized": False}
        try:
            from absl import flags
            train = load_trainer()
            if not flags.FLAGS.is_parsed():
                flags.FLAGS(["llm5"])
            with checkpoint_hub.capture() if checkpoint_hub is not None else nullcontext():
                return train.main(["llm5.train", str(config_path), *overrides])
        finally:
            _TRAINING_PROGRESS = previous


if __name__ == "__main__":
    main()
