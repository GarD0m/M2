"""Native MaxText entry point with fused-QKV Muon dimension support."""
from functools import wraps
from pathlib import Path
import warnings
import math

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
    if getattr(original, "_llm5_finite_check_version", 0) == 2:
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
        return original(self, metrics, step, metric_type)

    write_metrics._llm5_finite_check = True
    write_metrics._llm5_finite_check_version = 2
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


def main():
    install_json_config_support()
    install_fused_qkv_muon_support()
    install_metric_diagnostics()
    from absl import app
    from maxtext.trainers.pre_train import train
    app.run(train.main)


def run_training_config(config_path):
    """Run MaxText directly in the caller's process; preserve native exceptions."""
    install_json_config_support()
    install_fused_qkv_muon_support()
    install_metric_diagnostics()
    from absl import flags
    from maxtext.trainers.pre_train import train
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["llm5"])
    return train.main(["llm5.train", str(config_path)])


if __name__ == "__main__":
    main()
