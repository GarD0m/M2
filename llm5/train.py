import os
from contextlib import nullcontext
from functools import cache
from importlib import import_module
from pathlib import Path
from unittest.mock import patch


def setup():
    os.environ["JAX_PLATFORMS"] = "tpu"
    from absl import logging
    from llm5.data import setup as setup_data

    logging.set_verbosity(logging.WARNING)
    setup_data()


def load_hf():
    from transformers.configuration_utils import PreTrainedConfig

    module = "maxtext.checkpoint_conversion.utils.hf_model_configs"
    native = PreTrainedConfig.__post_init__

    def init(cfg, **kwargs):
        if (type(cfg).__module__ == module and type(cfg).__name__ in ("DeepseekV32Config", "DeepseekV4Config")
                and not hasattr(cfg, "rope_parameters") and kwargs.get("rope_scaling")):
            cfg.rope_parameters = dict(kwargs.pop("rope_scaling"))
        return native(cfg, **kwargs)

    with patch.object(PreTrainedConfig, "__post_init__", init):
        import_module(module)


@cache
def trainer():
    load_hf()
    from maxtext.common import checkpointing as cp
    from llm5 import csa, data, logits, mhc, moe, opt

    native = cp.create_orbax_checkpoint_manager

    def manager(path, enabled, async_save, period, dataset_type=None, *args, **kwargs):
        return native(path, enabled, async_save, period, None, *args, **kwargs)

    cp.create_orbax_checkpoint_manager = manager
    data.patch()
    opt.patch()
    mhc.patch()
    moe.patch()
    csa.patch()
    from maxtext.trainers.pre_train import train
    from absl import flags

    logits.patch(train)
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["m2"])
    return train


METRICS = {"learning/loss", "learning/current_learning_rate", "eval/avg_loss", "perf/per_device_tokens_per_sec"}


def run(cfg, hub=None):
    train = trainer()
    from maxtext.common import metric_logger as metrics
    from maxtext.configs import pyconfig

    write = metrics.MetricLogger.write_metrics_to_tensorboard
    init = pyconfig.initialize
    base = str(Path(pyconfig.MAXTEXT_CONFIGS_DIR) / "base.yml")

    def config(_argv):
        return init(["m2", base], **cfg)

    def log(self, values, step, kind):
        if hub is not None and kind != "running_eval":
            scalars = {key: val for key, val in values.get("scalar", {}).items() if key in METRICS}
            return write(self, {"scalar": scalars}, step, kind)

    def setup_metrics(self, params):
        self.metadata[metrics.MetadataKey.PER_DEVICE_TFLOPS], _, _ = (
            metrics.maxtext_utils.calculate_tflops_training_per_device(self.config)
        )
        self.metadata[metrics.MetadataKey.PER_DEVICE_TOKENS] = (
            metrics.maxtext_utils.calculate_tokens_training_per_device(self.config)
        )

    capture = hub.capture(cfg["run_name"]) if hub is not None else nullcontext()
    with (capture, patch.object(pyconfig, "initialize", config),
          patch.object(metrics.MetricLogger, "write_metrics_to_tensorboard", log),
          patch.object(metrics.MetricLogger, "write_setup_info_to_tensorboard", setup_metrics)):
        train.main(["m2"])
