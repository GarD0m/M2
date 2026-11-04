from types import SimpleNamespace
import numpy as np
import pytest
from llm5.train import install_metric_diagnostics, nonfinite_metrics


def logger_module():
    class Logger:
        def __init__(self):
            self.written = []
        def write_metrics(self, metrics, step, metric_type="train"):
            self.written.append((metrics, step, metric_type))
    return SimpleNamespace(MetricLogger=Logger)


def test_nonfinite_loss_raises_before_tensorboard():
    module = logger_module()
    install_metric_diagnostics(module)
    method = module.MetricLogger.write_metrics
    install_metric_diagnostics(module)
    assert module.MetricLogger.write_metrics is method
    logger = module.MetricLogger()
    with pytest.raises(FloatingPointError, match=r"step=3.*learning/loss=\[nan\]"):
        logger.write_metrics({"scalar": {"learning/loss": float("nan")}}, 3)
    assert not logger.written


def test_nonfinite_gradient_reports_its_name():
    module = logger_module()
    install_metric_diagnostics(module)
    with pytest.raises(FloatingPointError, match="learning/raw_grad_norm"):
        module.MetricLogger().write_metrics({"scalar": {"learning/loss": 11., "learning/raw_grad_norm": np.inf}}, 0)


def test_auxiliary_nonfinite_warns_without_modifying_metrics():
    module = logger_module()
    install_metric_diagnostics(module)
    logger = module.MetricLogger()
    metrics = {"scalar": {"learning/loss": 11., "perf/rate": np.inf},
               "scalars": {"activation": {"layer0": np.array([1., np.nan])}}}
    with pytest.warns(RuntimeWarning, match="perf/rate.*activation/layer0"):
        logger.write_metrics(metrics, 1)
    assert logger.written[0][0] is metrics
    assert nonfinite_metrics({"scalar": {"learning/loss": 11., "label": "train"}}) == {}
