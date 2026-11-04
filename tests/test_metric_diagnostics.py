from types import SimpleNamespace
import numpy as np
import pytest
from llm5.train import install_metric_diagnostics, nonfinite_metrics
from llm5.train import prepare_perplexity_metrics, serializable_perplexity_metrics


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


def test_perplexity_uses_host_range_and_keeps_log_metric():
    metrics = {"scalar": {"learning/loss": 100., "learning/lm_loss": 100., "learning/perplexity": np.inf}}
    report = prepare_perplexity_metrics(metrics)
    assert report["scalar"]["learning/perplexity"] == pytest.approx(np.exp(100.))
    assert report["scalar"]["learning/log_perplexity"] == 100.
    assert "learning/perplexity" not in serializable_perplexity_metrics(report, float32=True)["scalar"]
    assert "learning/perplexity" in serializable_perplexity_metrics(report)["scalar"]
    assert metrics["scalar"]["learning/perplexity"] == np.inf
    report = prepare_perplexity_metrics({"scalar": {"learning/lm_loss": 1000.}})
    assert "learning/perplexity" not in serializable_perplexity_metrics(report)["scalar"]
    assert report["scalar"]["learning/log_perplexity"] == 1000.


def test_writers_omit_only_unrepresentable_perplexity():
    class Logger:
        def write_metrics(self, metrics, step, metric_type="train"):
            self.console = metrics
            self.write_metrics_to_tensorboard(metrics, step, metric_type)
            self.write_metrics_locally(metrics, step)
        def write_metrics_to_tensorboard(self, metrics, step, metric_type):
            self.tensorboard = metrics
        def write_metrics_locally(self, metrics, step):
            self.json = metrics
    module = SimpleNamespace(MetricLogger=Logger)
    install_metric_diagnostics(module)
    logger = Logger()
    logger.write_metrics({"scalar": {"learning/loss": 100., "learning/lm_loss": 100., "learning/perplexity": np.inf}}, 0)
    assert "learning/perplexity" not in logger.tensorboard["scalar"]
    assert logger.tensorboard["scalar"]["learning/log_perplexity"] == 100.
    assert np.isfinite(logger.json["scalar"]["learning/perplexity"])
    logger.write_metrics({"scalar": {"learning/loss": 1000., "learning/lm_loss": 1000., "learning/perplexity": np.inf}}, 1)
    assert "learning/perplexity" not in logger.json["scalar"]
    assert logger.json["scalar"]["learning/log_perplexity"] == 1000.
