import json
import os
import sys
import pytest
from llm5 import tune
from llm5.plan import ROOT


def test_native_error_is_raised_directly_without_subprocess(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["tune", "--prepare-plan"])
    monkeypatch.setattr(tune, "check_runtime", lambda: 1)
    error = ValueError("native training error")
    calls = []
    def native(config):
        calls.append(os.getpid())
        assert json.loads(config.read_text())["load_full_state_path"] == ""
        raise error
    monkeypatch.setattr(tune, "run_training_config", native)
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: pytest.fail("Do not spawn a training process"))
    with pytest.raises(ValueError) as caught:
        tune.main()
    assert caught.value is error
    assert calls == [os.getpid()]


def test_successful_trials_share_one_process_and_cleanup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["tune", "--prepare-plan"])
    monkeypatch.setattr(tune, "check_runtime", lambda: 1)
    pids, cleanup = [], []
    def native(config_path):
        pids.append(os.getpid())
        cfg = json.loads(config_path.read_text())
        rows = [{"step": i, "perf/step_time_seconds": 10/cfg["per_device_batch_size"]}
                for i in range(12)]
        from pathlib import Path
        Path(cfg["metrics_file"]).write_text("\n".join(map(json.dumps, rows)))
    monkeypatch.setattr(tune, "run_training_config", native)
    monkeypatch.setattr(tune, "release_probe_resources", lambda: cleanup.append(True))
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: pytest.fail("Do not spawn a training process"))
    tune.main()
    assert len(pids) > 6 and set(pids) == {os.getpid()}
    assert len(cleanup) == len(pids)
    choices = json.loads((tmp_path / "runs/microbatches.json").read_text())
    assert choices["1k"] == 128
    manifest = json.loads((tmp_path / "runs/plan/manifest.json").read_text())
    for stage in manifest["stages"]:
        assert stage["microbatch"]*stage["accumulation"]*stage["length"] == 524288


def test_training_modules_do_not_launch_subprocesses():
    import ast
    for file in ("llm5/tune.py", "llm5/launch.py", "llm5/train.py"):
        tree = ast.parse((ROOT / file).read_text())
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        assert all(getattr(n, "module", None) != "subprocess" for n in imports)
        assert all(alias.name != "subprocess" for n in imports if isinstance(n, ast.Import) for alias in n.names)
