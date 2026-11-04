import ast
import os
import pytest
from llm5.plan import ROOT, generate, read_json


def test_all_project_configs_are_json(tmp_path):
    assert not list((ROOT / "configs").glob("*.yml"))
    assert not list((ROOT / "configs").glob("*.yaml"))
    for path in (ROOT / "configs").glob("*.json"):
        assert isinstance(read_json(path), dict)
    manifest = generate(tmp_path, "data/packed-512k", "models/tokenizer", "runs/checkpoints", "test")
    for stage in manifest["stages"]:
        assert stage["config"].endswith(".json")
        cfg = read_json(stage["config"])
        assert cfg["base_config"] == str(ROOT / "configs/maxtext-base.json")


def test_native_config_inheritance_reads_json(tmp_path):
    from pathlib import Path
    omegaconf = pytest.importorskip("omegaconf")
    source = ROOT / "upstream/maxtext/src/maxtext/configs/pyconfig.py"
    if not source.exists():
        pytest.skip("Native config loader source unavailable")
    module = ast.parse(source.read_text(encoding="utf-8"))
    loader = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "_load_config")
    namespace = {"os": os, "omegaconf": omegaconf, "_BASE_CONFIG_ATTR": "base_config",
                 "MAXTEXT_CONFIGS_DIR": str(ROOT / "configs")}
    exec(compile(ast.Module(body=[loader], type_ignores=[]), str(source), "exec"), namespace)
    manifest = generate(tmp_path, "data/packed-512k", "models/tokenizer", "runs/checkpoints", "test")
    native = namespace["_load_config"](manifest["stages"][0]["config"])
    assert native.load_full_state_path == ""
    assert native.weight_dtype == "bfloat16"
    assert native.base_num_decoder_layers == 5
    assert native.use_tokamax_splash is True
    assert native.lr_schedule_type == "wsd"
    assert Path(native.base_config).suffix == ".json"
