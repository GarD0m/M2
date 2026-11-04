import ast
import json
from types import SimpleNamespace

import pytest

from llm5.plan import ROOT, read_json, curriculum, memory_components
from llm5.tune import candidates, steady_step_seconds
from llm5.train import install_fused_qkv_muon_support
from llm5.launch import command


def test_bf16_tokamax_and_batch_accounting():
    model = read_json(ROOT / "configs/model.json")
    assert model["weight_dtype"] == "bfloat16"
    assert model["use_tokamax_splash"] is True
    assert model["fused_qkv"] is True and model["fused_mlp"] is True
    assert [model[f"sa_{x}_layout"] for x in "qkv"] == ["SEQ_MINOR"]*3
    for stage in curriculum(read_json(ROOT / "configs/curriculum.json")):
        assert stage["microbatch"]*stage["length"] == 32768
        assert stage["accumulation"] == 16
        assert stage["effective_batch"] * stage["length"] == 524288
        sizes = memory_components(model, stage)
        assert sizes["bf16_weights_gib"] == pytest.approx(.7187943458557129)
        assert sizes["saved_qkv_gib"] == 1.25
        assert sizes["one_fp32_logits_tile_gib"] == .25


def test_timing_discards_compilation_and_rejects_missing_metrics(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n".join(json.dumps({"step": i, "perf/step_time_seconds": t})
                             for i, t in enumerate([1000, 100, 8, 4, 2, 3, 2])), encoding="utf-8")
    assert steady_step_seconds(path, 4) == 2
    with pytest.raises(ValueError):
        steady_step_seconds(path, 6)
    assert candidates(64) == [1, 2, 4, 8, 16, 32, 64]
    assert candidates(2) == [1, 2]
    assert candidates(512)[-1] == 128
    assert candidates(16, 32768) == [1, 2, 4]


def test_fused_qkv_muon_axes():
    src = ROOT / "upstream/maxtext/src/maxtext/utils/muon_utils.py"
    if not src.exists():
        pytest.skip("Pinned upstream unavailable")
    module = ast.parse(src.read_text(encoding="utf-8"))
    nodes = [n for n in module.body if isinstance(n, (ast.Assign, ast.FunctionDef))
             and (isinstance(n, ast.Assign) or n.name in ("transform_logic", "_is_path_contain_any"))]
    ns = {"Tuple": tuple, "Optional": __import__('typing').Optional,
          "mdn": lambda reduction, output: (reduction, output)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(src), "exec"), ns)
    stub = SimpleNamespace(ATTENTION_QKV_NAMES=ns["ATTENTION_QKV_NAMES"])
    install_fused_qkv_muon_support(stub)
    install_fused_qkv_muon_support(stub)
    assert stub.ATTENTION_QKV_NAMES.count("qkv_proj") == 1
    ns["ATTENTION_QKV_NAMES"] = stub.ATTENTION_QKV_NAMES
    assert ns["transform_logic"](("self_attention", "qkv_proj", "kernel"), (2048,32,128)) == ((0,),(-2,-1))
    # The activation axis in fused gate/up MLP remains a batch of two matrices.
    assert ns["transform_logic"](("mlp", "wi", "kernel"), (2048,2,6144)) == ((0,),(-1,))
    assert "llm5.train" in command("test.json")


def test_tune_prepares_default_plan_and_applies_choice(tmp_path, monkeypatch):
    from llm5.tune import prepare_plan
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "runs/plan-512k/manifest.json"
    manifest = prepare_plan(path, 8)
    assert path.exists()
    config = read_json(manifest["stages"][0]["config"])
    assert config["tokenizer_path"] == str(tmp_path / "models/tokenizer")
    selected = {s["name"]: s["effective_batch"] for s in manifest["stages"]}
    tuned = prepare_plan(path, 8, selected)
    for stage in tuned["stages"]:
        assert stage["accumulation"] == 1
        assert stage["per_device_microbatch"]*8*stage["length"] == 524288


def test_probe_json_preserves_empty_checkpoint_strings(tmp_path, monkeypatch):
    from llm5.tune import prepare_plan, write_probe_config
    monkeypatch.chdir(tmp_path)
    manifest = prepare_plan(tmp_path / "plan/manifest.json", 1)
    stage = manifest["stages"][1]
    original = read_json(stage["config"])
    assert original["load_full_state_path"]
    probe = tmp_path / "probe.json"
    write_probe_config(stage, probe, 1, 1, 12, tmp_path, "test", tmp_path / "metrics.jsonl")
    config = json.loads(probe.read_text())
    assert config["load_full_state_path"] == ""
    assert config["load_parameters_path"] == ""
    assert config["enable_checkpointing"] is False
    assert config["dataset_type"] == "synthetic"
    assert config["gradient_accumulation_steps"] == 256
    assert read_json(stage["config"]) == original
    # Validate through the real OmegaConf loader used by MaxText.
    omegaconf = pytest.importorskip("omegaconf")
    loaded = omegaconf.OmegaConf.load(probe)
    assert isinstance(loaded.load_full_state_path, str)
    assert loaded.load_full_state_path == ""
    assert loaded.enable_checkpointing is False


def test_json_entrypoint_does_not_use_cli_value_parsing():
    from llm5.train import install_json_config_support
    def native(argv, **kwargs):
        return "base.json", argv[1:]
    module = SimpleNamespace(_resolve_or_infer_config=native, resolve_config_path=lambda p: p)
    install_json_config_support(module)
    patched = module._resolve_or_infer_config
    install_json_config_support(module)
    assert patched is module._resolve_or_infer_config
    assert patched(["train", "probe.json"]) == ("probe.json", [])
    assert patched(["train", "probe.json", "steps=3"]) == ("probe.json", ["steps=3"])
    assert patched(["train", "other.txt"]) == native(["train", "other.txt"])
