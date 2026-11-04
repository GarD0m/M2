import pytest
from llm5.plan import ROOT, read_yaml, curriculum, generate
from llm5.tune import candidates


def test_eight_devices_keep_global_batch_and_data_budget(tmp_path):
    raw = read_yaml(ROOT / "configs/curriculum.yml")
    single = curriculum(raw)
    multi = curriculum(raw, device_count=8)
    for a, b in zip(single, multi):
        assert b["per_device_microbatch"] * 8 * b["accumulation"] * b["length"] == 524288
        assert a["sequences"] == b["sequences"]
        assert a["num_steps"] == b["num_steps"]
    assert multi[-1]["microbatch"] == 8
    assert multi[-1]["per_device_microbatch"] == 1
    assert multi[-1]["accumulation"] == 2
    m = generate(tmp_path / "plan", tmp_path / "data", tmp_path / "tok",
                 "gs://example/m2", "test", device_count=8)
    for stage in m["stages"]:
        cfg = read_yaml(stage["config"])
        assert cfg["ici_data_parallelism"] == 8
        assert cfg["per_device_batch_size"]*8*cfg["gradient_accumulation_steps"]*stage["length"] == 524288
    assert candidates(16, 32768, device_count=8) == [8, 16]


def test_invalid_global_microbatch_rejected():
    raw = read_yaml(ROOT / "configs/curriculum.yml")
    with pytest.raises(ValueError):
        curriculum(raw, {"32k": 1}, device_count=8)
