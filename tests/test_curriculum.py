import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from llm5.plan import ROOT, curriculum, generate, read_yaml
from llm5.data import TokenStream, domain_order, is_eval, write_shards
from llm5.tokenizer import build_bpe
from llm5.tune import is_oom
from llm5.launch import check_data, load_manifest


def test_curriculum_preserves_batch_and_step_continuity(tmp_path):
    m = generate(tmp_path / "plan", tmp_path / "data", tmp_path / "tok",
                 "gs://example-bucket/llm5", "test", {"1k": 2, "2k": 1})
    stages = m["stages"]
    assert [s["length"] for s in stages] == [1024, 2048, 4096, 8192, 16384, 32768]
    assert abs(m["approximate_loss_tokens"] - 10_000_000_000) < 6 * 524288
    for i, s in enumerate(stages):
        cfg = read_yaml(s["config"])
        assert s["microbatch"] * s["accumulation"] == s["effective_batch"]
        assert int(cfg["warmup_steps_fraction"] * cfg["learning_rate_schedule_steps"]) == 64
        assert cfg["wsd_decay_steps_fraction"] == 0
        assert cfg["learning_rate_schedule_steps"] == stages[-1]["end_step"]
        if i:
            assert s["start_step"] == stages[i-1]["end_step"]
            assert cfg["load_full_state_path"].endswith(f"/{s['start_step']}/items")
            assert stages[i-1]["name"] in cfg["load_full_state_path"]
        else:
            assert cfg["load_full_state_path"] == ""
    assert m["parameters"] == 385899776
    assert load_manifest(tmp_path / "plan/manifest.json")["maxtext_commit"] == m["maxtext_commit"]


def test_rejects_invalid_curriculum():
    raw = read_yaml(ROOT / "configs/curriculum.yml")
    with pytest.raises(ValueError):
        curriculum(raw, {"1k": 3})
    bad = copy.deepcopy(raw)
    bad["stages"][0]["code"] = 1
    with pytest.raises(ValueError):
        curriculum(bad)


def test_domain_mix_is_by_equal_length_blocks():
    order, quotas = domain_order(10003, [0.6, 0.15, 0.25], 42)
    assert len(order) == sum(quotas) == 10003
    for quota, weight in zip(quotas, [0.6, 0.15, 0.25]):
        assert abs(quota / len(order) - weight) < 1 / len(order)
    assert np.array_equal(order, domain_order(10003, [0.6, 0.15, 0.25], 42)[0])


@pytest.fixture
def tokenizer(tmp_path):
    texts = ["繁體中文 English emoji 😀\n", "def f(x):\n    return x**2\n", "0123456789 +-*/"] * 20
    return build_bpe(texts, tmp_path / "tok", vocab_size=300, min_frequency=1)


def test_bpe_roundtrip_and_stream_boundaries(tokenizer):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_str(tokenizer.backend_tokenizer.to_str())
    for text in ["繁體中文\t😀", "print('hi')\n    pass", "任意 UTF-8: 𠮷"]:
        assert tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)) == text
    assert tok.token_to_id("<|pad|>") == 0
    docs = ["hello", "world", "中文"]
    expected = []
    for text in docs:
        expected.extend(tok.encode(text).ids + [tok.token_to_id("<|endoftext|>")])
    stream = TokenStream(iter(docs), tok)
    assert stream.take(3) + stream.take(len(expected) - 3) == expected
    with pytest.raises(RuntimeError, match="exhausted"):
        stream.take(1)


def test_corpus_too_small_does_not_fake_64k(tmp_path):
    with pytest.raises(ValueError, match="increase the training corpus"):
        build_bpe(["abc"], tmp_path / "tiny", vocab_size=65536)


def test_packed_parquet_roundtrip(tmp_path, tokenizer):
    import pyarrow.parquet as pq
    tok = tokenizer.backend_tokenizer
    streams = {d: TokenStream(iter([f"{d} sample " * 40] * 50), tok)
               for d in ("general", "math", "code")}
    counts = write_shards(tmp_path, "train", 20, 128, streams, [0.6, 0.15, 0.25], 42)
    rows = pq.read_table(next(tmp_path.glob("*.parquet"))).column("text").to_pylist()
    assert counts == {"general": 12, "math": 3, "code": 5}
    assert len(rows) == 20 and all(len(r) == 128 for r in rows)
    assert all(0 < min(r) <= max(r) < 300 for r in rows)
    assert not list(tmp_path.glob("*.partial"))


def test_holdout_deterministic():
    results = [is_eval(f"doc {i}") for i in range(10000)]
    assert 10 < sum(results) < 100
    assert results == [is_eval(f"doc {i}") for i in range(10000)]


def test_oom_only_retry():
    assert is_oom("RESOURCE_EXHAUSTED: failed allocation")
    assert not is_oom("TypeError: invalid optimizer config")


UPSTREAM = ROOT / "upstream/maxtext/src/maxtext"


@pytest.mark.skipif(not UPSTREAM.exists(), reason="Pinned upstream checkout unavailable")
def test_config_keys_and_native_shift():
    base = read_yaml(UPSTREAM / "configs/base.yml")
    cfg = read_yaml(ROOT / "configs/model.yml")
    assert not (cfg.keys() - base.keys() - {"base_config"})
    src = UPSTREAM / "input_pipeline/input_pipeline_utils.py"
    module = ast.parse(src.read_text(encoding="utf-8"))
    funcs = [n for n in module.body if isinstance(n, ast.FunctionDef)
             and n.name in ("shift_left", "shift_and_refine")]
    ns = {"np": np}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(src), "exec"), ns)
    batch = {"targets": np.array([[1, 3, 4, 2]], dtype=np.int32),
             "targets_segmentation": np.ones((1, 4), dtype=np.int32)}
    shifted = ns["shift_and_refine"](batch, [0])
    assert shifted["targets"].tolist() == [[3, 4, 2, 0]]
    assert shifted["targets_segmentation"].sum() == 3


@pytest.mark.skipif(not UPSTREAM.exists(), reason="Pinned upstream checkout unavailable")
def test_native_wso_schedule_has_no_decay():
    # Execute the pinned upstream function with small NumPy schedule adapters.
    # This checks its actual boundary construction without installing TPU JAX on Windows.
    def linear_schedule(init_value, end_value, transition_steps):
        return lambda step: init_value + (end_value-init_value)*np.clip(step/transition_steps, 0, 1)

    def join_schedules(pieces, boundaries):
        def schedule(step):
            index = sum(step >= b for b in boundaries[:len(pieces)-1])
            offset = boundaries[index-1] if index else 0
            return pieces[index](step-offset)
        return schedule

    optax = SimpleNamespace(linear_schedule=linear_schedule,
                            constant_schedule=lambda value: lambda step: value,
                            join_schedules=join_schedules)
    types = SimpleNamespace(LearningRateScheduleType=SimpleNamespace(COSINE="cosine"),
                            WsdDecayStyle=SimpleNamespace(LINEAR="linear"))
    src = UPSTREAM / "utils/maxtext_utils.py"
    module = ast.parse(src.read_text(encoding="utf-8"))
    func = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                and n.name == "create_learning_rate_schedule")
    ns = {"jnp": np, "optax": optax, "types": types}
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(src), "exec"), ns)
    raw = read_yaml(ROOT / "configs/curriculum.yml")
    stages = curriculum(raw)
    total_steps = stages[-1]["end_step"]
    warmup = raw["warmup_steps"]
    for stage in stages:
        cfg = SimpleNamespace(learning_rate=.001, learning_rate_final_fraction=1.,
                              learning_rate_schedule_steps=total_steps, warmup_steps_fraction=(warmup+.01)/total_steps,
                              steps=stage["end_step"], lr_schedule_type="wsd", wsd_decay_steps_fraction=0.)
        schedule = ns["create_learning_rate_schedule"](cfg)
        assert schedule(0) == 0
        assert schedule(warmup//2) == pytest.approx(.0005)
        assert schedule(warmup) == pytest.approx(.001)
        assert schedule(stage["end_step"]-1) == pytest.approx(.001)
