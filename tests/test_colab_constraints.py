import runpy
from llm5.plan import ROOT, MAXTEXT_VERSION


def test_requirements_use_release_versions_and_portable_sources():
    requirements = (ROOT / "requirements.txt").read_text()
    assert f"maxtext=={MAXTEXT_VERSION}" in requirements
    assert "maxtext[tpu]" not in requirements
    assert "file://" not in requirements and "git+" not in requirements
    assert "jax==0.11.2" in requirements and "mpmath==1.3.0" in requirements


def test_colab_install_uses_version_and_never_clones_maxtext():
    block = runpy.run_path(str(ROOT / "scripts/build_notebook.py"))["BLOCKS"][1][1]
    assert "%pip install -r requirements.txt" in block
    assert "git clone" not in block and "archive/" not in block


def test_runtime_uses_release_version_without_model_check(monkeypatch):
    import sys
    import importlib.metadata
    from types import SimpleNamespace
    from llm5.launch import check_runtime
    monkeypatch.setattr(importlib.metadata, "version", lambda name: MAXTEXT_VERSION)
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(devices=lambda: [
        SimpleNamespace(platform="tpu", device_kind="any TPU model") for _ in range(8)]))
    assert check_runtime() == 8
