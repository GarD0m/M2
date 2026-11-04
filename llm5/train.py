"""Native MaxText entry point with fused-QKV Muon dimension support."""
from functools import wraps
from pathlib import Path


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
    from absl import app
    from maxtext.trainers.pre_train import train
    app.run(train.main)


if __name__ == "__main__":
    main()
