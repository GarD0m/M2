"""Native MaxText entry point with fused-QKV Muon dimension support."""


def install_fused_qkv_muon_support(module=None):
    if module is None:
        from maxtext.utils import muon_utils as module
    # Fused QKV has shape [D, Hq+2*Hkv, head_dim]. Muon must reduce
    # over D and flatten both output axes, rather than batch over heads.
    if "qkv_proj" not in module.ATTENTION_QKV_NAMES:
        module.ATTENTION_QKV_NAMES += ("qkv_proj",)


def main():
    install_fused_qkv_muon_support()
    from absl import app
    from maxtext.trainers.pre_train import train
    app.run(train.main)


if __name__ == "__main__":
    main()
