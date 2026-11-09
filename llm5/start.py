from llm5.train import run, setup


def train(cfgs, store):
    step, name = store.restore()
    for cfg in cfgs:
        if cfg["steps"] <= step:
            continue
        if cfg["run_name"] == name:
            cfg = dict(cfg, load_full_state_path="")
        run(cfg, store)
        store.finish(cfg["run_name"])


def main():
    setup()
    import jax
    from llm5.cfg import configs
    from llm5.hub import Hub

    cfgs = list(configs(jax.device_count()))
    with Hub(cfgs) as hub:
        train(cfgs, hub)


if __name__ == "__main__":
    main()
