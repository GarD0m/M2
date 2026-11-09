from llm5.train import run, setup


def main():
    setup()
    import jax
    from llm5.cfg import configs
    from llm5.hub import Hub

    cfgs = list(configs(jax.device_count()))
    with Hub(cfgs) as hub:
        step, name = hub.restore()
        for cfg in cfgs:
            if cfg["steps"] <= step:
                continue
            if cfg["run_name"] == name:
                cfg["load_full_state_path"] = ""
            run(cfg, hub)
            hub.send(cfg["run_name"])
            hub.wait()


if __name__ == "__main__":
    main()
