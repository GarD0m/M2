import os
from datetime import datetime

from llm5.cfg import RUN, STAGES, configs
from llm5.train import run, setup


def main():
    root = RUN / "profiles"
    name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    hlo = root / f"{name}-1k" / "hlo"
    flags = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = f'{flags} --xla_dump_to="{hlo.as_posix()}" --xla_dump_hlo_as_text --xla_dump_hlo_as_proto'
    setup()
    import jax

    jax.config.update("jax_enable_compilation_cache", False)
    cfg = next(configs(jax.device_count(), root, name, STAGES[:1]))
    cfg.update(steps=2, learning_rate_schedule_steps=100, warmup_steps_fraction=.6401,
               profiler="xplane", profile_cleanly=True, skip_first_n_steps_for_profiler=1, profiler_steps=1,
               enable_checkpointing=False, save_checkpoint_on_completion=False, eval_interval=-1)
    run(cfg)


if __name__ == "__main__":
    main()
