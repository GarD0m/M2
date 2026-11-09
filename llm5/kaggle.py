import shutil
from pathlib import Path

from llm5.cfg import configs
from llm5.start import setup, train


def main():
    setup()
    import jax
    from llm5.ckpt import Store, ready

    class Local(Store):
        def saved(self, name, step):
            if not ready(self.ckpt(name, step)):
                return
            for previous in self.cfgs:
                if previous == name:
                    break
                for path in (self.dir(previous) / "checkpoints").glob("[0-9]*"):
                    if path.name.isdigit() and path.is_dir():
                        shutil.rmtree(path)

        def finish(self, name):
            pass

    cfgs = [dict(cfg, max_num_checkpoints_to_keep=1)
            for cfg in configs(jax.device_count(), Path("/kaggle/working"))]
    train(cfgs, Local(cfgs))


if __name__ == "__main__":
    main()
