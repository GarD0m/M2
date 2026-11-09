from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from llm5.data import CURSOR


def ready(path):
    return (path / "items").is_dir() and (path / CURSOR).is_file()


class Store:
    def __init__(self, cfgs):
        self.cfgs = {cfg["run_name"]: cfg for cfg in cfgs}
        self.writer = None

    def dir(self, name):
        return Path(self.cfgs[name]["base_output_directory"]) / name

    def ckpt(self, name, step):
        return self.dir(name) / "checkpoints" / str(step)

    def restore(self):
        return max(((int(p.name), name) for name in self.cfgs
                    for p in (self.dir(name) / "checkpoints").glob("[0-9]*")
                    if p.name.isdigit() and ready(p)),
                   default=(0, ""))

    @contextmanager
    def capture(self, name):
        from maxtext.common import checkpointing as cp
        from maxtext.utils import max_utils
        from tensorboardX import SummaryWriter

        save = cp.save_checkpoint
        maybe = cp.maybe_save_checkpoint
        close = max_utils.close_summary_writer

        def writer(directory, run_name, enable_tensorboard=True):
            if enable_tensorboard:
                self.writer = SummaryWriter(str(self.dir(run_name) / "tensorboard"), flush_secs=10)
            return self.writer

        def finish(active):
            try:
                return close(active)
            finally:
                self.writer = None

        def save_step(manager, step, state, config=None, data_iterator=None, force=False):
            saved = save(manager, step, state, config, data_iterator, force)
            if saved:
                cp.wait_until_finished(manager)
                self.saved(name, int(step))
            return saved

        def maybe_step(manager, state, config, iterator, step=None):
            if step is not None:
                return maybe(manager, state, config, iterator, step=int(step) + 1)
            completed = int(state.optimizer.step.get_value())
            if not self.ckpt(name, completed).is_dir():
                save_step(manager, completed, state, config, iterator, force=True)

        with (patch.object(cp, "save_checkpoint", save_step),
              patch.object(cp, "maybe_save_checkpoint", maybe_step),
              patch.object(max_utils, "initialize_summary_writer", writer),
              patch.object(max_utils, "close_summary_writer", finish)):
            try:
                yield
            finally:
                if self.writer is not None:
                    self.writer.close()
                    self.writer = None
