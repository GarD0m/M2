import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError
from tensorboardX import SummaryWriter

from llm5.cfg import REPO
from llm5.data import CURSOR


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class Hub:
    def __init__(self, cfgs):
        self.cfgs = {c["run_name"]: c for c in cfgs}
        self.api = HfApi(token=os.environ["HF_TOKEN"])
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.jobs = []
        self.writer = None
        self.index = "runs/m2/latest.json"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        try:
            if exc_type is None:
                self.wait()
        finally:
            self.pool.shutdown()

    def dir(self, name):
        return Path(self.cfgs[name]["base_output_directory"]) / name

    def ckpt(self, name, step):
        return self.dir(name) / "checkpoints" / str(step)

    def download(self, temp, paths, revision=None):
        return snapshot_download(REPO, repo_type="model", local_dir=temp,
                                 allow_patterns=paths, revision=revision, token=self.api.token)

    def restore(self):
        local = [(int(p.name), name) for name in self.cfgs
                 for p in (self.dir(name) / "checkpoints").glob("[0-9]*")
                 if p.name.isdigit() and (p / "items").is_dir() and (p / CURSOR).is_file()]
        best = max(local, default=(0, ""))
        try:
            path = hf_hub_download(REPO, self.index, repo_type="model", token=self.api.token)
        except EntryNotFoundError:
            remote = (0, "")
        else:
            latest = json.loads(Path(path).read_text(encoding="utf-8"))
            remote = latest["step"], latest["run_name"]
            with TemporaryDirectory() as temp:
                self.download(temp, ["tensorboard/m2-*/events.*"])
                for name in self.cfgs:
                    for src in (Path(temp) / "tensorboard" / name).glob("events.*"):
                        dst = self.dir(name) / "tensorboard" / src.name
                        if not dst.exists() or src.stat().st_size > dst.stat().st_size:
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(src, dst)
            if remote > best:
                step, name = remote
                dst = self.ckpt(name, step)
                dst.parent.mkdir(parents=True, exist_ok=True)
                key = f"runs/{name}/checkpoints/{step}"
                with TemporaryDirectory(dir=dst.parent) as temp:
                    self.download(temp, [f"{key}/**"], latest["checkpoint_commit"])
                    os.replace(Path(temp) / key, dst)
                return remote
        if best > remote:
            self.send(best[1], best[0])
        return best

    def drain(self):
        pending = []
        for job in self.jobs:
            if job.done():
                job.result()
            else:
                pending.append(job)
        self.jobs = pending

    def wait(self):
        for job in self.jobs:
            job.result()
        self.jobs.clear()

    def send(self, name, step=None):
        self.drain()
        src = self.dir(name)
        tmp = TemporaryDirectory(dir=src)
        snap = Path(tmp.name)
        try:
            if step is not None:
                shutil.copytree(self.ckpt(name, step), snap / "ckpt", copy_function=link)
            if self.writer is not None:
                self.writer.flush()
            logs = list((src / "tensorboard").glob("events.*"))
            if logs:
                (snap / "logs").mkdir()
                for log in logs:
                    shutil.copy2(log, snap / "logs" / log.name)
            self.jobs.append(self.pool.submit(self.upload, tmp, name, step))
        except BaseException:
            tmp.cleanup()
            raise

    def upload(self, tmp, name, step):
        try:
            snap = Path(tmp.name)
            if (snap / "logs").is_dir():
                self.api.upload_folder(repo_id=REPO, repo_type="model", folder_path=snap / "logs",
                                       path_in_repo=f"tensorboard/{name}", commit_message=f"Logs {name}")
            if step is not None:
                key = f"runs/{name}/checkpoints/{step}"
                commit = self.api.upload_folder(repo_id=REPO, repo_type="model", folder_path=snap / "ckpt",
                                                path_in_repo=key, commit_message=f"{name} {step}")
                meta = dict(run_name=name, step=step, checkpoint_commit=commit.oid)
                payload = json.dumps(meta).encode()
                self.api.create_commit(REPO, repo_type="model", commit_message=f"Latest {name} {step}",
                                       operations=[CommitOperationAdd(path, payload)
                                                   for path in (self.index, "latest.json")])
        finally:
            tmp.cleanup()

    @contextmanager
    def capture(self, name):
        from maxtext.common import checkpointing as cp
        from maxtext.utils import max_utils

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
            self.drain()
            saved = save(manager, step, state, config, data_iterator, force)
            if saved:
                cp.wait_until_finished(manager)
                self.send(name, int(step))
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
