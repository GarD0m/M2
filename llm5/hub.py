import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError

from llm5.cfg import NAME, REPO
from llm5.ckpt import Store


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


class Hub(Store):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        self.api = HfApi(token=os.environ["HF_TOKEN"])
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.jobs = []
        self.index = f"runs/{NAME}/latest.json"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        try:
            if exc_type is None:
                self.wait()
        finally:
            self.pool.shutdown()

    def download(self, temp, paths, revision=None):
        return snapshot_download(REPO, repo_type="model", local_dir=temp,
                                 allow_patterns=paths, revision=revision, token=self.api.token)

    def restore(self):
        best = super().restore()
        try:
            path = hf_hub_download(REPO, self.index, repo_type="model", token=self.api.token)
        except EntryNotFoundError:
            remote = (0, "")
        else:
            latest = json.loads(Path(path).read_text(encoding="utf-8"))
            remote = latest["step"], latest["run_name"]
            with TemporaryDirectory() as temp:
                self.download(temp, [f"tensorboard/{NAME}-*/events.*"])
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

    def saved(self, name, step):
        self.send(name, step)

    def finish(self, name):
        self.send(name)
        self.wait()

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
