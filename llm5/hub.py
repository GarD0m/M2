"""Background Hugging Face checkpoint uploads and full-state recovery."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, hf_hub_download, snapshot_download, get_token
from huggingface_hub.utils import disable_progress_bars
from huggingface_hub.errors import EntryNotFoundError
from tqdm.auto import tqdm
from llm5.plan import ROOT, digest, read_json


class CheckpointHub:
    def __init__(self, repo, manifest):
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        disable_progress_bars()
        token = get_token()
        if not token:
            raise ValueError("Set HF_TOKEN before starting checkpoint uploads")
        self.api = HfApi(token=token)
        self.repo, self.manifest = repo, manifest
        self.configs = [read_json(s["config"]) for s in manifest["stages"]]
        self.indices = {c["run_name"]: i for i, c in enumerate(self.configs)}
        prefix = self.configs[0]["run_name"].removesuffix("-" + manifest["stages"][0]["name"])
        self.latest_path = f"runs/{prefix}/latest.json"
        if any(c["base_output_directory"].startswith("gs://") for c in self.configs):
            raise ValueError("Hugging Face checkpoint sync requires a local --output-root")
        self.fingerprints = {k: manifest[k] for k in
                             ("curriculum_sha256", "sources_sha256", "model_sha256", "base_sha256", "maxtext_version")}
        self.fingerprints["tokenizer_sha256"] = digest(Path(self.configs[0]["tokenizer_path"]) / "tokenizer.json")
        self.fingerprints["input_pipeline"] = manifest["input_pipeline"]
        self.api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-checkpoint")
        self.futures, self.queued = [], set()
        self.assets_uploaded = False
        self.writers = []
        self.latest = None

    def directory(self, index, step):
        cfg = self.configs[index]
        return Path(cfg["base_output_directory"]) / cfg["run_name"] / "checkpoints" / str(step)

    def validate_cursor(self, index, step):
        path = self.directory(index, step) / "iter/process_0-of-1.json"
        if not path.exists() or read_json(path).get("format") != self.manifest["input_pipeline"]:
            raise ValueError(f"Checkpoint has no compatible HF streaming cursor: {path}")

    def logdir(self, index):
        cfg = self.configs[index]
        return Path(cfg["base_output_directory"]) / cfg["run_name"] / "tensorboard"

    def snapshot_logs(self, destination):
        # Event files are append-only: copy flushed bytes, never hard-link them.
        for writer in self.writers:
            writer.flush()
        for index, cfg in enumerate(self.configs):
            source = self.logdir(index)
            if source.exists():
                shutil.copytree(source, destination / cfg["run_name"], dirs_exist_ok=True)

    def restore_logs(self, latest):
        revision = latest.get("tensorboard_commit", latest["checkpoint_commit"])
        with TemporaryDirectory(prefix=".hf-tensorboard-") as temporary:
            snapshot_download(self.repo, repo_type="model", revision=revision,
                              allow_patterns=["tensorboard/**"], local_dir=temporary, token=self.api.token)
            for index, cfg in enumerate(self.configs):
                source = Path(temporary) / "tensorboard" / cfg["run_name"]
                if source.exists():
                    for file in source.rglob("*"):
                        if file.is_file():
                            target = self.logdir(index) / file.relative_to(source)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            if not target.exists() or target.stat().st_size < file.stat().st_size:
                                shutil.copy2(file, target)

    def publish_latest(self, latest):
        payload = json.dumps(latest, indent=2).encode()
        for path in (self.latest_path, "latest.json"):
            self.api.upload_file(repo_id=self.repo, repo_type="model", path_in_repo=path,
                                 path_or_fileobj=payload,
                                 commit_message=f"Latest checkpoint: {latest['run_name']} step {latest['step']}")
        self.latest = latest

    def sync_logs(self):
        self.check()
        temporary = TemporaryDirectory(prefix=".hf-tensorboard-")
        snapshot = Path(temporary.name) / "tensorboard"
        try:
            self.snapshot_logs(snapshot)
        except BaseException:
            temporary.cleanup()
            raise
        self.futures.append(self.executor.submit(self._upload_logs, snapshot, temporary))

    def _upload_logs(self, snapshot, temporary):
        try:
            if snapshot.exists() and self.latest is not None:
                commit = self.api.upload_folder(repo_id=self.repo, repo_type="model", folder_path=str(snapshot),
                                               path_in_repo="tensorboard", commit_message="Save TensorBoard events")
                self.publish_latest({**self.latest, "tensorboard_commit": commit.oid})
        finally:
            temporary.cleanup()

    def check(self):
        for future in self.futures:
            if future.done():
                future.result()

    def wait(self):
        for future in self.futures:
            future.result()
        self.futures.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                self.wait()
        finally:
            self.executor.shutdown(wait=True)

    def restore_latest(self):
        """Prefer a newer local checkpoint, otherwise recover the committed Hub snapshot."""
        local = []
        for index, stage in enumerate(self.manifest["stages"]):
            root = self.directory(index, 0).parent
            if root.exists():
                local.extend((int(p.name), index) for p in root.iterdir()
                             if p.name.isdigit() and (p / "items").is_dir()
                             and stage["start_step"] <= int(p.name) <= stage["end_step"])
        latest = None
        try:
            file = hf_hub_download(self.repo, self.latest_path, repo_type="model", token=self.api.token)
            latest = read_json(file)
        except EntryNotFoundError:
            try:
                file = hf_hub_download(self.repo, "latest.json", repo_type="model", token=self.api.token)
                legacy = read_json(file)
                if legacy.get("run_name") in self.indices:
                    latest = legacy
            except EntryNotFoundError:
                pass
        if latest is not None:
            if latest.get("run_name") not in self.indices:
                raise ValueError("Hub latest checkpoint belongs to another run prefix; use the matching --run-prefix")
            if latest.get("fingerprints") != self.fingerprints:
                raise ValueError("Hub checkpoint model/curriculum/tokenizer differs from this run")
            index = self.indices[latest["run_name"]]
            stage = self.manifest["stages"][index]
            step = latest["step"]
            if not isinstance(step, int) or not stage["start_step"] <= step <= stage["end_step"]:
                raise ValueError("Invalid checkpoint step in Hub latest.json")
            self.latest = latest
            self.restore_logs(latest)
            if not local or max(local)[0] < step:
                target = self.directory(index, step)
                target.parent.mkdir(parents=True, exist_ok=True)
                remote = f"runs/{latest['run_name']}/checkpoints/{step}"
                tqdm.write(f"Restoring {self.repo}: {latest['run_name']} step {step}")
                with TemporaryDirectory(prefix=".hf-restore-", dir=target.parent) as temporary:
                    snapshot_download(self.repo, repo_type="model", revision=latest["checkpoint_commit"],
                                      allow_patterns=[f"{remote}/**"], local_dir=temporary, token=self.api.token)
                    source = Path(temporary) / remote
                    if not (source / "items").is_dir():
                        raise ValueError("Hub checkpoint has no full-state items directory")
                    os.replace(source, target)
                self.validate_cursor(index, step)
                return {"index": index, "step": step}
        if local:
            step, index = max(local)
            self.validate_cursor(index, step)
            self.enqueue(index, step)
            tqdm.write(f"Resuming local {self.configs[index]['run_name']} step {step}")
            return {"index": index, "step": step}
        tqdm.write("No saved checkpoint found; starting from step 0")
        return None

    def enqueue(self, index, step):
        self.check()
        if (index, step) in self.queued:
            return
        source = self.directory(index, step)
        if not (source / "items").is_dir():
            raise FileNotFoundError(f"Completed checkpoint missing: {source}")
        temporary = TemporaryDirectory(prefix=".hf-upload-", dir=source.parent)
        snapshot = Path(temporary.name) / "checkpoint"
        def link_or_copy(src, dst):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        try:
            # Hard links preserve pending data even when Orbax prunes its old paths.
            shutil.copytree(source, snapshot, copy_function=link_or_copy)
            self.snapshot_logs(Path(temporary.name) / "tensorboard")
        except BaseException:
            temporary.cleanup()
            raise
        self.queued.add((index, step))
        self.futures.append(self.executor.submit(self._upload, index, step, snapshot, temporary))

    def _upload(self, index, step, snapshot, temporary):
        try:
            if not self.assets_uploaded:
                self.api.upload_folder(repo_id=self.repo, repo_type="model", folder_path=self.configs[0]["tokenizer_path"],
                                       path_in_repo="tokenizer", commit_message="Upload training tokenizer")
                self.api.upload_folder(repo_id=self.repo, repo_type="model", folder_path=str(ROOT / "configs"),
                                       path_in_repo="configs", commit_message="Upload training configuration")
                self.api.upload_folder(repo_id=self.repo, repo_type="model", folder_path=str(ROOT / "LICENSES"),
                                       path_in_repo="LICENSES", commit_message="Upload configuration license")
                self.assets_uploaded = True
            name = self.configs[index]["run_name"]
            tqdm.write(f"Uploading {name} checkpoint {step} to {self.repo}")
            logs = Path(temporary.name) / "tensorboard"
            if logs.exists():
                self.api.upload_folder(repo_id=self.repo, repo_type="model", folder_path=str(logs),
                                       path_in_repo="tensorboard", commit_message=f"TensorBoard at {name} step {step}")
            commit = self.api.upload_folder(repo_id=self.repo, repo_type="model", folder_path=str(snapshot),
                                           path_in_repo=f"runs/{name}/checkpoints/{step}",
                                           commit_message=f"Save {name} step {step}")
            latest = dict(run_name=name, step=step, checkpoint_commit=commit.oid, fingerprints=self.fingerprints)
            self.publish_latest(latest)
            tqdm.write(f"Uploaded {name} step {step}")
        finally:
            temporary.cleanup()

    @contextmanager
    def capture(self):
        from maxtext.common import checkpointing
        from maxtext.utils import max_utils
        from tensorboardX import SummaryWriter
        native_save = checkpointing.save_checkpoint
        native_maybe = checkpointing.maybe_save_checkpoint
        native_writer = max_utils.initialize_summary_writer
        native_close = max_utils.close_summary_writer
        def writer(tensorboard_dir, run_name, enable_tensorboard=True):
            if not enable_tensorboard:
                return native_writer(tensorboard_dir, run_name, enable_tensorboard)
            result = SummaryWriter(str(self.logdir(self.indices[run_name])), flush_secs=10)
            self.writers.append(result)
            return result
        def close_writer(writer):
            try:
                return native_close(writer)
            finally:
                if writer in self.writers:
                    self.writers.remove(writer)
        def save(manager, step, state, config=None, data_iterator=None, force=False):
            self.check()
            saved = native_save(manager, step, state, config, data_iterator, force)
            if saved:
                checkpointing.wait_until_finished(manager)
                self.enqueue(self.indices[config.run_name], int(step))
            return saved
        def maybe(manager, state, config, iterator, step=None):
            self.check()
            if manager is None:
                return
            if step is None:
                value = state.optimizer.step
                completed = int(value.get_value() if hasattr(value, "get_value") else value)
                if completed > 0 and not self.directory(self.indices[config.run_name], completed).exists():
                    save(manager, completed, state, config, iterator, force=True)
                return
            completed = int(step) + 1
            return native_maybe(manager, state, config, iterator, step=completed)
        checkpointing.save_checkpoint, checkpointing.maybe_save_checkpoint = save, maybe
        max_utils.initialize_summary_writer, max_utils.close_summary_writer = writer, close_writer
        try:
            yield
        finally:
            checkpointing.save_checkpoint, checkpointing.maybe_save_checkpoint = native_save, native_maybe
            max_utils.initialize_summary_writer, max_utils.close_summary_writer = native_writer, native_close
            for active in self.writers:
                active.flush()
                active.close()
            self.writers.clear()
