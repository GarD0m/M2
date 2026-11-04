"""Background Hugging Face checkpoint uploads and full-state recovery."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from huggingface_hub import HfApi, hf_hub_download, snapshot_download, get_token
from huggingface_hub.errors import EntryNotFoundError
from tqdm.auto import tqdm
from llm5.plan import ROOT, digest, read_json


class CheckpointHub:
    def __init__(self, repo, manifest):
        token = get_token()
        if not token:
            raise ValueError("Set HF_TOKEN before starting checkpoint uploads")
        self.api = HfApi(token=token)
        self.repo, self.manifest = repo, manifest
        self.configs = [read_json(s["config"]) for s in manifest["stages"]]
        self.indices = {c["run_name"]: i for i, c in enumerate(self.configs)}
        if any(c["base_output_directory"].startswith("gs://") for c in self.configs):
            raise ValueError("Hugging Face checkpoint sync requires a local --output-root")
        self.fingerprints = {k: manifest[k] for k in
                             ("curriculum_sha256", "sources_sha256", "model_sha256", "base_sha256", "maxtext_version")}
        self.fingerprints["tokenizer_sha256"] = digest(Path(self.configs[0]["tokenizer_path"]) / "tokenizer.json")
        self.api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-checkpoint")
        self.futures, self.queued = [], set()
        self.assets_uploaded = False

    def directory(self, index, step):
        cfg = self.configs[index]
        return Path(cfg["base_output_directory"]) / cfg["run_name"] / "checkpoints" / str(step)

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
            file = hf_hub_download(self.repo, "latest.json", repo_type="model", token=self.api.token)
            latest = read_json(file)
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
                return {"index": index, "step": step}
        if local:
            step, index = max(local)
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
            commit = self.api.upload_folder(repo_id=self.repo, repo_type="model", folder_path=str(snapshot),
                                           path_in_repo=f"runs/{name}/checkpoints/{step}",
                                           commit_message=f"Save {name} step {step}")
            latest = dict(run_name=name, step=step, checkpoint_commit=commit.oid, fingerprints=self.fingerprints)
            self.api.upload_file(repo_id=self.repo, repo_type="model", path_in_repo="latest.json",
                                 path_or_fileobj=json.dumps(latest, indent=2).encode(),
                                 commit_message=f"Latest checkpoint: {name} step {step}")
            tqdm.write(f"Uploaded {name} step {step}")
        finally:
            temporary.cleanup()

    @contextmanager
    def capture(self):
        from maxtext.common import checkpointing
        native_save = checkpointing.save_checkpoint
        native_maybe = checkpointing.maybe_save_checkpoint
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
        try:
            yield
        finally:
            checkpointing.save_checkpoint, checkpointing.maybe_save_checkpoint = native_save, native_maybe
