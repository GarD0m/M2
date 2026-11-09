import hashlib
import json
import os
import random
from itertools import accumulate
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from llm5.cfg import DATA, STAGES

DOMAINS = tuple(DATA)
CURSOR = Path("iter/process_0-of-1.json")


def setup():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ.setdefault("RAYON_NUM_THREADS", str(min(8, os.cpu_count() or 1)))
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from datasets import disable_progress_bars
    from huggingface_hub.utils import disable_progress_bars as disable_hub
    disable_progress_bars()
    disable_hub()


def tuples(value):
    return tuple(map(tuples, value)) if isinstance(value, list) else value


def heldout(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % 1000 < 5


def open_data(repo, rev, name, split):
    from datasets import load_dataset
    from pyarrow.dataset import ParquetFragmentScanOptions

    return load_dataset(repo, name, split=split, revision=rev, streaming=True,
                        columns=["content"], batch_size=16,
                        fragment_scan_options=ParquetFragmentScanOptions(pre_buffer=False))


class Stream:
    def __init__(self, domain, train, tok, seed):
        self.repo, self.rev, self.subsets = DATA[domain]
        self.train, self.tok = train, tok
        self.eos = tok.token_to_id("<|endoftext|>")
        self.rng = random.Random(seed)
        self.buf = np.empty(0, np.int32)
        self.data, self.it, self.pending = {}, {}, {}
        self.done = set()
        self.refresh()

    def refresh(self):
        self.active = [i for i in range(len(self.subsets)) if i not in self.done]
        self.weights = list(accumulate(self.subsets[i][2] for i in self.active))

    def text(self):
        while self.active:
            i = self.rng.choices(self.active, cum_weights=self.weights)[0]
            if i not in self.it:
                name, split, _ = self.subsets[i]
                ds = open_data(self.repo, self.rev, name, split)
                if str(i) in self.pending:
                    ds.load_state_dict(self.pending.pop(str(i)))
                self.data[i], self.it[i] = ds, iter(ds)
            row = next(self.it[i], None)
            if row is None:
                self.done.add(i)
                self.refresh()
            else:
                text = row["content"]
                if text.strip() and heldout(text) != self.train:
                    return text
        raise RuntimeError(f"Dataset exhausted: {self.repo}")

    def take(self, count):
        out = np.empty(count, np.int32)
        pos = 0
        while pos < count:
            if not len(self.buf):
                batch = [self.text() for _ in range(256)]
                ids = []
                for item in self.tok.encode_batch_fast(batch, add_special_tokens=False):
                    ids.extend(item.ids)
                    ids.append(self.eos)
                ids = np.asarray(ids, np.int32)
                self.buf = ids[ids != 0]
            n = min(count - pos, len(self.buf))
            out[pos:pos+n] = self.buf[:n]
            self.buf = self.buf[n:]
            pos += n
        return out

    def state(self):
        return dict(rng=self.rng.getstate(), buf=self.buf.tolist(), done=sorted(self.done),
                    data={**self.pending, **{str(i): ds.state_dict() for i, ds in self.data.items()}})

    def load(self, state):
        self.close()
        self.rng.setstate(tuples(state["rng"]))
        self.buf = np.asarray(state["buf"], np.int32)
        self.done = set(state["done"])
        self.pending = state["data"]
        self.refresh()

    def close(self):
        for it in self.it.values():
            close = getattr(it, "close", None)
            if close:
                close()
        self.it.clear()
        self.data.clear()


class Batches:
    def __init__(self, cfg, train, tok):
        length = cfg.max_target_length
        batch = cfg.global_batch_size_to_load if train else cfg.global_batch_size_to_load_eval
        self.shape = (int(batch), length)
        self.pos = np.broadcast_to(np.arange(length, dtype=np.int32), self.shape)
        self.ones = np.broadcast_to(np.int32(1), self.shape)
        spec = next(s for s in STAGES if s[0] == length)
        math, code = spec[-2:] if train else (.1, .2)
        self.weights = (1 - math - code, math, code) if train else (.7, .1, .2)
        seed = cfg.data_shuffle_seed + (0 if train else 10000)
        self.rng = random.Random(seed)
        self.streams = [Stream(name, train, tok, seed+i) for i, name in enumerate(DOMAINS)]

    def __iter__(self):
        return self

    def __next__(self):
        batch, length = self.shape
        rows = [[], [], []]
        for i, domain in enumerate(self.rng.choices(range(3), weights=self.weights, k=batch)):
            rows[domain].append(i)
        x = np.empty(self.shape, np.int32)
        for indices, stream in zip(rows, self.streams):
            if indices:
                x[indices] = stream.take(len(indices) * length).reshape(-1, length)
        y = np.empty_like(x)
        y[:, :-1] = x[:, 1:]
        y[:, -1] = 0
        mask = np.empty_like(y)
        np.not_equal(y, 0, out=mask)
        return dict(inputs=x, targets=y, inputs_position=self.pos, targets_position=self.pos,
                    inputs_segmentation=self.ones, targets_segmentation=mask)

    def get_state(self):
        return json.dumps([self.rng.getstate(), [s.state() for s in self.streams]], separators=(",", ":")).encode()

    def set_state(self, raw):
        rng, states = json.loads(raw)
        self.rng.setstate(tuples(rng))
        for stream, state in zip(self.streams, states):
            stream.load(state)

    def close(self):
        for stream in self.streams:
            stream.close()


class Data:
    def __init__(self, cfg, train, tok):
        self.cfg, self.train, self.tok = cfg, train, tok
        self.it = None
        self.cache = []

    def __iter__(self):
        if not self.train and self.cfg.eval_steps > 0:
            return self.eval()
        if self.it is not None:
            self.it.close()
        self.it = Batches(self.cfg, self.train, self.tok)
        if self.train and self.cfg.load_full_state_path:
            self.it.set_state((Path(self.cfg.load_full_state_path).parent / CURSOR).read_bytes())
        return self.it

    def eval(self):
        if not self.cache:
            self.it = Batches(self.cfg, False, self.tok)
            try:
                self.cache = [next(self.it) for _ in range(self.cfg.eval_steps)]
            finally:
                self.it.close()
                self.it = None
        yield from self.cache


def patch():
    from maxtext.input_pipeline import input_pipeline_interface as interface
    from maxtext.input_pipeline.multihost_dataloading import MultiHostDataLoadIterator

    def create(cfg, mesh):
        tok = Tokenizer.from_file(str(Path(cfg.tokenizer_path) / "tokenizer.json"))
        train = MultiHostDataLoadIterator(Data(cfg, True, tok), mesh)
        val = MultiHostDataLoadIterator(Data(cfg, False, tok), mesh) if cfg.eval_interval > 0 else None
        return train, val

    interface.create_data_iterator = create
