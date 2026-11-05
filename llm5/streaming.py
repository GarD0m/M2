"""Checkpointable HF streaming -> batched BPE -> native MaxText batches."""
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
from tokenizers import Tokenizer
from llm5.plan import ROOT, read_json

FORMAT = "hf-stream-v1"
DOMAINS = ("general", "math", "code")
_BATCH_SIZE = 256
_CONFIGURED = False


def configure(batch_size=256, threads=None):
    global _BATCH_SIZE, _CONFIGURED
    threads = min(8, os.cpu_count() or 1) if threads is None else threads
    if batch_size < 1 or threads < 1:
        raise ValueError("Tokenizer batch size and threads must be positive")
    _BATCH_SIZE = batch_size
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ.setdefault("RAYON_NUM_THREADS", str(threads))
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from datasets import disable_progress_bars
    from huggingface_hub.utils import disable_progress_bars as disable_hub_bars
    disable_progress_bars()
    disable_hub_bars()
    _CONFIGURED = True


def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value, list) else value


def is_eval(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % 1000 < 5


class DomainStream:
    def __init__(self, domain, partition, tokenizer, seed):
        self.domain, self.partition, self.tokenizer = domain, partition, tokenizer
        self.eos_id = tokenizer.token_to_id("<|endoftext|>")
        self.source = read_json(ROOT / "configs/sources.json")[domain]
        self.rng = random.Random(seed)
        self.buffer = deque()
        self.datasets, self.iterators, self.pending = {}, {}, {}
        self.exhausted = set()
        self.documents_seen = 0

    def next_text(self):
        from datasets import load_dataset
        from pyarrow.dataset import ParquetFragmentScanOptions
        while True:
            active = [i for i in range(len(self.source["subsets"])) if i not in self.exhausted]
            if not active:
                raise RuntimeError(f"{self.domain} source exhausted; refusing to silently repeat data")
            index = self.rng.choices(active, weights=[self.source["subsets"][i]["weight"] for i in active])[0]
            if index not in self.datasets:
                sub = self.source["subsets"][index]
                ds = load_dataset(self.source["repo"], sub["config"], split=sub["split"],
                                  revision=self.source["revision"], streaming=True,
                                  columns=[self.source["column"]], batch_size=16,
                                  fragment_scan_options=ParquetFragmentScanOptions(pre_buffer=False))
                if str(index) in self.pending:
                    ds.load_state_dict(self.pending[str(index)])
                self.datasets[index], self.iterators[index] = ds, iter(ds)
            try:
                text = next(self.iterators[index])[self.source["column"]]
            except StopIteration:
                self.exhausted.add(index)
                continue
            if not isinstance(text, str):
                raise TypeError(f"{self.domain} content must be text")
            if text.strip() and is_eval(text) == (self.partition == "eval"):
                return text

    def take(self, count):
        while len(self.buffer) < count:
            batch = [self.next_text() for _ in range(_BATCH_SIZE)]
            self.documents_seen += len(batch)
            for encoded in self.tokenizer.encode_batch(batch, add_special_tokens=False):
                self.buffer.extend(t for t in encoded.ids if t != 0)
                self.buffer.append(self.eos_id)
        return np.fromiter((self.buffer.popleft() for _ in range(count)), dtype=np.int32, count=count)

    def get_state(self):
        positions = dict(self.pending)
        positions.update({str(i): ds.state_dict() for i, ds in self.datasets.items()})
        return dict(rng=self.rng.getstate(), buffer=list(self.buffer), datasets=positions,
                    exhausted=sorted(self.exhausted), documents_seen=self.documents_seen)

    def set_state(self, state):
        self.close()
        self.pending = state["datasets"]
        self.rng.setstate(tuples(state["rng"]))
        self.buffer = deque(state["buffer"])
        self.exhausted = set(state["exhausted"])
        self.documents_seen = state["documents_seen"]

    def close(self):
        for iterator in self.iterators.values():
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        self.datasets.clear()
        self.iterators.clear()


class StreamingIterator:
    def __init__(self, config, training):
        self.config, self.training = config, training
        self.length = config.max_target_length
        self.batch = int(config.global_batch_size_to_load if training else config.global_batch_size_to_load_eval)
        stage = next(s for s in read_json(ROOT / "configs/curriculum.json")["stages"] if s["length"] == self.length)
        self.weights = [1-stage["math"]-stage["code"], stage["math"], stage["code"]] if training else [.7, .1, .2]
        seed = config.data_shuffle_seed + (0 if training else 10000)
        self.rng = random.Random(seed)
        tok = Tokenizer.from_file(str(Path(config.tokenizer_path) / "tokenizer.json"))
        if tok.get_vocab_size() != config.vocab_size or tok.token_to_id("<|pad|>") != 0 or tok.token_to_id("<|endoftext|>") is None:
            raise ValueError("Tokenizer vocabulary, PAD or EOS differs from the model")
        self.streams = {d: DomainStream(d, "train" if training else "eval", tok, seed+i)
                        for i, d in enumerate(DOMAINS)}

    def __iter__(self):
        return self

    def __next__(self):
        rows = {domain: [] for domain in DOMAINS}
        for row, domain in enumerate(self.rng.choices(DOMAINS, weights=self.weights, k=self.batch)):
            rows[domain].append(row)
        inputs = np.empty((self.batch, self.length), dtype=np.int32)
        for domain, indices in rows.items():
            if indices:
                inputs[indices] = self.streams[domain].take(len(indices) * self.length).reshape(-1, self.length)
        targets = np.zeros_like(inputs)
        targets[:, :-1] = inputs[:, 1:]
        positions = np.broadcast_to(np.arange(self.length, dtype=np.int32), inputs.shape)
        return dict(inputs=inputs, targets=targets, inputs_position=positions, targets_position=positions,
                    inputs_segmentation=np.ones_like(inputs), targets_segmentation=(targets != 0).astype(np.int32))

    def get_state(self):
        return json.dumps(dict(format=FORMAT, length=self.length, batch=self.batch,
                               rng=self.rng.getstate(), streams={d: s.get_state() for d, s in self.streams.items()})).encode()

    def set_state(self, state, carry=False):
        state = json.loads(state)
        if state.get("format") != FORMAT:
            raise ValueError("Checkpoint iterator is not from the HF streaming pipeline")
        if not carry and (state["length"] != self.length or state["batch"] != self.batch):
            raise ValueError("Checkpoint stream batch/length differs from this stage")
        self.rng.setstate(tuples(state["rng"]))
        for domain, stream in self.streams.items():
            stream.set_state(state["streams"][domain])

    def close(self):
        for stream in self.streams.values():
            stream.close()


class StreamingDataset:
    def __init__(self, config, training):
        self.config, self.training = config, training
        self.iterator = None
        self.eval_cache = []

    def __iter__(self):
        if not self.training and self.config.eval_steps > 0:
            return self.cached_eval()
        self.close()
        iterator = StreamingIterator(self.config, self.training)
        if self.training and self.config.load_full_state_path:
            previous = Path(self.config.load_full_state_path).parent / "iter/process_0-of-1.json"
            if not previous.exists():
                raise FileNotFoundError(f"Previous streaming cursor missing: {previous}")
            iterator.set_state(previous.read_bytes(), carry=True)
        self.iterator = iterator
        return iterator

    def cached_eval(self):
        # Evaluation always starts at the same seed and held-out split.
        for index in range(self.config.eval_steps):
            if index == len(self.eval_cache):
                if self.iterator is None:
                    self.iterator = StreamingIterator(self.config, False)
                self.eval_cache.append(next(self.iterator))
                if len(self.eval_cache) == self.config.eval_steps:
                    self.iterator.close()
                    self.iterator = None
            yield self.eval_cache[index]

    def close(self):
        if self.iterator is not None:
            self.iterator.close()
            self.iterator = None
        self.eval_cache.clear()


def install_streaming_input():
    if not _CONFIGURED:
        configure()
    from maxtext.input_pipeline import input_pipeline_interface as interface
    from maxtext.input_pipeline.multihost_dataloading import MultiHostDataLoadIterator
    original = interface.create_data_iterator
    if getattr(original, "_llm5_streaming", False):
        return
    def create(config, mesh):
        if config.dataset_type != "grain":
            return original(config, mesh)
        import jax
        if jax.process_count() != 1:
            raise ValueError("This streaming pipeline uses one Python process")
        train = MultiHostDataLoadIterator(StreamingDataset(config, True), mesh)
        evaluation = MultiHostDataLoadIterator(StreamingDataset(config, False), mesh) if config.eval_interval > 0 else None
        return train, evaluation
    create._llm5_streaming = True
    interface.create_data_iterator = create
