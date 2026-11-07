"""Checkpointable HF streaming -> batched BPE -> native MaxText batches."""
import hashlib
from itertools import accumulate
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
_TRAIN_WEIGHTS = None


def configure(batch_size=256, threads=None, train_weights=None):
    global _BATCH_SIZE, _CONFIGURED, _TRAIN_WEIGHTS
    threads = min(8, os.cpu_count() or 1) if threads is None else threads
    if batch_size < 1 or threads < 1:
        raise ValueError("Tokenizer batch size and threads must be positive")
    _BATCH_SIZE = batch_size
    _TRAIN_WEIGHTS = train_weights
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


def load_tokenizer(directory, expected_vocab_size):
    path = Path(directory) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(path))
    vocab_size = tokenizer.get_vocab_size()
    pad = tokenizer.token_to_id("<|pad|>")
    eos = tokenizer.token_to_id("<|endoftext|>")
    if vocab_size != expected_vocab_size or pad != 0 or eos in (None, 0):
        raise ValueError(
            f"Tokenizer {path}: vocab={vocab_size} (model expects {expected_vocab_size}), "
            f"PAD={pad} (expected 0), EOS={eos} (expected a nonzero ID)"
        )
    return tokenizer


def open_subset(source, subset):
    from datasets import load_dataset
    from pyarrow.dataset import ParquetFragmentScanOptions

    return load_dataset(
        source["repo"], subset["config"], split=subset["split"],
        revision=source["revision"], streaming=True,
        columns=[source["column"]], batch_size=16,
        fragment_scan_options=ParquetFragmentScanOptions(pre_buffer=False),
    )


class DomainStream:
    def __init__(self, domain, partition, tokenizer, seed):
        self.domain, self.partition, self.tokenizer = domain, partition, tokenizer
        self.eos_id = tokenizer.token_to_id("<|endoftext|>")
        self.source = read_json(ROOT / "configs/sources.json")[domain]
        self.rng = random.Random(seed)
        self.buffer = np.empty(0, dtype=np.int32)
        self.datasets, self.iterators, self.pending = {}, {}, {}
        self.exhausted = set()
        self.update_subset_choices()
        self.documents_seen = 0

    def update_subset_choices(self):
        subsets = self.source["subsets"]
        self.active = [index for index in range(len(subsets)) if index not in self.exhausted]
        self.cumulative_weights = list(accumulate(subsets[index]["weight"] for index in self.active))

    def next_text(self):
        while True:
            if not self.active:
                raise RuntimeError(f"{self.domain} source exhausted; refusing to silently repeat data")
            index = self.rng.choices(self.active, cum_weights=self.cumulative_weights)[0]
            if index not in self.datasets:
                sub = self.source["subsets"][index]
                ds = open_subset(self.source, sub)
                if str(index) in self.pending:
                    ds.load_state_dict(self.pending[str(index)])
                self.datasets[index], self.iterators[index] = ds, iter(ds)
            try:
                text = next(self.iterators[index])[self.source["column"]]
            except StopIteration:
                self.exhausted.add(index)
                self.update_subset_choices()
                continue
            if not isinstance(text, str):
                raise TypeError(f"{self.domain} content must be text")
            if text.strip() and is_eval(text) == (self.partition == "eval"):
                return text

    def take(self, count):
        tokens = np.empty(count, dtype=np.int32)
        offset = 0
        while offset < count:
            size = min(len(self.buffer), count - offset)
            tokens[offset:offset + size] = self.buffer[:size]
            self.buffer = self.buffer[size:]
            offset += size
            if offset == count:
                break

            batch = [self.next_text() for _ in range(_BATCH_SIZE)]
            self.documents_seen += len(batch)
            ids = []
            for encoded in self.tokenizer.encode_batch_fast(batch, add_special_tokens=False):
                ids.extend(encoded.ids)
                ids.append(self.eos_id)
            encoded_tokens = np.asarray(ids, dtype=np.int32)
            self.buffer = encoded_tokens[encoded_tokens != 0]
        return tokens

    def get_state(self):
        positions = dict(self.pending)
        positions.update({str(i): ds.state_dict() for i, ds in self.datasets.items()})
        return dict(rng=self.rng.getstate(), buffer=self.buffer.tolist(), datasets=positions,
                    exhausted=sorted(self.exhausted), documents_seen=self.documents_seen)

    def set_state(self, state):
        self.close()
        self.pending = state["datasets"]
        self.rng.setstate(tuples(state["rng"]))
        self.buffer = np.asarray(state["buffer"], dtype=np.int32)
        self.exhausted = set(state["exhausted"])
        self.update_subset_choices()
        self.documents_seen = state["documents_seen"]

    def close(self):
        for iterator in self.iterators.values():
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        self.datasets.clear()
        self.iterators.clear()


class StreamingIterator:
    def __init__(self, config, training, tokenizer=None):
        self.length = config.max_target_length
        self.batch = int(config.global_batch_size_to_load if training else config.global_batch_size_to_load_eval)
        self.shape = (self.batch, self.length)
        self.positions = np.broadcast_to(np.arange(self.length, dtype=np.int32), self.shape)
        self.input_segments = np.broadcast_to(np.int32(1), self.shape)
        self.weights = [.7, .1, .2]
        if training and _TRAIN_WEIGHTS is not None:
            self.weights = _TRAIN_WEIGHTS
        elif training:
            stage = next(s for s in read_json(ROOT / "configs/curriculum.json")["stages"] if s["length"] == self.length)
            self.weights = [1-stage["math"]-stage["code"], stage["math"], stage["code"]]
        seed = config.data_shuffle_seed + (0 if training else 10000)
        self.rng = random.Random(seed)
        if tokenizer is None:
            tokenizer = load_tokenizer(config.tokenizer_path, config.vocab_size)
        self.streams = {d: DomainStream(d, "train" if training else "eval", tokenizer, seed+i)
                        for i, d in enumerate(DOMAINS)}

    def __iter__(self):
        return self

    def __next__(self):
        rows = {domain: [] for domain in DOMAINS}
        for row, domain in enumerate(self.rng.choices(DOMAINS, weights=self.weights, k=self.batch)):
            rows[domain].append(row)
        inputs = np.empty(self.shape, dtype=np.int32)
        for domain, indices in rows.items():
            if indices:
                inputs[indices] = self.streams[domain].take(len(indices) * self.length).reshape(-1, self.length)
        targets = np.empty_like(inputs)
        targets[:, :-1] = inputs[:, 1:]
        targets[:, -1] = 0
        target_segments = np.empty_like(inputs)
        np.not_equal(targets, 0, out=target_segments)
        return dict(inputs=inputs, targets=targets, inputs_position=self.positions, targets_position=self.positions,
                    inputs_segmentation=self.input_segments, targets_segmentation=target_segments)

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
    def __init__(self, config, training, tokenizer=None):
        self.config, self.training = config, training
        self.tokenizer = tokenizer
        self.iterator = None
        self.eval_cache = []

    def __iter__(self):
        if not self.training and self.config.eval_steps > 0:
            return self.cached_eval()
        self.close()
        iterator = StreamingIterator(self.config, self.training, self.tokenizer)
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
                    self.iterator = StreamingIterator(self.config, False, self.tokenizer)
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
        tokenizer = load_tokenizer(config.tokenizer_path, config.vocab_size)
        train = MultiHostDataLoadIterator(StreamingDataset(config, True, tokenizer), mesh)
        evaluation = None
        if config.eval_interval > 0:
            evaluation = MultiHostDataLoadIterator(StreamingDataset(config, False, tokenizer), mesh)
        return train, evaluation
    create._llm5_streaming = True
    interface.create_data_iterator = create
