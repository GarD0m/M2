"""Stream pinned UltraData L3 and build equal-length pretokenized Parquet records."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from llm5.plan import ROOT, digest, read_json, curriculum

DOMAINS = ("general", "math", "code")


def is_eval(text):
    # Content hashing keeps duplicate text on the same side of the split.
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % 1000 < 5


def documents(domain, partition, seed=42, local_root=None, subset_index=None):
    if partition not in ("train", "eval"):
        raise ValueError("partition must be train or eval")
    if local_root:
        # JSONL fixtures or an already curated local export, one `content` per line.
        with (Path(local_root) / f"{domain}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["content"]
                if text.strip() and is_eval(text) == (partition == "eval"):
                    yield text
        return
    from datasets import load_dataset
    from pyarrow.dataset import ParquetFragmentScanOptions
    source = read_json(ROOT / "configs/sources.json")[domain]
    active = []
    subsets = source["subsets"] if subset_index is None else [source["subsets"][subset_index]]
    for sub in subsets:
        ds = load_dataset(source["repo"], sub["config"], split=sub["split"],
                          revision=source["revision"], streaming=True,
                          columns=[source["column"]], batch_size=16,
                          fragment_scan_options=ParquetFragmentScanOptions(pre_buffer=False))
        # No shuffle(): datasets 5.x otherwise prefetches ten shards by default.
        active.append((iter(ds), sub["weight"]))
    rng = random.Random(seed)
    while active:
        i = rng.choices(range(len(active)), weights=[s[1] for s in active])[0]
        try:
            row = next(active[i][0])
        except StopIteration:
            active.pop(i)
            continue
        text = row[source["column"]]
        if not isinstance(text, str):
            raise TypeError(f"{source['repo']} content is not a string")
        if text.strip() and is_eval(text) == (partition == "eval"):
            yield text


class TokenStream:
    """Continuous document stream, separated by EOS, with bounded token buffers."""
    def __init__(self, docs, tokenizer):
        self.docs, self.tokenizer, self.buffer = iter(docs), tokenizer, deque()
        self.eos = tokenizer.token_to_id("<|endoftext|>")
        if self.eos is None:
            raise ValueError("Tokenizer has no EOS")

    def take(self, n):
        while len(self.buffer) < n:
            # Batch BPE encoding runs in the tokenizers Rust thread pool.
            batch = []
            for _ in range(32):
                try:
                    batch.append(next(self.docs))
                except StopIteration:
                    break
            if not batch:
                raise RuntimeError("Source exhausted; add data rather than silently repeating documents")
            for encoded in self.tokenizer.encode_batch(batch, add_special_tokens=False):
                self.buffer.extend(t for t in encoded.ids if t != 0)
                self.buffer.append(self.eos)
        return [self.buffer.popleft() for _ in range(n)]


def domain_order(count, weights, seed):
    """Largest-remainder quotas; every example has the same token length."""
    expected = np.asarray(weights, dtype=np.float64) * count
    quotas = np.floor(expected).astype(int)
    remainder = count - int(quotas.sum())
    for i in np.argsort(-(expected - quotas), kind="stable")[:remainder]:
        quotas[i] += 1
    order = np.repeat(np.arange(len(weights), dtype=np.uint8), quotas)
    np.random.default_rng(seed).shuffle(order)
    return order, quotas.tolist()


def write_shards(directory, prefix, count, length, streams, weights, seed):
    order, quotas = domain_order(count, weights, seed)
    rows_per_shard = max(1, (64 * 1024**2) // (4 * length))
    schema = pa.schema([("text", pa.list_(pa.int32()))])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for start in range(0, count, rows_per_shard):
        path = directory / f"{prefix}-{start // rows_per_shard:05d}.parquet"
        temporary = path.with_suffix(".parquet.partial")
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            for batch_start in range(start, min(start + rows_per_shard, count), 64):
                rows = [streams[DOMAINS[int(i)]].take(length)
                        for i in order[batch_start:min(batch_start + 64, start + rows_per_shard, count)]]
                writer.write_table(pa.Table.from_pydict({"text": rows}, schema=schema))
        temporary.replace(path)
        print(f"Wrote {path}", flush=True)
    return dict(zip(DOMAINS, quotas))


def prepare(out, tokenizer, eval_sequences=128, local_root=None, limit_stages=None):
    out = Path(out)
    raw = read_json(ROOT / "configs/curriculum.json")
    stages = curriculum(raw)
    tok = Tokenizer.from_file(str(Path(tokenizer) / "tokenizer.json"))
    if tok.get_vocab_size() != 65536 or tok.token_to_id("<|pad|>") != 0:
        raise ValueError("Production preparation requires V=65536 and PAD=0")
    token_hash = digest(Path(tokenizer) / "tokenizer.json")
    streams = {part: {d: TokenStream(documents(d, part, raw["seed"], local_root), tok)
                      for d in DOMAINS} for part in ("train", "eval")}
    for i, s in enumerate(stages):
        if limit_stages and i >= limit_stages:
            break
        directory = out / s["name"]
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"{directory} is nonempty; use a fresh output directory")
        counts = write_shards(directory, "train", s["sequences"], s["length"], streams["train"],
                              [s[d] for d in DOMAINS], raw["seed"] + i)
        # Fixed eval domain mixture across all context stages.
        write_shards(directory, "eval", eval_sequences, s["length"], streams["eval"],
                     [0.7, 0.1, 0.2], raw["seed"] + 100 + i)
        metadata = dict(stage=s, domain_sequences=counts, tokenizer_sha256=token_hash,
                        curriculum_sha256=digest(ROOT / "configs/curriculum.json"),
                        sources_sha256=digest(ROOT / "configs/sources.json"),
                        eval_sequences=eval_sequences, schema="length-L token list; native shift masks final target")
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/packed")
    p.add_argument("--tokenizer", default="artifacts/tokenizer")
    p.add_argument("--eval-sequences", type=int, default=128)
    p.add_argument("--local-root")
    p.add_argument("--limit-stages", type=int)
    prepare(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
