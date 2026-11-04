"""Train a byte-level BPE with Hugging Face's Rust tokenizers library."""
import argparse
import json
from pathlib import Path
import random
from math import ceil

from tokenizers import Tokenizer, Regex, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast
from llm5.data import documents, DOMAINS
from llm5.plan import ROOT, digest, read_json

SPECIAL = ["<|pad|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def build_bpe(texts, out, vocab_size=65536, min_frequency=2):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  special_tokens=SPECIAL, show_progress=True)
    tok.train_from_iterator(texts, trainer=trainer)
    if tok.get_vocab_size() != vocab_size:
        raise ValueError(f"Only learned {tok.get_vocab_size()} tokens; increase the training corpus to reach {vocab_size}")
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token=SPECIAL[0], eos_token=SPECIAL[1],
                                  bos_token=SPECIAL[1], additional_special_tokens=SPECIAL[2:],
                                  model_max_length=32768, clean_up_tokenization_spaces=False)
    fast.chat_template = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(out)
    return fast


def sample_texts(max_documents, max_characters, local_root=None):
    sources = read_json(ROOT / "configs/sources.json")
    weights = dict(zip(DOMAINS, [0.7, 0.1, 0.2]))
    chars = count = 0
    for domain in DOMAINS:
        indices = [None] if local_root else range(len(sources[domain]["subsets"]))
        doc_quota = ceil(max_documents * weights[domain] / len(indices))
        char_quota = ceil(max_characters * weights[domain] / len(indices))
        for index in indices:
            stream = documents(domain, "train", local_root=local_root, subset_index=index)
            subset_chars = subset_docs = 0
            try:
                while subset_docs < doc_quota and subset_chars < char_quota:
                    try:
                        text = next(stream)
                    except StopIteration:
                        break
                    text = text[:min(char_quota-subset_chars, max_characters-chars)]
                    yield text
                    chars += len(text)
                    count += 1
                    subset_chars += len(text)
                    subset_docs += 1
                    if chars >= max_characters or count >= max_documents:
                        return
            finally:
                stream.close()


def sample_file_texts(path):
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)["content"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="artifacts/tokenizer")
    p.add_argument("--max-documents", type=int, default=100000)
    p.add_argument("--max-characters", type=int, default=100000000)
    p.add_argument("--local-root")
    p.add_argument("--sample-file", help="Reuse a saved JSONL sample; no network access")
    args = p.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Tokenizer output {out} must be empty")
    if min(args.max_documents, args.max_characters) <= 0:
        p.error("Sampling limits must be positive")
    sample = Path(args.sample_file) if args.sample_file else out.parent / (out.name + "-hf-sample.jsonl")
    if not args.sample_file:
        if sample.exists():
            p.error(f"Sample exists: {sample}. Reuse with --sample-file or select another --out.")
        sample.parent.mkdir(parents=True, exist_ok=True)
        temporary = sample.with_suffix(".jsonl.partial")
        count = chars = 0
        print("Sampling datasets sequentially, no shuffle; one shard reader at a time", flush=True)
        with temporary.open("w", encoding="utf-8") as target:
            for text in sample_texts(args.max_documents, args.max_characters, args.local_root):
                target.write(json.dumps({"content": text}, ensure_ascii=False) + "\n")
                count += 1
                chars += len(text)
                if count % 100 == 0:
                    print(f"Sample: {count:,} docs / {chars:,} chars", flush=True)
        temporary.replace(sample)
        print(f"Saved sample: {count:,} docs / {chars:,} chars. Starting offline BPE.", flush=True)
        import gc
        import pyarrow as pa
        gc.collect()
        pa.default_memory_pool().release_unused()
    tok = build_bpe(sample_file_texts(sample), out)
    for text in ["繁體中文與 English\n", "def f(x):\n    return x**2\n", "1234567890 + π = 😀"]:
        assert tok.decode(tok.encode(text, add_special_tokens=False)) == text
    metadata = dict(vocab_size=len(tok), sources_sha256=digest(ROOT / "configs/sources.json"),
                    tokenizer_sha256=digest(out / "tokenizer.json"),
                    max_documents=args.max_documents, max_characters=args.max_characters,
                    seed=42, sample_domain_weights=[0.7, 0.1, 0.2], algorithm="ByteLevel BPE",
                    sampling_mode="sequential-no-shuffle", sample_file=str(sample))
    (out / "training.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(tok)}-token BPE to {out}")


if __name__ == "__main__":
    main()
