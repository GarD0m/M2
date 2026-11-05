"""Train a 64K byte-level BPE from a bounded text sample."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import shutil

from tokenizers import Tokenizer, Regex, models, trainers, pre_tokenizers, decoders
from tqdm.auto import tqdm

from llm5.plan import ROOT, read_json
from llm5.streaming import configure, is_eval, load_tokenizer, open_subset

SPECIAL = ["<|pad|>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def sample_stream(limit):
    sources = read_json(ROOT / "configs/sources.json")
    budgets = {"math": limit // 10, "code": limit // 5}
    budgets["general"] = limit - sum(budgets.values())
    for domain in ("general", "math", "code"):
        source = sources[domain]
        per_subset, extra = divmod(budgets[domain], len(source["subsets"]))
        for index, subset in enumerate(source["subsets"]):
            remaining = per_subset + (index < extra)
            if not remaining:
                continue
            with closing(iter(open_subset(source, subset))) as rows:
                for row in rows:
                    text = row[source["column"]]
                    if text.strip() and not is_eval(text):
                        text = text[:remaining]
                        yield text
                        remaining -= len(text)
                        if remaining == 0:
                            break


def sample_file(path, limit):
    with path.open(encoding="utf-8") as source:
        for line in source:
            text = json.loads(line)["content"][:limit]
            yield text
            limit -= len(text)
            if limit == 0:
                break


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/tokenizer"))
    parser.add_argument("--max-characters", type=int, default=10_000_000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--sample-file", type=Path, help="Local JSONL with a content field")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"Output already exists: {args.out}. Choose a new --out.")
    if min(args.max_characters, args.min_frequency) < 1:
        parser.error("Sampling and frequency limits must be positive")
    if args.sample_file is None:
        configure()
        texts = sample_stream(args.max_characters)
    else:
        texts = sample_file(args.sample_file, args.max_characters)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=65536, min_frequency=args.min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIAL, show_progress=False,
    )
    with closing(texts), tqdm(total=args.max_characters, unit="char", unit_scale=True, desc="BPE sample") as bar:
        def tracked_texts():
            for text in texts:
                bar.update(len(text))
                yield text
        tokenizer.train_from_iterator(tracked_texts(), trainer=trainer)
    size = tokenizer.get_vocab_size()
    if size != 65536:
        raise ValueError(f"Learned {size:,} tokens. Increase --max-characters or lower --min-frequency to reach 65,536.")
    args.out.mkdir(parents=True)
    tokenizer.save(str(args.out / "tokenizer.json"))
    for name in ("tokenizer_config.json", "chat_template.jinja"):
        shutil.copyfile(ROOT / "models/tokenizer" / name, args.out / name)
    load_tokenizer(args.out, 65536)
    print(f"Saved 65,536-token BPE to {args.out.resolve()}")


if __name__ == "__main__":
    main()
