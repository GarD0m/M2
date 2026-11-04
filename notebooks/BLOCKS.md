# Notebook code cells

## 1. Clone repository

```python
from pathlib import Path
import os, subprocess, sys

REPO = next((p for p in (Path.cwd(), Path.cwd().parent)
             if (p / "configs/curriculum.yml").exists()), Path("/content/M2"))
if not (REPO / ".git").exists():
    subprocess.run(["git", "clone", "https://github.com/GarD0m/M2.git", str(REPO)], check=True)
os.chdir(REPO)
print(REPO)
```

## 2. Install project requirements directly in Colab

```python
# No MaxText clone, fixed commit, virtual environment or TPU model check.
%pip install -r requirements.txt

PYTHON = sys.executable
def run(*args):
    subprocess.run([PYTHON, *map(str, args)], cwd=REPO, check=True)
```

## 3. Paths and run settings

```python
# Change OUTPUT_ROOT to your own gs://bucket/prefix for durable checkpoints.
OUTPUT_ROOT = str(REPO / "runs/checkpoints")
RUN_PREFIX = "m2-bf16-512k-v1"
TOKENIZER = REPO / "models/tokenizer"
DATA_ROOT = REPO / "data/packed-512k"
PLAN = REPO / "runs/plan-512k"
MICROBATCHES = REPO / "runs/microbatches-512k.json"
RUN_TUNING = True
# Count devices for batch arithmetic; do not check their model.
DEVICE_COUNT = int(subprocess.check_output([
    PYTHON, "-c", "from llm5.launch import check_runtime; print(check_runtime())"
], cwd=REPO, text=True).strip().splitlines()[-1])
print("TPU device count:", DEVICE_COUNT)
```

## 4. Verify the bundled tokenizer, offline

```python
run("-c", """
from tokenizers import Tokenizer
from pathlib import Path
p = Path('models/tokenizer/tokenizer.json')
t = Tokenizer.from_file(str(p))
assert t.get_vocab_size() == 65536
assert t.token_to_id('<|pad|>') == 0
for text in ['繁體中文 English', 'def f(x):\n    return x**2\n', '123 + π = 😀']:
    assert t.decode(t.encode(text, add_special_tokens=False).ids) == text
print('BPE verified: V=65536; no dataset download')
""")
```

## 5. Generate the 512K-token plan

```python
PLAN_ARGS = ["--out", PLAN, "--data-root", DATA_ROOT, "--tokenizer", TOKENIZER,
             "--output-root", OUTPUT_ROOT, "--run-prefix", RUN_PREFIX, "--device-count", DEVICE_COUNT]
run("-m", "llm5.plan", *PLAN_ARGS)
import json
manifest = json.loads((PLAN / "manifest.json").read_text())
for s in manifest["stages"]:
    assert s["effective_batch"] * s["length"] == 524288
    print(s["name"], "micro", s["microbatch"], "GA", s["accumulation"],
          "sequences/update", s["effective_batch"], "steps", s["num_steps"])
```

## 6. Prepare the actual training data

```python
# Long-running: ~10B training tokens, ~40GB uncompressed int32 plus cache.
# Uses HF datasets streaming without shuffle; does not download the entire corpus.
# This is LLM training-data preparation, not tokenizer training.
stages = [s["name"] for s in manifest["stages"]]
if all((DATA_ROOT / name / "metadata.json").exists() for name in stages):
    print("Packed stages already exist; block 8 will validate them.")
else:
    run("-m", "llm5.data", "--out", DATA_ROOT, "--tokenizer", TOKENIZER)
```

## 7. Benchmark microbatch and GA on the TPU

```python
# Optional but recommended; many compilations and short training probes.
# Keeps effective batch at 512K tokens; compares measured steady-state step time.
if RUN_TUNING:
    run("-m", "llm5.tune", "--manifest", PLAN / "manifest.json",
        "--out", MICROBATCHES, "--log-dir", REPO / "runs/probes-512k",
        "--steps", 12, "--skip-steps", 4, "--max-micro-tokens", 131072)
    run("-m", "llm5.plan", *PLAN_ARGS, "--microbatches", MICROBATCHES)
    print(MICROBATCHES.with_suffix(".report.json").read_text())
```

## 8. Validate data and preview commands

```python
run("-m", "llm5.launch", "--manifest", PLAN / "manifest.json", "--check-data")
```

## 9. Train or resume the six stages

```python
# Starts the real ~10B-token run on the current TPU.
# Re-running resumes full optimizer state and the current stage's Grain iterator.
run("-m", "llm5.launch", "--manifest", PLAN / "manifest.json", "--execute")
```
