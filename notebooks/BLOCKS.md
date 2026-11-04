# Notebook code cells

## 1. Clone repository

```python
from pathlib import Path
import os, subprocess, sys

REPO = next((p for p in (Path.cwd(), Path.cwd().parent)
             if (p / "configs/curriculum.json").exists()), Path("/content/M2"))
if not (REPO / ".git").exists():
    subprocess.run(["git", "clone", "https://github.com/GarD0m/M2.git", str(REPO)], check=True)
os.chdir(REPO)
print(REPO)
```

## 2. Tune directly in this kernel

```python
# Run in the notebook kernel; no training subprocesses or error wrappers.
%run -m llm5.tune --prepare-plan --manifest runs/plan-512k/manifest.json --out runs/microbatches-512k.json --log-dir runs/probes-512k
```

## 3. Prepare actual training data

```python
# Long-running: ~10B tokens, ~40GB uncompressed int32 plus cache.
from pathlib import Path
import json
manifest = json.loads(Path("runs/plan-512k/manifest.json").read_text())
if not all((Path("data/packed-512k") / s["name"] / "metadata.json").exists()
           for s in manifest["stages"]):
    %run -m llm5.data --out data/packed-512k --tokenizer models/tokenizer
```

## 4. Validate data and preview commands

```python
%run -m llm5.launch --manifest runs/plan-512k/manifest.json --check-data
```

## 5. Train or resume in this kernel

```python
%run -m llm5.launch --manifest runs/plan-512k/manifest.json --execute
```
