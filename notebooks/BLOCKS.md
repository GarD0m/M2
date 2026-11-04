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

## 2. Tune with automatic plan preparation

```python
# Assumes requirements.txt is already installed in this Colab runtime.
# Synthetic data only: no training corpus download is required.
import subprocess, sys
subprocess.run([
    sys.executable, "-m", "llm5.tune", "--prepare-plan",
    "--manifest", "runs/plan-512k/manifest.json",
    "--out", "runs/microbatches-512k.json",
    "--log-dir", "runs/probes-512k",
    "--steps", "12", "--skip-steps", "4", "--max-micro-tokens", "131072"
], check=True)
```

## 3. Prepare actual training data

```python
# Long-running: ~10B tokens, ~40GB uncompressed int32 plus cache.
from pathlib import Path
import json, subprocess, sys
manifest = json.loads(Path("runs/plan-512k/manifest.json").read_text())
if not all((Path("data/packed-512k") / s["name"] / "metadata.json").exists()
           for s in manifest["stages"]):
    subprocess.run([sys.executable, "-m", "llm5.data", "--out", "data/packed-512k",
                    "--tokenizer", "models/tokenizer"], check=True)
```

## 4. Validate data and preview commands

```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "llm5.launch", "--manifest",
                "runs/plan-512k/manifest.json", "--check-data"], check=True)
```

## 5. Train or resume

```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "llm5.launch", "--manifest",
                "runs/plan-512k/manifest.json", "--execute"], check=True)
```
