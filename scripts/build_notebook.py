"""Generate the Colab notebook and matching code-cell reference."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = [('1. Clone repository',
  'from pathlib import Path\n'
  'import os, subprocess, sys\n'
  '\n'
  'REPO = next((p for p in (Path.cwd(), Path.cwd().parent)\n'
  '             if (p / "configs/curriculum.yml").exists()), Path("/content/M2"))\n'
  'if not (REPO / ".git").exists():\n'
  '    subprocess.run(["git", "clone", "https://github.com/GarD0m/M2.git", str(REPO)], check=True)\n'
  'os.chdir(REPO)\n'
  'print(REPO)\n'),
 ('2. Install directly into Colab with pip',
  '# Select a TPU runtime. No virtual environment or TPU model restriction.\n'
  'MAXTEXT = REPO / "upstream/maxtext"\n'
  'PIN = "ce608b918c4fdfbfdb62f5a7976d2e15e4fdd720"\n'
  'if not (MAXTEXT / ".git").exists():\n'
  '    subprocess.run(["git", "clone", "https://github.com/AI-Hypercomputer/maxtext.git", str(MAXTEXT)], check=True)\n'
  'subprocess.run(["git", "-C", str(MAXTEXT), "fetch", "origin", PIN], check=True)\n'
  'subprocess.run(["git", "-C", str(MAXTEXT), "checkout", "--detach", PIN], check=True)\n'
  '\n'
  '%pip install -e "./upstream/maxtext[tpu]" -e .\n'
  '%pip install --no-deps -r ./upstream/maxtext/src/dependencies/extra_deps/pre_train_github_deps.txt\n'
  '\n'
  'PYTHON = sys.executable\n'
  'def run(*args):\n'
  '    subprocess.run([PYTHON, *map(str, args)], cwd=REPO, check=True)\n'),
 ('3. Paths and run settings',
  '# Change OUTPUT_ROOT to your own gs://bucket/prefix for durable checkpoints.\n'
  'OUTPUT_ROOT = str(REPO / "runs/checkpoints")\n'
  'RUN_PREFIX = "m2-bf16-512k-v1"\n'
  'TOKENIZER = REPO / "models/tokenizer"\n'
  'DATA_ROOT = REPO / "data/packed-512k"\n'
  'PLAN = REPO / "runs/plan-512k"\n'
  'MICROBATCHES = REPO / "runs/microbatches-512k.json"\n'
  'RUN_TUNING = True\n'
  '# Count devices for batch arithmetic; do not check their model.\n'
  'DEVICE_COUNT = int(subprocess.check_output([\n'
  '    PYTHON, "-c", "from llm5.launch import check_runtime; print(check_runtime())"\n'
  '], cwd=REPO, text=True).strip().splitlines()[-1])\n'
  'print("TPU device count:", DEVICE_COUNT)\n'),
 ('4. Verify the bundled tokenizer, offline',
  'run("-c", """\n'
  'from tokenizers import Tokenizer\n'
  'from pathlib import Path\n'
  "p = Path('models/tokenizer/tokenizer.json')\n"
  't = Tokenizer.from_file(str(p))\n'
  'assert t.get_vocab_size() == 65536\n'
  "assert t.token_to_id('<|pad|>') == 0\n"
  "for text in ['繁體中文 English', 'def f(x):\\n    return x**2\\n', '123 + π = 😀']:\n"
  '    assert t.decode(t.encode(text, add_special_tokens=False).ids) == text\n'
  "print('BPE verified: V=65536; no dataset download')\n"
  '""")\n'),
 ('5. Generate the 512K-token plan',
  'PLAN_ARGS = ["--out", PLAN, "--data-root", DATA_ROOT, "--tokenizer", TOKENIZER,\n'
  '             "--output-root", OUTPUT_ROOT, "--run-prefix", RUN_PREFIX, "--device-count", DEVICE_COUNT]\n'
  'run("-m", "llm5.plan", *PLAN_ARGS)\n'
  'import json\n'
  'manifest = json.loads((PLAN / "manifest.json").read_text())\n'
  'for s in manifest["stages"]:\n'
  '    assert s["effective_batch"] * s["length"] == 524288\n'
  '    print(s["name"], "micro", s["microbatch"], "GA", s["accumulation"],\n'
  '          "sequences/update", s["effective_batch"], "steps", s["num_steps"])\n'),
 ('6. Prepare the actual training data',
  '# Long-running: ~10B training tokens, ~40GB uncompressed int32 plus cache.\n'
  '# Uses HF datasets streaming without shuffle; does not download the entire corpus.\n'
  '# This is LLM training-data preparation, not tokenizer training.\n'
  'stages = [s["name"] for s in manifest["stages"]]\n'
  'if all((DATA_ROOT / name / "metadata.json").exists() for name in stages):\n'
  '    print("Packed stages already exist; block 8 will validate them.")\n'
  'else:\n'
  '    run("-m", "llm5.data", "--out", DATA_ROOT, "--tokenizer", TOKENIZER)\n'),
 ('7. Benchmark microbatch and GA on the TPU',
  '# Optional but recommended; many compilations and short training probes.\n'
  '# Keeps effective batch at 512K tokens; compares measured steady-state step time.\n'
  'if RUN_TUNING:\n'
  '    run("-m", "llm5.tune", "--manifest", PLAN / "manifest.json",\n'
  '        "--out", MICROBATCHES, "--log-dir", REPO / "runs/probes-512k",\n'
  '        "--steps", 12, "--skip-steps", 4, "--max-micro-tokens", 131072)\n'
  '    run("-m", "llm5.plan", *PLAN_ARGS, "--microbatches", MICROBATCHES)\n'
  '    print(MICROBATCHES.with_suffix(".report.json").read_text())\n'),
 ('8. Validate data and preview commands',
  'run("-m", "llm5.launch", "--manifest", PLAN / "manifest.json", "--check-data")\n'),
 ('9. Train or resume the six stages',
  '# Starts the real ~10B-token run on the current TPU.\n'
  "# Re-running resumes full optimizer state and the current stage's Grain iterator.\n"
  'run("-m", "llm5.launch", "--manifest", PLAN / "manifest.json", "--execute")\n')]


def main():
    cells = [dict(cell_type="markdown", metadata={}, source=[
        "# M2 — five-layer Qwen3 / TPU v6e-1\n",
        "Run in a Colab TPU runtime with Python 3.12+. Install directly using %pip; no virtual environment or TPU model restriction.\n",
        "BF16 stored weights (no FP32 master copy), Muon, Tokamax, SEQ_MINOR QKV, fused QKV/MLP, WSO, 512K tokens/update.\n",
        "The bundled 64K BPE is already trained; only the actual pretraining data needs preparation.\n",
    ])]
    reference = ["# Notebook code cells\n"]
    for i, (title, code) in enumerate(BLOCKS):
        cells += [dict(cell_type="markdown", metadata={}, source=["## " + title]),
                  dict(cell_type="code", metadata={}, execution_count=None, outputs=[],
                       id=f"block-{i+1}", source=code.splitlines(keepends=True))]
        reference.append(f"## {title}\n\n```python\n{code}```\n")
    for index, cell in enumerate(cells):
        cell.setdefault("id", f"cell-{index:02d}")
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                               language_info=dict(name="python", version="3.12")), nbformat=4, nbformat_minor=5)
    directory = ROOT / "notebooks"
    directory.mkdir(exist_ok=True)
    (directory / "M2_TPU_v6e_512K.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    (directory / "BLOCKS.md").write_text("\n".join(reference), encoding="utf-8")


if __name__ == "__main__":
    main()
