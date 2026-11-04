"""Generate the checked-in TPU notebook and a matching code-cell reference."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKS = [
("1. Clone repository", '''from pathlib import Path
import os, subprocess, sys, sysconfig

REPO = next((p for p in (Path.cwd(), Path.cwd().parent)
             if (p / "configs/curriculum.yml").exists()), Path.home() / "M2")
if not (REPO / ".git").exists():
    subprocess.run(["git", "clone", "https://github.com/GarD0m/M2.git", str(REPO)], check=True)
os.chdir(REPO)
print(REPO)
'''),
("2. Install on the existing v6e-1 VM", '''# Linux TPU VM only. This does not create cloud resources.
subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
os.environ["PATH"] = os.pathsep.join([
    sysconfig.get_path("scripts"), str(Path.home() / ".local/bin"), os.environ["PATH"]])
subprocess.run(["bash", "scripts/setup_tpu.sh"], check=True)
PYTHON = str(REPO / ".venv-tpu/bin/python")
def run(*args):
    subprocess.run([PYTHON, *map(str, args)], cwd=REPO, check=True)
run("-c", "from llm5.launch import check_runtime; check_runtime()")
'''),
("3. Paths and run settings", '''# Change OUTPUT_ROOT to your own gs://bucket/prefix for durable checkpoints.
OUTPUT_ROOT = str(REPO / "runs/checkpoints")
RUN_PREFIX = "m2-bf16-512k-v1"
TOKENIZER = REPO / "models/tokenizer"
DATA_ROOT = REPO / "data/packed-512k"
PLAN = REPO / "runs/plan-512k"
MICROBATCHES = REPO / "runs/microbatches-512k.json"
RUN_TUNING = True
'''),
("4. Verify the bundled tokenizer, offline", '''run("-c", """
from tokenizers import Tokenizer
from pathlib import Path
p = Path('models/tokenizer/tokenizer.json')
t = Tokenizer.from_file(str(p))
assert t.get_vocab_size() == 65536
assert t.token_to_id('<|pad|>') == 0
for text in ['繁體中文 English', 'def f(x):\\n    return x**2\\n', '123 + π = 😀']:
    assert t.decode(t.encode(text, add_special_tokens=False).ids) == text
print('BPE verified: V=65536; no dataset download')
""")
'''),
("5. Generate the 512K-token plan", '''PLAN_ARGS = ["--out", PLAN, "--data-root", DATA_ROOT, "--tokenizer", TOKENIZER,
             "--output-root", OUTPUT_ROOT, "--run-prefix", RUN_PREFIX]
run("-m", "llm5.plan", *PLAN_ARGS)
import json
manifest = json.loads((PLAN / "manifest.json").read_text())
for s in manifest["stages"]:
    assert s["effective_batch"] * s["length"] == 524288
    print(s["name"], "micro", s["microbatch"], "GA", s["accumulation"],
          "sequences/update", s["effective_batch"], "steps", s["num_steps"])
'''),
("6. Prepare the actual training data", '''# Long-running: ~10B training tokens, ~40GB uncompressed int32 plus cache.
# Uses HF datasets streaming without shuffle; does not download the entire corpus.
# This is LLM training-data preparation, not tokenizer training.
stages = [s["name"] for s in manifest["stages"]]
if all((DATA_ROOT / name / "metadata.json").exists() for name in stages):
    print("Packed stages already exist; block 8 will validate them.")
else:
    run("-m", "llm5.data", "--out", DATA_ROOT, "--tokenizer", TOKENIZER)
'''),
("7. Benchmark microbatch and GA on the TPU", '''# Optional but recommended; many compilations and short training probes.
# Keeps effective batch at 512K tokens; compares measured steady-state step time.
if RUN_TUNING:
    run("-m", "llm5.tune", "--manifest", PLAN / "manifest.json",
        "--out", MICROBATCHES, "--log-dir", REPO / "runs/probes-512k",
        "--steps", 12, "--skip-steps", 4, "--max-micro-tokens", 131072)
    run("-m", "llm5.plan", *PLAN_ARGS, "--microbatches", MICROBATCHES)
    print(MICROBATCHES.with_suffix(".report.json").read_text())
'''),
("8. Validate data and preview commands", '''run("-m", "llm5.launch", "--manifest", PLAN / "manifest.json", "--check-data")
'''),
("9. Train or resume the six stages", '''# Starts the real ~10B-token run on the current TPU.
# Re-running resumes full optimizer state and the current stage's Grain iterator.
run("-m", "llm5.launch", "--manifest", PLAN / "manifest.json", "--execute")
'''),
]


def main():
    cells = [dict(cell_type="markdown", metadata={}, source=[
        "# M2 — five-layer Qwen3 / TPU v6e-1\n",
        "Run on an existing Linux TPU v6e-1 VM with a Jupyter server. Colab does not guarantee v6e-1 hardware.\n",
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
