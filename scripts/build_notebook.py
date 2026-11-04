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
 ('2. Tune with automatic plan preparation',
  '# Assumes requirements.txt is already installed in this Colab runtime.\n'
  '# Synthetic data only: no training corpus download is required.\n'
  'import subprocess, sys\n'
  'subprocess.run([\n'
  '    sys.executable, "-m", "llm5.tune", "--prepare-plan",\n'
  '    "--manifest", "runs/plan-512k/manifest.json",\n'
  '    "--out", "runs/microbatches-512k.json",\n'
  '    "--log-dir", "runs/probes-512k",\n'
  '    "--steps", "12", "--skip-steps", "4", "--max-micro-tokens", "131072"\n'
  '], check=True)\n'),
 ('3. Prepare actual training data',
  '# Long-running: ~10B tokens, ~40GB uncompressed int32 plus cache.\n'
  'from pathlib import Path\n'
  'import json, subprocess, sys\n'
  'manifest = json.loads(Path("runs/plan-512k/manifest.json").read_text())\n'
  'if not all((Path("data/packed-512k") / s["name"] / "metadata.json").exists()\n'
  '           for s in manifest["stages"]):\n'
  '    subprocess.run([sys.executable, "-m", "llm5.data", "--out", "data/packed-512k",\n'
  '                    "--tokenizer", "models/tokenizer"], check=True)\n'),
 ('4. Validate data and preview commands',
  'import subprocess, sys\n'
  'subprocess.run([sys.executable, "-m", "llm5.launch", "--manifest",\n'
  '                "runs/plan-512k/manifest.json", "--check-data"], check=True)\n'),
 ('5. Train or resume',
  'import subprocess, sys\n'
  'subprocess.run([sys.executable, "-m", "llm5.launch", "--manifest",\n'
  '                "runs/plan-512k/manifest.json", "--execute"], check=True)\n')]


def main():
    cells = [dict(cell_type="markdown", metadata={}, source=[
        "# M2 — five-layer Qwen3 / TPU v6e-1\n",
        "Colab TPU runtime, Python 3.12+. Install requirements separately with %pip install -r requirements.txt. No virtual environment or TPU model restriction. The original setup blocks 2-5 are removed; tune prepares its own plan.\n",
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
