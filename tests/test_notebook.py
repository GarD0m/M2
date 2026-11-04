import hashlib
import json
from llm5.plan import ROOT


def test_notebook_blocks_compile_and_are_unexecuted():
    notebook = json.loads((ROOT / "notebooks/M2_TPU_v6e_512K.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len({c["id"] for c in notebook["cells"]}) == len(notebook["cells"])
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 5
    for c in cells:
        assert c["outputs"] == [] and c["execution_count"] is None
        source = "".join(c["source"])
        python = "\n".join("pass" if line.startswith("%pip ") else line for line in source.splitlines())
        compile(python, c["id"], "exec")
    code = "\n".join("".join(c["source"]) for c in cells)
    assert "%pip install" not in code and ".venv" not in code
    assert "--prepare-plan" in code


def test_bundled_tokenizer_matches_metadata():
    from tokenizers import Tokenizer
    directory = ROOT / "models/tokenizer"
    meta = json.loads((directory / "training.json").read_text())
    assert hashlib.sha256((directory / "tokenizer.json").read_bytes()).hexdigest() == meta["tokenizer_sha256"]
    t = Tokenizer.from_file(str(directory / "tokenizer.json"))
    assert t.get_vocab_size() == 65536
    assert t.token_to_id("<|pad|>") == 0
