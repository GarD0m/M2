#!/usr/bin/env bash
# Run inside an existing Linux TPU v6e-1 VM. Does not provision cloud resources.
set -euo pipefail
cd "$(dirname "$0")/.."
commit=ce608b918c4fdfbfdb62f5a7976d2e15e4fdd720
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/'; exit 1; }
if [[ ! -d upstream/maxtext/.git ]]; then
  git clone https://github.com/AI-Hypercomputer/maxtext.git upstream/maxtext
fi
if [[ "$(git -C upstream/maxtext rev-parse HEAD)" != "$commit" ]]; then
  git -C upstream/maxtext fetch origin "$commit"
  git -C upstream/maxtext checkout --detach "$commit"
fi
uv venv --python 3.12 .venv-tpu
source .venv-tpu/bin/activate
uv pip install -e 'upstream/maxtext[tpu]' --resolution=lowest
install_tpu_pre_train_extra_deps
uv pip install -e . --resolution=lowest
uv pip check
python -c 'from llm5.launch import check_runtime; check_runtime()'
uv pip freeze > .venv-tpu/resolved-requirements.txt
echo 'Ready. Activate with: source .venv-tpu/bin/activate'
