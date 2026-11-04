#!/usr/bin/env bash
# Run inside an existing Linux TPU v6e-1 VM. Does not provision cloud resources.
set -euo pipefail
cd "$(dirname "$0")/.."
commit=ce608b918c4fdfbfdb62f5a7976d2e15e4fdd720
if [[ ! -d upstream/maxtext/.git ]]; then
  git clone https://github.com/AI-Hypercomputer/maxtext.git upstream/maxtext
fi
if [[ "$(git -C upstream/maxtext rev-parse HEAD)" != "$commit" ]]; then
  git -C upstream/maxtext fetch origin "$commit"
  git -C upstream/maxtext checkout --detach "$commit"
fi
python -m pip install -e 'upstream/maxtext[tpu]' -e .
python -m pip install --no-deps -r upstream/maxtext/src/dependencies/extra_deps/pre_train_github_deps.txt
python -m pip check
python -c 'from llm5.launch import check_runtime; check_runtime()'
mkdir -p runs
python -m pip freeze > runs/resolved-requirements.txt
echo 'Ready. Packages installed into the current Python environment.'
