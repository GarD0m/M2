#!/usr/bin/env bash
# Install into the current Python environment; no MaxText clone or venv.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -r requirements.txt
python -c 'from llm5.launch import check_runtime; check_runtime()'
mkdir -p runs
python -m pip freeze > runs/resolved-requirements.txt
echo 'Ready. Packages installed into the current Python environment.'
