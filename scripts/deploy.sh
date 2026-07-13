#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# deploy.sh — deploy the ArSSR environment on the H100 training machine
# (or any CUDA-capable Linux box). Self-contained, no sudo, no conda.
#
#   bash scripts/deploy.sh
#
# Override via env vars:
#   VENV=~/.venvs/arssr  PY_VERSION=3.10  CUDA_IDX=cu132  bash scripts/deploy.sh
#
# CUDA_IDX: per project rule CUDA 13.2 -> cu132. If the H100 driver is older
# (pre-CUDA-13), set CUDA_IDX=cu128. H100 (sm_90) is supported from CUDA 11.8+.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

VENV="${VENV:-$HOME/.venvs/arssr}"
PY_VERSION="${PY_VERSION:-3.10}"
CUDA_IDX="${CUDA_IDX:-cu132}"

echo "=== ArSSR deploy ==="
echo " repo:  $REPO"
echo " venv:  $VENV  (python $PY_VERSION)"
echo " torch: $CUDA_IDX"

# 1) uv (installs to ~/.local/bin, no sudo)
if ! command -v uv >/dev/null 2>&1; then
    echo "installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# 2) python venv (uv downloads a standalone cpython if needed)
uv venv --python "$PY_VERSION" "$VENV"
# shellcheck disable=SC1091
. "$VENV/bin/activate"
python --version

# 3) torch with the chosen CUDA build
echo "=== torch ($CUDA_IDX) ==="
uv pip install --index-url "https://download.pytorch.org/whl/$CUDA_IDX/" torch

# 4) ArSSR python deps (Readme: SimpleITK, tqdm, numpy, scipy, scikit-image, tensorboard)
echo "=== ArSSR deps ==="
uv pip install SimpleITK tqdm numpy scipy scikit-image tensorboard

# 5) smoke test: torch + cuda + ArSSR modules import
echo "=== smoke test ==="
python - <<'PY'
import torch, SimpleITK, numpy, scipy, skimage, tqdm
from torch.utils.tensorboard import SummaryWriter
import model, data, utils, encoder, decoder
print("imports OK")
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "=== deploy done ==="
echo "activate later with:  . $VENV/bin/activate"
