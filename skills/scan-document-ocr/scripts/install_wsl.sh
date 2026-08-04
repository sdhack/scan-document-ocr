#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV="${OCR_VENV_WSL:-$HOME/.venvs/scan-document-ocr}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: WSL/Linux 中未检测到 NVIDIA GPU。" >&2
  exit 1
fi

python3.10 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip wheel setuptools
"$VENV/bin/python" -m pip uninstall -y onnxruntime >/dev/null 2>&1 || true
"$VENV/bin/python" -m pip install -r "$SKILL_DIR/requirements.txt"

SITE_PACKAGES="$($VENV/bin/python -c 'import site; print(site.getsitepackages()[0])')"
NVIDIA_DIR="$SITE_PACKAGES/nvidia"
LD_PATH="$NVIDIA_DIR/cublas/lib:$NVIDIA_DIR/cudnn/lib:$NVIDIA_DIR/cuda_runtime/lib:$NVIDIA_DIR/cuda_nvrtc/lib:$NVIDIA_DIR/cufft/lib:$NVIDIA_DIR/curand/lib:$NVIDIA_DIR/cusolver/lib:$NVIDIA_DIR/cusparse/lib:$NVIDIA_DIR/nvtx/lib"

env LD_LIBRARY_PATH="$LD_PATH" "$VENV/bin/python" "$SCRIPT_DIR/verify_cuda.py"
echo "Installed: $VENV"

