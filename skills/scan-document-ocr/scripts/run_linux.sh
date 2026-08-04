#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/run_linux.sh INPUT.pdf OUTPUT_DIR" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="${OCR_VENV_WSL:-$HOME/.venvs/scan-document-ocr}"
SITE_PACKAGES="$($VENV/bin/python -c 'import site; print(site.getsitepackages()[0])')"
NVIDIA_DIR="${OCR_CUDA_LIB_WSL:-$SITE_PACKAGES/nvidia}"
export LD_LIBRARY_PATH="$NVIDIA_DIR/cublas/lib:$NVIDIA_DIR/cudnn/lib:$NVIDIA_DIR/cuda_runtime/lib:$NVIDIA_DIR/cuda_nvrtc/lib:$NVIDIA_DIR/cufft/lib:$NVIDIA_DIR/curand/lib:$NVIDIA_DIR/cusolver/lib:$NVIDIA_DIR/cusparse/lib:$NVIDIA_DIR/nvtx/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENCV_FOR_THREADS_NUM=2
export OCR_ENGINE_SCRIPT="$SCRIPT_DIR/scan_pdf.py"

"$VENV/bin/python" "$SCRIPT_DIR/verify_cuda.py"
exec "$VENV/bin/python" "$SCRIPT_DIR/persistent_ocr.py" "$1" "$2"

