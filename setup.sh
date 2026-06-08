#!/usr/bin/env bash
# Hunyuan3D-2 EasySetup - one command to set up everything and launch the UI (Linux/macOS).
#
# Usage (from the repo folder):
#     bash setup.sh
#
# Options via environment variables:
#     PORT=7860 bash setup.sh            # serve on a different port
#     CUDA=cu121 bash setup.sh           # pick a CUDA wheel (cu121|cu124|cu128|cpu)
#     NO_LAUNCH=1 bash setup.sh          # install only, don't start the UI
#     NO_T23D=1 bash setup.sh            # image->3D only (skip text->3D)
#
# Prerequisites:
#   * Miniconda / Anaconda     https://docs.conda.io/en/latest/miniconda.html
#   * NVIDIA GPU + driver (8GB+ VRAM recommended) and a working C++/CUDA toolchain
#     (build-essential + matching CUDA) to compile the texturing extensions.
set -euo pipefail

ENV_NAME="${ENV_NAME:-hunyuan3d}"
PORT="${PORT:-8080}"
CUDA="${CUDA:-cu124}"
PYVER="${PYVER:-3.11}"
MODEL="${MODEL:-tencent/Hunyuan3D-2}"
SUBFOLDER="${SUBFOLDER:-hunyuan3d-dit-v2-0}"
NO_LAUNCH="${NO_LAUNCH:-0}"
NO_T23D="${NO_T23D:-0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

info(){ printf '\033[36m==> %s\033[0m\n' "$*"; }
warn(){ printf '\033[33m[warn] %s\033[0m\n' "$*"; }
die (){ printf '\033[31m[error] %s\033[0m\n' "$*"; exit 1; }

# --- 1. conda ---
command -v conda >/dev/null 2>&1 || die "conda not found. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- 2. create / reuse env ---
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  info "Reusing existing conda env '$ENV_NAME'"
else
  info "Creating conda env '$ENV_NAME' (python $PYVER)"
  conda create -y -n "$ENV_NAME" "python=$PYVER"
fi
conda activate "$ENV_NAME"
info "Python: $(which python)"

# --- 3. PyTorch ---
info "Installing PyTorch ($CUDA)"
if [ "$CUDA" = "cpu" ]; then
  pip install --upgrade torch torchvision
else
  pip install --upgrade torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA"
fi

# --- 4. Python dependencies ---
info "Installing Python requirements"
pip install -r "$REPO/requirements.txt"
pip install tiktoken sentencepiece

# --- 5. build texturing CUDA extensions ---
if python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  CAP="$(python -c 'import torch;m,n=torch.cuda.get_device_capability();print(f"{m}.{n}")')"
  info "Building texturing extensions for CUDA arch $CAP"
  export TORCH_CUDA_ARCH_LIST="$CAP"
  if pip install --no-build-isolation "$REPO/hy3dgen/texgen/custom_rasterizer" && \
     pip install --no-build-isolation "$REPO/hy3dgen/texgen/differentiable_renderer"; then
    :
  else
    warn "Texturing extension build failed. Shape generation will still work; texturing disabled."
  fi
else
  warn "CUDA not available. Skipping texturing extensions (CPU shape gen only, and slow)."
fi

python -c "import importlib.util as u; print('==> Texturing available:', 'yes' if u.find_spec('custom_rasterizer') else 'no')"
info "Setup complete."

# --- 6. launch ---
LAUNCH=(python "$REPO/gradio_app.py" --model_path "$MODEL" --subfolder "$SUBFOLDER" \
        --texgen_model_path tencent/Hunyuan3D-2 --port "$PORT")
[ "$NO_T23D" = "1" ] || LAUNCH+=(--enable_t23d)

if [ "$NO_LAUNCH" = "1" ]; then
  echo
  printf '\033[32mDone. Launch the UI anytime with:\033[0m\n'
  echo "  conda activate $ENV_NAME && ${LAUNCH[*]}"
else
  info "Launching UI on http://localhost:$PORT   (first run downloads ~10-15 GB of model weights)"
  "${LAUNCH[@]}"
fi
