#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE_ENV_FILE="$REPO_ROOT/environment/conda-base.yml"

ENV_NAME="semift"
PYTHON_VERSION="3.10"
CUDA_VARIANT="auto"
WITH_XFORMERS=0
WITH_MMSEG=0

usage() {
  cat <<EOF
Usage:
  bash scripts/setup_conda_env.sh [options]

Options:
  --env-name NAME       Conda environment name. Default: semift
  --python VERSION      Python version. Default: 3.10
  --cuda VARIANT        auto | 12.1 | 11.8 | cpu. Default: auto
  --with-xformers       Install optional xformers (Linux + CUDA only)
  --with-mmseg          Install optional mmseg ecosystem for UPerNet/mmseg paths
  -h, --help            Show this help message

Examples:
  bash scripts/setup_conda_env.sh
  bash scripts/setup_conda_env.sh --env-name semift-gpu --cuda 12.1 --with-xformers
  bash scripts/setup_conda_env.sh --env-name semift-mmseg --cuda cpu --with-mmseg
EOF
}

log() {
  printf '[setup_conda_env] %s\n' "$*"
}

die() {
  printf '[setup_conda_env] ERROR: %s\n' "$*" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      [[ $# -ge 2 ]] || die "--env-name requires a value"
      ENV_NAME="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires a value"
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --cuda)
      [[ $# -ge 2 ]] || die "--cuda requires a value"
      CUDA_VARIANT="$2"
      shift 2
      ;;
    --with-xformers)
      WITH_XFORMERS=1
      shift
      ;;
    --with-mmseg)
      WITH_MMSEG=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -f "$BASE_ENV_FILE" ]] || die "Missing environment spec: $BASE_ENV_FILE"
command_exists conda || die "conda not found. Please install Miniconda/Anaconda first."

cd "$REPO_ROOT"
OS_NAME="$(uname -s)"

resolve_cuda_variant() {
  case "$CUDA_VARIANT" in
    auto)
      if [[ "$OS_NAME" == "Linux" ]] && command_exists nvidia-smi; then
        printf '12.1\n'
      else
        printf 'cpu\n'
      fi
      ;;
    12.1|11.8|cpu)
      printf '%s\n' "$CUDA_VARIANT"
      ;;
    *)
      die "Unsupported --cuda value '$CUDA_VARIANT'. Use auto, 12.1, 11.8, or cpu."
      ;;
  esac
}

CUDA_RESOLVED="$(resolve_cuda_variant)"
log "Repository root: $REPO_ROOT"
log "Environment name: $ENV_NAME"
log "Python version: $PYTHON_VERSION"
log "CUDA variant: $CUDA_RESOLVED"

if [[ "$WITH_XFORMERS" -eq 1 ]]; then
  if [[ "$OS_NAME" != "Linux" ]]; then
    die "--with-xformers is only supported by this script on Linux."
  fi
  if [[ "$CUDA_RESOLVED" == "cpu" ]]; then
    die "--with-xformers requires a CUDA environment, not --cuda cpu."
  fi
fi

ENV_EXISTS=0
if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  ENV_EXISTS=1
fi

if [[ "$ENV_EXISTS" -eq 0 ]]; then
  log "Creating conda environment '$ENV_NAME' from base spec"
  conda env create -n "$ENV_NAME" -f "$BASE_ENV_FILE"
else
  log "Environment '$ENV_NAME' already exists; updating from base spec"
  conda env update -n "$ENV_NAME" -f "$BASE_ENV_FILE"
fi

log "Pinning Python to $PYTHON_VERSION"
conda install -y -n "$ENV_NAME" -c conda-forge "python=$PYTHON_VERSION"

install_torch() {
  if [[ "$CUDA_RESOLVED" == "cpu" ]]; then
    log "Installing CPU PyTorch"
    conda install -y -n "$ENV_NAME" -c pytorch pytorch torchvision cpuonly
  else
    log "Installing CUDA PyTorch ($CUDA_RESOLVED)"
    conda install -y -n "$ENV_NAME" -c pytorch -c nvidia pytorch torchvision "pytorch-cuda=$CUDA_RESOLVED"
  fi
}

install_torch

if [[ "$WITH_XFORMERS" -eq 1 ]]; then
  log "Installing optional xformers"
  conda run -n "$ENV_NAME" python -m pip install xformers
fi

if [[ "$WITH_MMSEG" -eq 1 ]]; then
  log "Installing optional mmseg ecosystem"
  conda run -n "$ENV_NAME" python -m pip install mmengine mmcv-lite mmsegmentation
fi

log "Running smoke tests"
conda run -n "$ENV_NAME" python - <<'PY'
import importlib
mods = [
    'torch', 'torchvision', 'yaml', 'numpy', 'PIL', 'matplotlib', 'sklearn',
    'cv2', 'h5py', 'einops', 'tensorboard', 'accelerate', 'transformers',
    'huggingface_hub', 'pytest'
]
missing = []
for mod in mods:
    try:
        importlib.import_module(mod)
    except Exception as exc:
        missing.append((mod, repr(exc)))
if missing:
    raise SystemExit('Missing imports: ' + '; '.join(f'{m} -> {e}' for m, e in missing))
import torch
from peft.tuners.semift import SemiFTConfig
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('semift_config_ok', SemiFTConfig().__class__.__name__)
PY

if [[ "$WITH_XFORMERS" -eq 1 ]]; then
  log "Validating xformers import"
  conda run -n "$ENV_NAME" python - <<'PY'
import xformers.ops
print('xformers_ok')
PY
fi

if [[ "$WITH_MMSEG" -eq 1 ]]; then
  log "Validating mmseg import"
  conda run -n "$ENV_NAME" python - <<'PY'
import mmseg
print('mmseg_ok', mmseg.__version__)
PY
fi

cat <<EOF

[setup_conda_env] Done.

Activate environment:
  conda activate $ENV_NAME

Quick check:
  conda run -n $ENV_NAME python -c "import torch; from peft.tuners.semift import SemiFTConfig; print(torch.__version__)"

Recommended tests:
  conda run -n $ENV_NAME python -m pytest test/test_unimatchv2_peft_config.py test/test_batch_train.py
EOF
