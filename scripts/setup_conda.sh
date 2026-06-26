#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

find_conda() {
  if [[ -n "${GRADYN_CONDA:-}" && -x "${GRADYN_CONDA}" ]]; then
    printf '%s\n' "$GRADYN_CONDA"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return
  fi
  for candidate in \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "/opt/miniconda3/bin/conda" \
    "/opt/anaconda3/bin/conda"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  return 1
}

CONDA_BIN="$(find_conda || true)"
if [[ -z "$CONDA_BIN" ]]; then
  echo "Conda was not found. Install Miniconda, or set GRADYN_CONDA=/path/to/conda." >&2
  exit 1
fi
export GRADYN_CONDA="$CONDA_BIN"

env_exists() {
  "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$1"
}

for spec in core vocab objects inference; do
  env_name="gradyn-${spec}"
  if env_exists "$env_name"; then
    echo "Updating Conda environment $env_name…"
    "$CONDA_BIN" env update -n "$env_name" -f "$ROOT/environment/${spec}.yml"
  else
    echo "Creating Conda environment $env_name…"
    "$CONDA_BIN" env create -f "$ROOT/environment/${spec}.yml"
  fi
done

"$CONDA_BIN" run -n gradyn-core python -m pip install -e "$ROOT"
"$CONDA_BIN" run -n gradyn-core python "$ROOT/scripts/patch_wilor.py"
"$CONDA_BIN" run -n gradyn-core python "$ROOT/scripts/patch_cutie.py"
"$CONDA_BIN" run -n gradyn-vocab python -m pip install \
  "numpy>=2,<2.3" "opencv-python==4.12.0.88" \
  "torch==2.5.1" "torchvision==0.20.1" \
  "transformers==4.57.3" "mlx-lm==0.29.1" "mlx-vlm==0.3.4"
"$CONDA_BIN" run -n gradyn-inference python -m pip install \
  "numpy==1.26.4" "torch==2.5.1" "torchvision==0.20.1" \
  pytorch-lightning==2.2.4 lightning-utilities \
  torchmetrics==1.4.0 smplx==0.1.28 yacs timm einops hydra-core \
  hydra-submitit-launcher hydra-colorlog pyrootutils rich \
  ultralytics==8.1.34 "opencv-python==4.10.0.84" natsort joblib dill
"$CONDA_BIN" run -n gradyn-inference python -m pip install \
  -e "$ROOT/models/Cutie" --no-deps
"$CONDA_BIN" run -n gradyn-inference python -m pip install \
  --no-build-isolation chumpy
"$CONDA_BIN" run -n gradyn-inference python -m pip uninstall -y \
  opencv-python-headless

echo "Conda environments ready."
