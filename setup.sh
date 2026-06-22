#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WITH_QWEN=0
MANO_DIR="${GRADYN_MANO_DIR:-}"
MANO_DOWNLOAD_DIR="$ROOT/licensed_models/mano_v1_2/models"
MANO_LEFT_ID="1scfcvug_hc_VvYl-vKHouic8Xd9lgUXE"
MANO_RIGHT_ID="1nWOkgnUFqS9IVUG5_tTXjPaSmflwNLQ6"
MANO_LEFT_SHA256="c4022f7083f2ca7c78b2b3d595abbab52debd32b09d372b16923a801f0ea6a30"
MANO_RIGHT_SHA256="45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767"

usage() {
  cat <<'EOF'
Usage: ./setup.sh [--with-qwen] [--mano-dir /path/to/models]

Creates all Conda environments, installs Gradyn, clones pinned upstream
repositories, downloads required model files including MANO, and verifies the
installation.

Options:
  --with-qwen       Download the optional Qwen3-VL object-discovery model.
  --mano-dir PATH   Use an existing licensed MANO directory instead of the
                    configured Gradyn download.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-qwen)
      WITH_QWEN=1
      shift
      ;;
    --mano-dir)
      MANO_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$MANO_DIR" && (
  ! -f "$MANO_DIR/MANO_RIGHT.pkl" || ! -f "$MANO_DIR/MANO_LEFT.pkl"
) ]]; then
  echo "--mano-dir must contain MANO_RIGHT.pkl and MANO_LEFT.pkl: $MANO_DIR" >&2
  exit 2
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "Gradyn's local setup currently supports Apple Silicon macOS only." >&2
  exit 1
fi

for command in git curl; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Required command not found: $command" >&2
    exit 1
  fi
done

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
  echo "Conda was not found. Install Miniconda, then rerun ./setup.sh." >&2
  exit 1
fi
export GRADYN_CONDA="$CONDA_BIN"

echo "Using Conda: $CONDA_BIN"

# Bootstrap the lightweight CLI first so it can manage pinned model repositories.
if "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "gradyn-core"; then
  "$CONDA_BIN" env update -n gradyn-core -f "$ROOT/environment/core.yml"
else
  "$CONDA_BIN" env create -f "$ROOT/environment/core.yml"
fi
"$CONDA_BIN" run -n gradyn-core python -m pip install -e "$ROOT"
"$CONDA_BIN" run -n gradyn-core gradyn models setup

"$ROOT/scripts/setup_conda.sh"

download_args=()
if [[ "$WITH_QWEN" -eq 1 ]]; then
  download_args+=(--with-qwen)
fi
"$CONDA_BIN" run -n gradyn-core gradyn models download "${download_args[@]}"

if [[ -z "$MANO_DIR" ]]; then
  for candidate in \
    "$ROOT/mano_v1_2/models" \
    "$ROOT/licensed_models/mano" \
    "$HOME/Downloads/mano_v1_2/models"; do
    if [[ -f "$candidate/MANO_RIGHT.pkl" && -f "$candidate/MANO_LEFT.pkl" ]]; then
      MANO_DIR="$candidate"
      break
    fi
  done
fi

verify_sha256() {
  local path="$1"
  local expected="$2"
  [[ -f "$path" ]] || return 1
  [[ "$(shasum -a 256 "$path" | awk '{print $1}')" == "$expected" ]]
}

if [[ -z "$MANO_DIR" ]]; then
  mkdir -p "$MANO_DOWNLOAD_DIR"
  if ! verify_sha256 "$MANO_DOWNLOAD_DIR/MANO_LEFT.pkl" "$MANO_LEFT_SHA256"; then
    rm -f "$MANO_DOWNLOAD_DIR/MANO_LEFT.pkl"
    echo "Downloading licensed MANO_LEFT.pkl…"
    "$CONDA_BIN" run --no-capture-output -n gradyn-core gdown \
      "https://drive.google.com/uc?id=$MANO_LEFT_ID" \
      --output "$MANO_DOWNLOAD_DIR/MANO_LEFT.pkl"
  fi
  if ! verify_sha256 "$MANO_DOWNLOAD_DIR/MANO_RIGHT.pkl" "$MANO_RIGHT_SHA256"; then
    rm -f "$MANO_DOWNLOAD_DIR/MANO_RIGHT.pkl"
    echo "Downloading licensed MANO_RIGHT.pkl…"
    "$CONDA_BIN" run --no-capture-output -n gradyn-core gdown \
      "https://drive.google.com/uc?id=$MANO_RIGHT_ID" \
      --output "$MANO_DOWNLOAD_DIR/MANO_RIGHT.pkl"
  fi
  MANO_DIR="$MANO_DOWNLOAD_DIR"
fi

if ! verify_sha256 "$MANO_DIR/MANO_LEFT.pkl" "$MANO_LEFT_SHA256" ||
  ! verify_sha256 "$MANO_DIR/MANO_RIGHT.pkl" "$MANO_RIGHT_SHA256"; then
  echo "MANO files failed integrity verification: $MANO_DIR" >&2
  exit 2
fi

"$CONDA_BIN" run -n gradyn-core gradyn models install-mano \
  --right "$MANO_DIR/MANO_RIGHT.pkl" \
  --left "$MANO_DIR/MANO_LEFT.pkl"

verify_args=()
if [[ "$WITH_QWEN" -eq 1 ]]; then
  verify_args+=(--with-qwen)
fi
"$CONDA_BIN" run -n gradyn-core gradyn models verify "${verify_args[@]}"
"$CONDA_BIN" run -n gradyn-core python -m pytest -q "$ROOT/tests"

echo
echo "Gradyn setup complete."
echo "Run: ./gradyn doctor"
echo "Then: ./gradyn process video.mp4 --camera \"DJI Osmo Nano\" --objects \"object one,object two\" --output result"
