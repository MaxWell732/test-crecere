# Source before any uv command:   source scripts/env.sh
# Confines the interpreter, uv cache, model downloads and temp files to this project directory.
# (The `extract` CLI also forces the model/temp variables itself, so `uv run extract ...` stays
#  confined even if this file was not sourced.)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/.cache/python"
export UV_PYTHON_BIN_DIR="$PROJECT_ROOT/.cache/python-bin"
export UV_PYTHON_INSTALL_BIN=0
export UV_TOOL_DIR="$PROJECT_ROOT/.cache/uv-tools"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export TORCH_HOME="$PROJECT_ROOT/.cache/torch"
export XDG_CACHE_HOME="$PROJECT_ROOT/.cache/xdg"
export MPLCONFIGDIR="$PROJECT_ROOT/.cache/matplotlib"
export TMPDIR="$PROJECT_ROOT/.cache/tmp"
mkdir -p "$TMPDIR"
