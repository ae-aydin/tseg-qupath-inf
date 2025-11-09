#!/usr/bin/env bash
set -Eeuo pipefail

log_info()    { echo "[INFO] $1"; }
log_success() { echo "[SUCCESS] $1"; }
log_warn()    { echo "[WARN] $1"; }
log_error()   { echo "[ERROR] $1"; exit 1; }

trap 'log_error "Failed at line $LINENO: ${BASH_COMMAND:-unknown}"' ERR

# dirs
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "=== QUPATH INFERENCE SETUP ===\n"

# prereqs
command -v curl >/dev/null 2>&1 || log_error "curl not found."

log_info "Checking for 'uv'..."
if ! command -v uv >/dev/null 2>&1; then
  log_warn "'uv' not found. Installing..."
  # install uv (https://astral.sh/uv)
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

command -v uv >/dev/null 2>&1 || log_error "'uv' not on PATH after install."

UV_VERSION="$(uv --version || true)"
log_success "uv verification successful. Version: ${UV_VERSION:-unknown}"

# venv
log_info "Setting up Python project..."
if [ ! -d ".venv" ]; then
  log_info "Creating virtual environment (.venv)..."
  uv venv -q >/dev/null
  log_success "Virtual environment created."
else
  log_info "Existing .venv found."
fi

export VIRTUAL_ENV="$(pwd)/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# dependencies
log_info "Looking for dependency files..."
if [ -f "pyproject.toml" ]; then
  log_info "Found pyproject.toml. Installing with 'uv sync'..."
  uv sync -q
  log_success "Dependencies installed via pyproject."
elif [ -f "requirements.txt" ]; then
  log_info "Found requirements.txt. Installing into venv..."
  uv add -r requirements.txt -q
  log_success "Dependencies from requirements.txt installed."
else
  log_error "No pyproject.toml or requirements.txt found."
fi

log_success "Project setup complete."

echo
