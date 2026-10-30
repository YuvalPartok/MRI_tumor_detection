#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Download the UTSW-Glioma data (metadata + selected MRI files) from TCIA.
#
# The images are only distributed through IBM Aspera, so this script makes sure the
# open-source Aspera CLI (`ascli`) is installed, then runs scripts/download_data.py,
# which downloads only the files this project needs for data.num_patients patients.
#
# Usage:  bash scripts/download_data.sh [--num-patients N]
# -----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

# macOS ships Ruby 2.6, but aspera-cli needs Ruby >= 3.1 -> use Homebrew Ruby.
if ! command -v ascli >/dev/null 2>&1; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required to install Ruby/aspera-cli: https://brew.sh" >&2
    exit 1
  fi
  RUBY_PREFIX="$(brew --prefix ruby 2>/dev/null || true)"
  if [ ! -x "${RUBY_PREFIX}/bin/ruby" ]; then
    echo ">> Installing Homebrew Ruby"
    brew install ruby
    RUBY_PREFIX="$(brew --prefix ruby)"
  fi
  export PATH="${RUBY_PREFIX}/bin:$PATH"
  GEM_BIN="$(gem environment | awk -F': ' '/EXECUTABLE DIRECTORY/{print $2}')"
  export PATH="${GEM_BIN}:$PATH"
  if ! command -v ascli >/dev/null 2>&1; then
    echo ">> Installing aspera-cli"
    gem install aspera-cli
  fi
fi

# Aspera's transfer engine (ascp) is installed once by ascli.
if ! ascli config ascp info >/dev/null 2>&1; then
  echo ">> Installing the Aspera transfer engine"
  ascli config ascp install
fi

PYTHON="${PYTHON:-.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3
"$PYTHON" scripts/download_data.py "$@"
