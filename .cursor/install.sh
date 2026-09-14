#!/usr/bin/env bash
# Idempotent Cloud Agent setup for the Windows Game RL Agent project.
# System packages: venv/build headers plus the shared libraries OpenCV,
# soundfile/librosa, and ffmpeg-backed audio decoding need at runtime.
# Python packages are installed into a project-local .venv from the pinned
# lockfile (requirements-dev.txt includes runtime requirements plus pytest/ruff).
set -euo pipefail

cd "$(dirname "$0")/.."

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  python3.12-venv \
  python3-dev \
  libgl1 \
  libglib2.0-0 \
  libsndfile1 \
  ffmpeg

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi

./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/pip install -r requirements-dev.txt
