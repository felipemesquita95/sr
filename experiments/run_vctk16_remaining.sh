#!/usr/bin/env bash
set -euo pipefail

cd /home/lsmsqt/Documents/sr
export MPLCONFIGDIR=/home/lsmsqt/Documents/sr/tmp/matplotlib
mkdir -p "$MPLCONFIGDIR"
exec .venv/bin/python -u experiments/run_vctk16_remaining.py
