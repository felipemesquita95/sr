#!/usr/bin/env bash
set -euo pipefail

cd /home/lsmsqt/Documents/sr
ready=output/vctk16_xvector_results/static/fold1/mic2/validation_metrics.json
model=output/vctk16_xvector_results/static/fold1/mic2/modelo.keras
until [[ -s "$ready" && -s "$model" ]]; do
    sleep 15
done
export MPLCONFIGDIR=/home/lsmsqt/Documents/sr/tmp/matplotlib
mkdir -p "$MPLCONFIGDIR"
exec .venv/bin/python -u experiments/finish_vctk16_xvector.py
