#!/usr/bin/env bash
# Reusa os MFCCs existentes, refaz as redes VCTK com 60/20/20 e publica o relatório.
set -euo pipefail

cd "$(dirname "$0")/.."
mountpoint -q /media/lsmsqt/HDD
test -f runs/features/relatorio_comprimentos_8k.json

export KERAS_BACKEND=torch
.venv/bin/python experiments/run_experiment.py --resume --config configs/vctk8k.env
.venv/bin/python experiments/run_experiment.py --resume --config configs/vctk8k_mic2.env
.venv/bin/python experiments/final_report_8k.py
