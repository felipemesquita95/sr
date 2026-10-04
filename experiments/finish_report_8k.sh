#!/usr/bin/env bash
# Espera os treinos em execução e gera o relatório quando os artefatos estiverem completos.
set -euo pipefail
cd "$(dirname "$0")/.."

while systemctl --user is-active --quiet sr-brsd-nativo.service || \
      systemctl --user is-active --quiet sr-rebuild-8k-minimo.service; do
    sleep 30
done

KERAS_BACKEND=torch .venv/bin/python experiments/final_report_8k.py > runs/relatorio_final_8k.log 2>&1
