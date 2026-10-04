#!/usr/bin/env bash
# Testa misturas com e sem baixa atividade, mantendo duração e gravações pareadas.
set -euo pipefail

cd "$(dirname "$0")/.."
mountpoint -q /media/lsmsqt/HDD
export KERAS_BACKEND=torch
export MPLCONFIGDIR=/tmp/sr-mpl

ionice -c3 nice -n 10 .venv/bin/python experiments/prepare_vctk_mixed_probe.py

for variant in vctk_activity10_mixed vctk_activity20_active vctk_activity20_mixed; do
  for mic in mic1 mic2; do
    echo "Treinando ${variant}_${mic}."
    .venv/bin/python experiments/run_activity_probe.py \
      --config "configs/${variant}_${mic}.env"
  done
done

for variant in vctk_activity10_mixed vctk_activity20_active vctk_activity20_mixed; do
  echo "Avaliando transferência ${variant}."
  .venv/bin/python experiments/evaluate_cross_8k.py --variant "${variant}"
done

.venv/bin/python experiments/report_vctk_mixed_probe.py
echo "Controles de atividade + baixa atividade concluídos."
