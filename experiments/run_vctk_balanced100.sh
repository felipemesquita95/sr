#!/usr/bin/env bash
# Controles pareados em 100 locutores × 25 gravações, com 60/20/20 exato.
set -euo pipefail

cd "$(dirname "$0")/.."
mountpoint -q /media/lsmsqt/HDD
export KERAS_BACKEND=torch
export MPLCONFIGDIR=/tmp/sr-mpl

ionice -c3 nice -n 10 .venv/bin/python experiments/prepare_vctk_balanced100.py

for condition in active20 low20 unfiltered40 active40 mixed40; do
  for mic in mic1 mic2; do
    echo "Treinando vctk_bal100_${condition}_${mic}."
    .venv/bin/python experiments/run_activity_probe.py \
      --config "configs/vctk_bal100_${condition}_${mic}.env"
  done
done

for condition in active20 low20 unfiltered40 active40 mixed40; do
  echo "Avaliando transferência vctk_bal100_${condition}."
  .venv/bin/python experiments/evaluate_cross_8k.py \
    --variant "vctk_bal100_${condition}"
done

.venv/bin/python experiments/report_vctk_balanced100.py
echo "Experimento balanceado de 100 locutores concluído."
