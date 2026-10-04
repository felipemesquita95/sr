#!/usr/bin/env bash
# Diagnóstico pareado: sem seleção, atividade e baixa atividade em ambos os microfones.
set -euo pipefail

cd "$(dirname "$0")/.."
mountpoint -q /media/lsmsqt/HDD
export KERAS_BACKEND=torch
export MPLCONFIGDIR=/tmp/sr-mpl

if [ -f docs/vctk_activity8k_candidates.jsonl.tmp ] \
   && [ ! -f docs/vctk_activity8k_candidates.jsonl ]; then
  echo "Aguardando a varredura de áudio já em andamento."
  for attempt in $(seq 1 120); do
    [ -f docs/vctk_activity8k_candidates.jsonl ] && break
    sleep 30
  done
fi
if [ ! -f docs/vctk_activity8k_candidates.jsonl ]; then
  if [ -f docs/vctk_activity8k_candidates.jsonl.tmp ]; then
    echo "Varredura existente não terminou; verificar processo de extração." >&2
    exit 1
  fi
  ionice -c3 nice -n 10 .venv/bin/python experiments/prepare_vctk_activity_probe.py scan
fi
ionice -c3 nice -n 10 .venv/bin/python experiments/prepare_vctk_activity_probe.py build

prefix=$(python3 -c 'import json; print(json.load(open("docs/vctk_activity_probe_selection.json"))["prefix"])')
for condition in unfiltered active low; do
  for mic in mic1 mic2; do
    echo "Treinando ${condition} ${mic} (${prefix})."
    .venv/bin/python experiments/run_activity_probe.py \
      --config "configs/${prefix}_${condition}_${mic}.env"
  done
done

for condition in unfiltered active low; do
  echo "Avaliando cross-mic ${condition} (${prefix})."
  .venv/bin/python experiments/evaluate_cross_8k.py \
    --variant "${prefix}_${condition}"
done

.venv/bin/python experiments/report_activity_probe.py
echo "Diagnóstico de atividade/baixa atividade concluído."
