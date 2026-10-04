#!/usr/bin/env bash
# Controle de duração: 77 quadros nas mesmas 18.067 gravações do teste de 153.
set -euo pipefail

cd "$(dirname "$0")/.."
echo "Aguardando treino de 153 quadros e avaliação cross-mic terminarem."
while systemctl --user is-active --quiet sr-vctk-153-queued.service \
   || systemctl --user is-active --quiet sr-vctk-cross-queued.service; do
  sleep 30
done

mountpoint -q /media/lsmsqt/HDD
python3 - <<'PY'
import json
from pathlib import Path

for mic in ('mic1', 'mic2'):
    for arch in ('cnn', 'temporal_cnn', 'attention'):
        for fold in range(1, 6):
            output = Path('runs/models') / f'vctk153_{mic}' / arch / f'particao{fold}'
            division = json.loads((output / 'divisao.json').read_text())
            metrics = json.loads((output / 'metricas.json').read_text())
            if not ((output / 'modelo.keras').is_file()
                    and division['particao'] == fold
                    and division['validation_fold_offset'] == 1
                    and division['teste'] == metrics['num_amostras_teste']):
                raise RuntimeError(f'Partição de 153 incompleta: {output}')
print('As 30 partições de 153 quadros estão completas.')
PY

export KERAS_BACKEND=torch
echo "Iniciando controle 77 quadros com seleção >=153: mic1."
.venv/bin/python experiments/run_experiment.py --resume --config configs/vctk77_matched153_mic1.env
echo "Iniciando controle 77 quadros com seleção >=153: mic2."
.venv/bin/python experiments/run_experiment.py --resume --config configs/vctk77_matched153_mic2.env
.venv/bin/python experiments/report_matched_77_153.py
echo "Controle pareado de 77 quadros concluído."
