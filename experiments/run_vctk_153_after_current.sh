#!/usr/bin/env bash
# Aguarda a reexecução VCTK de 77 quadros e inicia a variante de 153.
set -euo pipefail

cd "$(dirname "$0")/.."
echo "Aguardando sr-vctk-602020.service terminar."
while systemctl --user is-active --quiet sr-vctk-602020.service; do
  sleep 30
done

mountpoint -q /media/lsmsqt/HDD
python3 - <<'PY'
import json
from pathlib import Path

for mic in ('vctk8k_mic1', 'vctk8k_mic2'):
    for arch in ('cnn', 'temporal_cnn', 'attention'):
        for fold in range(1, 6):
            output = Path('runs/models') / mic / arch / f'particao{fold}'
            division = json.loads((output / 'divisao.json').read_text())
            metrics = json.loads((output / 'metricas.json').read_text())
            if not ((output / 'modelo.keras').is_file()
                    and division['particao'] == fold
                    and division['validation_fold_offset'] == 1
                    and division['teste'] == metrics['num_amostras_teste']):
                raise RuntimeError(f'Partição anterior incompleta: {output}')
print('As 30 partições de 77 quadros estão concluídas.')
PY

export KERAS_BACKEND=torch
echo "Iniciando VCTK mic1 com 153 quadros."
.venv/bin/python experiments/run_experiment.py --resume --config configs/vctk153_mic1.env
echo "Iniciando VCTK mic2 com 153 quadros."
.venv/bin/python experiments/run_experiment.py --resume --config configs/vctk153_mic2.env
echo "As 30 partições de 153 quadros foram concluídas."
