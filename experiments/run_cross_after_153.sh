#!/usr/bin/env bash
# Mede a transferência dos modelos de 77 e 153 quadros após os treinos.
set -euo pipefail

cd "$(dirname "$0")/.."
echo "Aguardando sr-vctk-153-queued.service terminar."
while systemctl --user is-active --quiet sr-vctk-153-queued.service; do
  sleep 30
done

mountpoint -q /media/lsmsqt/HDD
python3 - <<'PY'
import json
from pathlib import Path

for prefix in ('vctk8k', 'vctk153'):
    for mic in ('mic1', 'mic2'):
        for arch in ('cnn', 'temporal_cnn', 'attention'):
            for fold in range(1, 6):
                output = Path('runs/models') / f'{prefix}_{mic}' / arch / f'particao{fold}'
                division = json.loads((output / 'divisao.json').read_text())
                metrics = json.loads((output / 'metricas.json').read_text())
                if not ((output / 'modelo.keras').is_file()
                        and division['particao'] == fold
                        and division['validation_fold_offset'] == 1
                        and division['teste'] == metrics['num_amostras_teste']):
                    raise RuntimeError(f'Partição incompleta: {output}')
print('As 60 partições intra-microfone estão concluídas.')
PY

export KERAS_BACKEND=torch
echo "Avaliando transferência entre microfones para 77 quadros."
.venv/bin/python experiments/evaluate_cross_8k.py --variant vctk8k
echo "Avaliando transferência entre microfones para 153 quadros."
.venv/bin/python experiments/evaluate_cross_8k.py --variant vctk153
echo "Avaliação cross mic concluída para as duas durações."
