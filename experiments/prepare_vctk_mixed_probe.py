#!/usr/bin/env python3
"""Monta controles mistos de atividade e baixa atividade com largura pareada."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from prepare_vctk_activity_probe import (CANDIDATES, FEATURES, ROOT,
                                         config_text, select_even)

SUMMARY = ROOT / 'docs/vctk_mixed_probe_selection.json'


def digest(records: list[dict]) -> str:
    keys = [(row['locutor'], row['enunciado']) for row in records]
    return hashlib.sha256(json.dumps(keys, separators=(',', ':')).encode()).hexdigest()


def coverage(records: list[dict]) -> dict:
    per_speaker = Counter(row['locutor'] for row in records)
    per_group = Counter((row['locutor'], (row['enunciado'] - 1) % 5) for row in records)
    return {
        'pares': len(records),
        'locutores': sum(per_speaker.get(s, 0) > 0 for s in range(1, 109)),
        'minimo_por_locutor': min(per_speaker.get(s, 0) for s in range(1, 109)),
        'minimo_por_grupo_de_teste': min(per_group.get((s, f), 0)
                                         for s in range(1, 109) for f in range(5)),
        'sha256_gravacoes': digest(records),
    }


def frames_for(record: dict, condition: str) -> np.ndarray:
    if condition == 'mixed10':
        active = select_even(record['active'], 5)
        low = select_even(record['low'], 5)
    elif condition == 'active20':
        return select_even(record['active'], 20)
    elif condition == 'mixed20':
        active = select_even(record['active'], 10)
        low = select_even(record['low'], 10)
    else:
        raise ValueError(condition)
    return np.sort(np.concatenate((active, low)))


def main() -> None:
    records = [json.loads(line) for line in CANDIDATES.open()]
    if len(records) != 21_523:
        raise ValueError(f'Esperados 21.523 pares, encontrados {len(records)}.')
    cohort10 = [row for row in records
                if len(row['active']) >= 10 and len(row['low']) >= 10]
    cohort20 = [row for row in records
                if len(row['active']) >= 20 and len(row['low']) >= 10]
    original = json.loads((ROOT / 'docs/vctk_activity_probe_selection.json').read_text())
    if digest(cohort10) != original['sha256_gravacoes']:
        raise ValueError('O grupo misto de 10 não coincide com o controle anterior.')
    if coverage(cohort20)['minimo_por_grupo_de_teste'] < 3:
        raise ValueError('O grupo de 20 perdeu cobertura nas cinco partições.')

    conditions = {
        'mixed10': (cohort10, 10, 'vctk_activity10', 'mixed'),
        'active20': (cohort20, 20, 'vctk_activity20', 'active'),
        'mixed20': (cohort20, 20, 'vctk_activity20', 'mixed'),
    }
    for condition, (cohort, k, prefix, label) in conditions.items():
        print(f'Preparando {condition}: {len(cohort)} pares de gravações.', flush=True)
        for position, row in enumerate(cohort, 1):
            selected = frames_for(row, condition)
            speaker = str(row['locutor'])
            utterance = str(row['enunciado'])
            for mic in ('mic1', 'mic2'):
                source = FEATURES / f'vctk8k_{mic}' / speaker / utterance / 'mfccs.npy'
                target = FEATURES / f'{prefix}_{label}_{mic}' / speaker / utterance / 'mfccs.npy'
                if target.is_file():
                    continue
                matrix = np.load(source, mmap_mode='r')
                if matrix.shape != (40, row['quadros']):
                    raise ValueError(f'MFCC inesperado: {source}: {matrix.shape}')
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix('.tmp.npy')
                np.save(temporary, np.asarray(matrix[:, selected], dtype=np.float32))
                temporary.replace(target)
            if position % 1000 == 0:
                print(f'{condition}: {position}/{len(cohort)} pares', flush=True)
        for mic in ('mic1', 'mic2'):
            path = ROOT / 'configs' / f'{prefix}_{label}_{mic}.env'
            path.write_text(config_text(label, mic, k, prefix))

    summary = {
        'coorte_10': coverage(cohort10),
        'coorte_20': coverage(cohort20),
        'condicoes': {
            'mixed10': '5 quadros de atividade + 5 de baixa atividade, ordenados no tempo',
            'active20': '20 quadros de atividade, ordenados no tempo',
            'mixed20': '10 quadros de atividade + 10 de baixa atividade, ordenados no tempo',
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(SUMMARY, flush=True)


if __name__ == '__main__':
    main()
