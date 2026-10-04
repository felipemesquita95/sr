#!/usr/bin/env python3
"""Prepara 100 locutores × 25 gravações para controles de atividade em 8 kHz."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from prepare_vctk_activity_probe import (CANDIDATES, FEATURES, ROOT,
                                         config_text, select_even)


PREFIX = 'vctk_bal100'
MANIFEST = ROOT / 'docs/vctk_bal100_selection.json'
LOW_MIN = 20
ACTIVE_MIN = 40
SPEAKERS = 100
RECORDINGS = 25
SEED = 42
CONDITIONS = {
    'active20': 20,
    'low20': 20,
    'unfiltered40': 40,
    'active40': 40,
    'mixed40': 40,
}


def indices(row: dict, condition: str) -> np.ndarray:
    if condition == 'active20':
        return select_even(row['active'], 20)
    if condition == 'low20':
        return select_even(row['low'], 20)
    if condition == 'unfiltered40':
        return select_even(list(range(row['quadros'])), 40)
    if condition == 'active40':
        return select_even(row['active'], 40)
    if condition == 'mixed40':
        return np.sort(np.concatenate((select_even(row['active'], 20),
                                       select_even(row['low'], 20))))
    raise ValueError(condition)


def settings_text(condition: str, mic: str, frames: int) -> str:
    base = config_text(condition, mic, frames, PREFIX)
    base = base.replace('NUM_SPEAKERS=108', f'NUM_SPEAKERS={SPEAKERS}')
    base = base.replace('NUM_UTTERANCES=200', f'NUM_UTTERANCES={RECORDINGS}')
    return base


def main() -> None:
    candidates = [json.loads(line) for line in CANDIDATES.open()]
    if len(candidates) != 21_523:
        raise ValueError('Arquivo de candidatos incompleto.')
    by_speaker: dict[int, list[dict]] = {}
    for row in candidates:
        if len(row['low']) >= LOW_MIN and len(row['active']) >= ACTIVE_MIN:
            by_speaker.setdefault(row['locutor'], []).append(row)
    kept = sorted(s for s in range(1, 109)
                  if len(by_speaker.get(s, [])) >= RECORDINGS)
    if len(kept) != SPEAKERS:
        raise ValueError(f'Esperados 100 locutores elegíveis, encontrados {len(kept)}.')

    rng = np.random.default_rng(SEED)
    chosen: list[dict] = []
    for new_speaker, original_speaker in enumerate(kept, 1):
        available = sorted(by_speaker[original_speaker], key=lambda r: r['enunciado'])
        positions = sorted(rng.choice(len(available), size=RECORDINGS,
                                      replace=False).tolist())
        for new_utterance, position in enumerate(positions, 1):
            original = available[position]
            chosen.append({**original, 'novo_locutor': new_speaker,
                           'novo_enunciado': new_utterance})
    assert len(chosen) == SPEAKERS * RECORDINGS

    roles = Counter((row['novo_locutor'],
                     (row['novo_enunciado'] - 1) % 5) for row in chosen)
    if set(roles.values()) != {5} or len(roles) != SPEAKERS * 5:
        raise ValueError('Divisão 60/20/20 não tem cinco gravações por grupo.')

    for condition, frames in CONDITIONS.items():
        for mic in ('mic1', 'mic2'):
            target_root = FEATURES / f'{PREFIX}_{condition}_{mic}'
            for row in chosen:
                source = (FEATURES / f'vctk8k_{mic}' /
                          str(row['locutor']) / str(row['enunciado']) / 'mfccs.npy')
                target = (target_root / str(row['novo_locutor']) /
                          str(row['novo_enunciado']) / 'mfccs.npy')
                if target.is_file():
                    current = np.load(target, mmap_mode='r')
                    if current.shape != (40, frames):
                        raise ValueError(f'Feature existente incompatível: {target}')
                    continue
                matrix = np.load(source, mmap_mode='r')
                if matrix.shape != (40, row['quadros']):
                    raise ValueError(f'Feature original incompatível: {source}')
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix('.tmp.npy')
                np.save(temporary, np.asarray(matrix[:, indices(row, condition)],
                                              dtype=np.float32))
                temporary.replace(target)
            (ROOT / 'configs' / f'{PREFIX}_{condition}_{mic}.env').write_text(
                settings_text(condition, mic, frames))
            print(f'{condition} {mic}: {SPEAKERS * RECORDINGS} gravações.', flush=True)

    mapping = [{'novo_locutor': row['novo_locutor'],
                'novo_enunciado': row['novo_enunciado'],
                'locutor_original': row['locutor'],
                'enunciado_original': row['enunciado'],
                'arquivo_original': row['arquivo'],
                'quadros_atividade': len(row['active']),
                'quadros_baixa_atividade': len(row['low'])} for row in chosen]
    digest = hashlib.sha256(json.dumps(mapping, sort_keys=True,
                                       separators=(',', ':')).encode()).hexdigest()
    result = {
        'metodo': 'VAD pareado em 8 kHz; baixa >=20; atividade >=40',
        'semente_amostragem': SEED,
        'locutores': SPEAKERS,
        'gravacoes_por_locutor': RECORDINGS,
        'total_gravacoes': len(chosen),
        'divisao_por_locutor_em_cada_particao': {'treino': 15,
                                                'validacao': 5, 'teste': 5},
        'locutores_originais_incluidos': kept,
        'locutores_originais_excluidos': sorted(set(range(1, 109)) - set(kept)),
        'sha256_mapeamento': digest,
        'condicoes': CONDITIONS,
        'mapeamento': mapping,
    }
    MANIFEST.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(MANIFEST, flush=True)


if __name__ == '__main__':
    main()
