#!/usr/bin/env python3
"""Usa todos os quadros por condição em uma referência estática pareada."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from prepare_vctk_activity_probe import CANDIDATES, FEATURES, ROOT, select_even

OUTPUT = ROOT / 'docs/resultado_todos_quadros_vctk.md'
DATA = ROOT / 'docs/resultado_todos_quadros_vctk.json'
CONDITIONS = ('unfiltered_all', 'active_all', 'low_all', 'active20', 'mixed20')
LABELS = {
    'unfiltered_all': 'Todos, sem seleção',
    'active_all': 'Todos de atividade',
    'low_all': 'Todos de baixa atividade',
    'active20': '20 de atividade',
    'mixed20': '10 atividade + 10 baixa',
}


def pool(matrix: np.ndarray, indices: np.ndarray | None) -> np.ndarray:
    values = matrix if indices is None else matrix[:, indices]
    return np.concatenate((values.mean(axis=1), values.std(axis=1))).astype(np.float32)


def percent(value: float) -> str:
    return f'{value*100:.2f}'.replace('.', ',') + '%'


def main() -> None:
    records = [json.loads(line) for line in CANDIDATES.open()]
    selected = [row for row in records if len(row['active']) >= 20 and len(row['low']) >= 10]
    if len(selected) != 17_270:
        raise ValueError(f'Seleção inesperada: {len(selected)} gravações.')
    arrays = {mic: {condition: np.empty((len(selected), 80), dtype=np.float32)
                    for condition in CONDITIONS} for mic in ('mic1', 'mic2')}
    frame_counts = defaultdict(list)
    labels = np.empty(len(selected), dtype=np.int32)
    groups = np.empty(len(selected), dtype=np.int32)
    for index, row in enumerate(selected):
        labels[index] = row['locutor'] - 1
        groups[index] = (row['enunciado'] - 1) % 5
        active = np.asarray(row['active'], dtype=np.int64)
        low = np.asarray(row['low'], dtype=np.int64)
        active20 = select_even(row['active'], 20)
        mixed20 = np.sort(np.concatenate((select_even(row['active'], 10),
                                          select_even(row['low'], 10))))
        indices = {'unfiltered_all': None, 'active_all': active,
                   'low_all': low, 'active20': active20, 'mixed20': mixed20}
        frame_counts['unfiltered_all'].append(row['quadros'])
        frame_counts['active_all'].append(len(active))
        frame_counts['low_all'].append(len(low))
        frame_counts['active20'].append(20)
        frame_counts['mixed20'].append(20)
        for mic in ('mic1', 'mic2'):
            path = FEATURES / f'vctk8k_{mic}' / str(row['locutor']) / str(row['enunciado']) / 'mfccs.npy'
            matrix = np.load(path, mmap_mode='r')
            if matrix.shape != (40, row['quadros']):
                raise ValueError(f'MFCC inesperado: {path}: {matrix.shape}')
            for condition in CONDITIONS:
                arrays[mic][condition][index] = pool(matrix, indices[condition])
        if (index + 1) % 2000 == 0:
            print(f'Resumidos {index + 1}/{len(selected)} pares', flush=True)

    results = []
    for fold in range(5):
        test = groups == fold
        validation = groups == ((fold + 1) % 5)
        train = ~(test | validation)
        if min(train.sum(), validation.sum(), test.sum()) <= 0:
            raise ValueError(f'Divisão vazia na partição {fold + 1}.')
        for source, target in (('mic1', 'mic2'), ('mic2', 'mic1')):
            for condition in CONDITIONS:
                scaler = StandardScaler().fit(arrays[source][condition][train])
                classifier = RidgeClassifier(alpha=1.0).fit(
                    scaler.transform(arrays[source][condition][train]), labels[train])
                same = classifier.predict(scaler.transform(arrays[source][condition][test]))
                other = classifier.predict(scaler.transform(arrays[target][condition][test]))
                for tested_mic, predictions in ((source, same), (target, other)):
                    results.append({
                        'fold': fold + 1, 'origem': source, 'teste_em': tested_mic,
                        'condicao': condition, 'amostras_teste': int(test.sum()),
                        'acuracia': float(accuracy_score(labels[test], predictions)),
                        'f1_macro': float(f1_score(labels[test], predictions,
                                                   labels=range(108), average='macro',
                                                   zero_division=0)),
                    })
            print(f'Partição {fold + 1}/5, origem {source} concluída', flush=True)

    summary = {
        'classificador': 'StandardScaler no treino da origem + RidgeClassifier(alpha=1.0)',
        'numero_gravacoes': len(selected), 'numero_locutores': 108,
        'quadros_por_condicao': {
            condition: {'mediana': int(np.median(values)),
                        'media': float(np.mean(values)),
                        'minimo': int(min(values)), 'maximo': int(max(values))}
            for condition, values in frame_counts.items()},
        'resultados': results,
    }
    DATA.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')

    lines = [
        '# Todos os quadros de atividade e baixa atividade — VCTK', '',
        'Mesmas 17.270 gravações pareadas, 108 locutores e cinco partições em',
        'todas as condições. Cada gravação vira um vetor de 80 números:',
        'média e desvio dos 40 MFCCs calculados sobre **todos os quadros** da',
        'condição indicada. O classificador linear é o mesmo em todas as linhas.',
        'A normalização é ajustada apenas no treino do microfone de origem.',
        'O detector é de energia; baixa atividade não significa silêncio puro.', '',
        '| Condição | Quadros medianos por gravação | Mesmo mic1 | Mesmo mic2 | Mic1 → Mic2 | Mic2 → Mic1 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for condition in CONDITIONS:
        values = {}
        for source, target in (('mic1', 'mic1'), ('mic2', 'mic2'),
                               ('mic1', 'mic2'), ('mic2', 'mic1')):
            matching = [row['acuracia'] for row in results
                        if row['condicao'] == condition and row['origem'] == source
                        and row['teste_em'] == target]
            if len(matching) != 5:
                raise ValueError(f'Faltam partições para {condition} {source} → {target}.')
            values[(source, target)] = float(np.mean(matching))
        lines.append(f'| {LABELS[condition]} | '
                     f'{summary["quadros_por_condicao"][condition]["mediana"]} | '
                     + ' | '.join(percent(values[key]) for key in
                                  (('mic1', 'mic1'), ('mic2', 'mic2'),
                                   ('mic1', 'mic2'), ('mic2', 'mic1'))) + ' |')
    lines += ['', 'Acaso: 0,93% em 108 classes. As condições de 20 quadros usam',
              'os mesmos registros e o mesmo classificador das condições com',
              'todos os quadros; diferenças de duração são parte do efeito medido.',
              'A seleção de quadros de baixa atividade ainda pode conter fala fraca,',
              'respiração e ruído.', '']
    OUTPUT.write_text('\n'.join(lines))
    print(OUTPUT)


if __name__ == '__main__':
    main()
