#!/usr/bin/env python3
"""Gera o relatório comparativo após os três experimentos de 8 kHz terminarem."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import librosa
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dotenv import dotenv_values
from scipy.signal import welch
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from sr.preprocessing import signal as dsp  # noqa: E402
from sr.config import load_settings  # noqa: E402
from sr.features import FeatureAdjustmentSubsystem  # noqa: E402

FEATURES = ROOT / 'runs/features'
MODELS = ROOT / 'runs/models'
DOC = ROOT / 'docs/relatorio_8k.md'
FIGURES = ROOT / 'docs/figuras/relatorio_8k'
TRACKS = ('brsd', 'vctk8k_mic1', 'vctk8k_mic2')
ARCHITECTURES = ('cnn', 'temporal_cnn', 'attention')
PARAMETERS = ('SOURCE_SAMPLING_RATE', 'TARGET_SAMPLING_RATE', 'ENABLE_VAD',
              'VAD_TOP_DB', 'PRE_EMPHASIS_COEF', 'NUM_MFCCS', 'FRAME_SIZE')


def read_inputs():
    lengths = json.loads((FEATURES / 'relatorio_comprimentos_8k.json').read_text())
    if any(track not in lengths for track in TRACKS) or 'vctk_combinado' not in lengths:
        raise RuntimeError('Relatório de comprimentos incompleto; aguarde as extrações.')
    profiles = {track: dotenv_values(ROOT / 'configs' / (
        'brsd.env' if track == 'brsd' else 'vctk8k.env' if track.endswith('mic1')
        else 'vctk8k_mic2.env')) for track in TRACKS}
    reference = tuple(profiles['brsd'][key] for key in PARAMETERS)
    if any(tuple(profile[key] for key in PARAMETERS) != reference for profile in profiles.values()):
        raise RuntimeError('Os parâmetros de extração diferem entre os corpora.')
    results = {}
    for track in TRACKS:
        for architecture in ARCHITECTURES:
            path = MODELS / track / architecture / 'resumo.json'
            if not path.exists():
                raise RuntimeError(f'Treino pendente: {path}')
            result = json.loads(path.read_text())
            if len(result['acuracias']) != 5:
                raise RuntimeError(f'Partições incompletas: {path}')
            if track != 'brsd':
                for fold in range(1, 6):
                    partition_path = MODELS / track / architecture / f'particao{fold}' / 'divisao.json'
                    if not partition_path.exists():
                        raise RuntimeError(f'Treino 60/20/20 pendente: {partition_path}')
                    partition = json.loads(partition_path.read_text())
                    if (partition.get('validation_fold_offset') != 1 or
                            partition.get('validacao', 0) < 4000 or
                            partition.get('treino', 0) < 12000):
                        raise RuntimeError(f'Divisão VCTK incorreta: {partition_path}')
            results[(track, architecture)] = result
    return lengths, profiles, results


def count_model_parameters(lengths):
    from sr.models.registry import build_model

    sizes = {}
    for track in TRACKS:
        frames = (lengths['brsd']['menor']['quadros'] if track == 'brsd'
                  else lengths['vctk_combinado']['menor']['quadros'])
        classes = 80 if track == 'brsd' else 108
        for architecture in ARCHITECTURES:
            model = build_model(architecture, (40, frames), classes, .001)
            files = [MODELS / track / architecture / f'particao{fold}' / 'modelo.keras'
                     for fold in range(1, 6)]
            if any(not file.is_file() for file in files):
                raise RuntimeError(f'Modelo ausente: {track}/{architecture}')
            mean_mib = sum(file.stat().st_size for file in files) / len(files) / 2**20
            sizes[(track, architecture)] = (frames, model.count_params(), mean_mib)
    return sizes


def write_splits():
    by_track = defaultdict(lambda: defaultdict(list))
    with (FEATURES / 'relatorio_comprimentos_8k.csv').open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            by_track[row['trilha']][int(row['locutor'])].append(int(row['enunciado']))
    output = FEATURES / 'relatorio_particoes_8k.csv'
    counts = defaultdict(Counter)
    chosen = defaultdict(dict)
    speaker_counts = defaultdict(list)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['trilha', 'locutor', 'enunciado', 'particao', 'uso_em_todas_as_redes'])
        for track in TRACKS:
            profile = ('brsd.env' if track == 'brsd' else 'vctk8k.env' if track.endswith('mic1')
                       else 'vctk8k_mic2.env')
            adjustment = FeatureAdjustmentSubsystem(load_settings(ROOT / 'configs' / profile))
            for fold in range(1, 6):
                rng = np.random.default_rng(42)
                for speaker, utterances in sorted(by_track[track].items()):
                    available = sorted(utterances)
                    roles = adjustment.fold_roles(available, fold, rng)
                    speaker_counts[(track, fold)].append(Counter(roles.values()))
                    for utterance in available:
                        role = roles[utterance]
                        writer.writerow([track, speaker, utterance, fold, role])
                        counts[(track, fold)][role] += 1
                        if speaker == 1 and utterance <= 5:
                            chosen[(track, fold)][utterance] = role
    return counts, chosen, speaker_counts


def audio_paths(manifest, utterance):
    speaker = manifest['locutores']['1']
    item = manifest['enunciados'][speaker][str(utterance)]
    vctk = Path('/media/lsmsqt/HDD/datasets/vctk/VCTK-Corpus-0.92/wav48_silence_trimmed')
    brsd = Path('/media/lsmsqt/HDD/datasets/brsd/utterances')
    return [
        ('BrSD · locutor 1', brsd / f'{utterance}.wav', FEATURES / 'brsd/1' / str(utterance) / 'mfccs.npy'),
        (f'VCTK · {speaker} · mic1', vctk / speaker / f'{speaker}_{item}_mic1.flac',
         FEATURES / 'vctk8k_mic1/1' / str(utterance) / 'mfccs.npy'),
        (f'VCTK · {speaker} · mic2', vctk / speaker / f'{speaker}_{item}_mic2.flac',
         FEATURES / 'vctk8k_mic2/1' / str(utterance) / 'mfccs.npy'),
    ]


def plot_wave(ax, audio, rate, title, model_seconds):
    stride = max(1, len(audio) // 10000)
    ax.plot(np.arange(0, len(audio), stride) / rate, audio[::stride], lw=.45)
    ax.axvline(model_seconds, ls='--', lw=1, color='crimson')
    ax.set_xlim(0, len(audio) / rate)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Amplitude')


def plot_spectrum(ax, original, filtered, rate):
    for audio, name in ((original, 'original'), (filtered, 'filtrado')):
        frequency, spectrum = welch(audio, fs=rate, nperseg=4096)
        db = 10 * np.log10(np.maximum(spectrum, 1e-12))
        ax.plot(frequency / 1000, db, lw=.5, label=name)
    ax.axvline(3.6, ls='--', color='black', lw=.7)
    ax.set_xlim(0, 6)
    ax.set_title('Anti-aliasing · espectro antes/depois', fontsize=9)
    ax.set_xlabel('Frequência (kHz)')
    ax.set_ylabel('Densidade espectral (dB/Hz)')
    ax.legend(fontsize=7)


def figure_for(manifest, utterance, limits):
    fig, axes = plt.subplots(6, 3, figsize=(16, 17))
    for column, (name, audio_path, feature_path) in enumerate(audio_paths(manifest, utterance)):
        audio, rate = librosa.load(audio_path, sr=None)
        model_seconds = limits['brsd' if column == 0 else 'vctk']
        filtered = dsp.antialias_filter(audio, rate, 8000)
        resampled = dsp.resample(filtered, rate, 8000)
        emphasized = dsp.pre_emphasis(resampled, .97)
        mfcc = np.load(feature_path, allow_pickle=False)
        plot_wave(axes[0, column], audio, rate, 'Original completo', model_seconds)
        axes[0, column].set_title(
            f'{name}\n{audio_path.name} · {len(audio)/rate:.2f} s · {rate/1000:g} kHz',
            fontsize=10)
        plot_spectrum(axes[1, column], audio, filtered, rate)
        plot_wave(axes[2, column], filtered, rate, f'Após filtro · {rate/1000:g} kHz', model_seconds)
        plot_wave(axes[3, column], resampled, 8000, 'Reamostrado · 8 kHz', model_seconds)
        plot_wave(axes[4, column], emphasized, 8000, 'Após pré-ênfase · 8 kHz', model_seconds)
        axes[5, column].imshow(mfcc, origin='lower', aspect='auto', cmap='magma',
                               extent=(0, mfcc.shape[1] * .016, 0, 40))
        axes[5, column].axvline(model_seconds, ls='--', lw=1, color='crimson')
        axes[5, column].set_title(f'MFCC · 40 coeficientes × {mfcc.shape[1]} quadros', fontsize=9)
        axes[5, column].set_xlabel('Tempo aproximado (s)')
        axes[5, column].set_ylabel('Coeficiente')
    fig.suptitle(
        f'Gravação ilustrativa {utterance} · duração inteira; linha vermelha = trecho usado pela rede',
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .98))
    FIGURES.mkdir(parents=True, exist_ok=True)
    target = FIGURES / f'gravacao_{utterance}.png'
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return target


def render(lengths, profiles, results, sizes, counts, chosen, speaker_counts, figures, manifest):
    lines = ['# Reexecução comparativa · BrSD e VCTK a 8 kHz', '',
             'Os dados e as features estão no HDD. O BrSD contém 400 gravações de 80 locutores; '
             'o VCTK usa 21.523 pares de gravações de 108 locutores, em dois microfones. '
             'Neste relatório, uma **gravação** (ou **amostra**) é um arquivo de áudio '
             'e sua matriz de MFCC. “Enunciado” significa o trecho falado nesse arquivo; '
             'não designa uma duração fixa.', '',
             '## Parâmetros de extração', '',
             '| Parâmetro | BrSD | VCTK mic1 | VCTK mic2 |', '|---|---:|---:|---:|']
    labels = {'SOURCE_SAMPLING_RATE': 'Taxa nominal do áudio (Hz)',
              'TARGET_SAMPLING_RATE': 'Taxa final (Hz)', 'ENABLE_VAD': 'VAD',
              'VAD_TOP_DB': 'Limiar VAD (dB; desligado)',
              'PRE_EMPHASIS_COEF': 'Pré-ênfase', 'NUM_MFCCS': 'Coeficientes MFCC',
              'FRAME_SIZE': 'Janela (amostras)'}
    for key in PARAMETERS:
        lines.append('| ' + labels[key] + ' | ' + ' | '.join(profiles[t][key] for t in TRACKS) + ' |')
    lines += ['', 'Cinco WAVs do BrSD (106–110) têm taxa nativa de 44,1 kHz; todos os outros '
              'áudios usados têm 48 kHz. Cada arquivo é filtrado na própria taxa nativa, '
              'antes de ser reamostrado diretamente para 8 kHz.', '',
              'Filtro anti-aliasing: Butterworth de ordem 8, fase zero, corte de 3,6 kHz. '
              'Reamostragem polifásica; salto de 128 amostras (16 ms) para janelas de 256 '
              'amostras (32 ms). O limite do modelo é o menor enunciado do respectivo corpus; '
              'os dois microfones VCTK compartilham o menor valor observado entre eles.', '',
              '## Comprimento das gravações após a extração', '',
              '| Trilha | Gravações | Mínimo (quadros) | Média | Máximo (quadros) | Gravações extremas |',
              '|---|---:|---:|---:|---:|---|']
    for track in (*TRACKS, 'vctk_combinado'):
        item = lengths[track]
        low, high = item['menor'], item['maior']
        lines.append(f"| {track} | {item['gravacoes']} | {low['quadros']} | "
                     f"{item['media_quadros']:.2f} | {high['quadros']} | "
                     f"mín. L{low['locutor']}/E{low['enunciado']}; "
                     f"máx. L{high['locutor']}/E{high['enunciado']} |")
    lines += ['', 'A duração de cada gravação está no arquivo '
              '`runs/features/relatorio_comprimentos_8k.csv`. Um quadro avança 16 ms.', '',
              '## Amostras usadas pelas redes', '',
              'As três redes recebem as mesmas divisões. No BrSD, cada rede usa as cinco '
              'leituras por locutor. No VCTK, cada rede do experimento mic1 usa só mic1; '
              'cada rede do experimento mic2 usa só mic2. Não há mistura de microfones '
              'nestes experimentos.', '',
              '| Trilha | Redes | Partições | Treino por partição | Validação | Teste |',
              '|---|---|---:|---:|---:|---:|']
    for track in TRACKS:
        ranges = {role: sorted({counts[(track, fold)][role] for fold in range(1, 6)})
                  for role in ('treino', 'validacao', 'teste')}
        fmt = lambda values: '–'.join(str(value) for value in values)
        lines.append(f"| {track} | CNN, Temporal CNN, Attention | 5 | "
                     f"{fmt(ranges['treino'])} | {fmt(ranges['validacao'])} | "
                     f"{fmt(ranges['teste'])} |")
    lines += ['', 'Contagens exatas por partição (cada microfone do VCTK tem os mesmos totais):', '',
              '| Corpus | Partição | Treino | Validação | Teste |',
              '|---|---:|---:|---:|---:|']
    for track in ('brsd', 'vctk8k_mic1'):
        for fold in range(1, 6):
            item = counts[(track, fold)]
            label = 'VCTK por microfone' if track != 'brsd' else 'BrSD'
            lines.append(f"| {label} | {fold} | {item['treino']} | "
                         f"{item['validacao']} | {item['teste']} |")
    lines += ['', 'No BrSD, cada locutor tem 5 gravações: em cada partição, 3 vão para '
              'treino, 1 para validação e 1 para teste (60%/20%/20%). O teste gira '
              'de E1 a E5 entre as cinco partições; a validação é sorteada entre as '
              'outras quatro com semente 42. Cada teste contém o mesmo número de '
              'leitura de todos os locutores.', '',
              'No VCTK, 107 locutores têm 200 gravações por microfone: 120 para treino, '
              '40 para validação e 40 para teste em cada partição (60%/20%/20%). '
              'O locutor p362 tem 123 gravações; por isso seus números e os totais variam '
              'ligeiramente. Os índices ordenados são distribuídos em cinco grupos '
              'intercalados: na partição k, o grupo k é teste, o seguinte é validação '
              'e os três restantes são treino. Mic1 e mic2 usam exatamente os mesmos '
              'papéis para cada par de gravações.', '',
              '| Trilha | Treino por locutor (mín.–máx.) | Validação por locutor | Teste por locutor |',
              '|---|---:|---:|---:|']
    for track in TRACKS:
        ranges = []
        for role in ('treino', 'validacao', 'teste'):
            values = [item[role] for fold in range(1, 6) for item in speaker_counts[(track, fold)]]
            ranges.append(f'{min(values)}–{max(values)}')
        lines.append(f'| {track} | ' + ' | '.join(ranges) + ' |')
    lines += ['', 'O arquivo `runs/features/relatorio_particoes_8k.csv` informa o papel '
              'de **cada** gravação em **cada** partição. As tabelas acima mostram '
              'o intervalo de quantidades, não um intervalo contínuo de números de '
              'arquivo: os grupos do VCTK são intercalados.', '',
              'Para os locutores ilustrados, os cinco primeiros índices da partição 1 são:', '',
              '| Trilha | E1 | E2 | E3 | E4 | E5 |', '|---|---|---|---|---|---|']
    for track in TRACKS:
        lines.append('| ' + track + ' | ' + ' | '.join(
            chosen[(track, 1)].get(i, 'ausente') for i in range(1, 6)) + ' |')
    lines += ['', '## Tamanho das redes', '',
              '| Trilha | Rede | Entrada (MFCC × quadros) | Classes | Parâmetros | Modelo médio (MiB) |',
              '|---|---|---:|---:|---:|---:|']
    for track in TRACKS:
        for architecture in ARCHITECTURES:
            frames, parameters, mean_mib = sizes[(track, architecture)]
            lines.append(f'| {track} | {architecture} | 40 × {frames} | '
                         f'{80 if track == "brsd" else 108} | {parameters:,} | '
                         f'{mean_mib:.2f} |'.replace(',', '.'))
    lines += ['', '## Resultados das redes', '',
              '| Trilha | Rede | Acurácia média ± desvio | F1 macro médio | Cinco partições (%) |',
              '|---|---|---:|---:|---|']
    for track in TRACKS:
        for architecture in ARCHITECTURES:
            item = results[(track, architecture)]
            folds = ', '.join(f'{value * 100:.1f}' for value in item['acuracias'])
            lines.append(f"| {track} | {architecture} | "
                         f"{item['acuracia_media'] * 100:.2f} ± "
                         f"{item['acuracia_desvio'] * 100:.2f}% | "
                         f"{item['f1_medio']:.3f} | {folds} |")
    lines += ['', 'Os corpora diferem em número de locutores, duração e conteúdo. '
              'As acurácias são apresentadas lado a lado como resultados de tarefas '
              'fechadas próprias de cada corpus, sem tratá-las como uma medição direta '
              'da mesma população.', '', '## Cadeia de processamento lado a lado', '',
              f"À esquerda está o locutor 1 do BrSD; nas outras colunas, {manifest['locutores']['1']} "
              'do VCTK nos microfones 1 e 2. Os números de locutor dos dois corpora '
              'não identificam a mesma pessoa. Cada número abaixo é apenas um índice '
              'para organizar as figuras: as leituras 1–5 do BrSD são textos diferentes '
              'entre si, e também não correspondem aos textos do VCTK. No VCTK, mic1 '
              'e mic2 são as duas captações da mesma leitura.', '',
              'As formas de onda e os MFCCs mostram cada gravação inteira; a escala de '
              'tempo varia entre as colunas. A linha vermelha tracejada marca o fim do '
              'trecho entregue à rede: os primeiros '
              f"{lengths['brsd']['menor']['quadros']} quadros (aproximadamente "
              f"{lengths['brsd']['menor']['quadros'] * .016:.2f} s) no BrSD e "
              f"{lengths['vctk_combinado']['menor']['quadros']} quadros (aproximadamente "
              f"{lengths['vctk_combinado']['menor']['quadros'] * .016:.2f} s) no VCTK. "
              'A extração guarda o MFCC completo; o corte ocorre na entrada da rede.', '',
              '| Figura | BrSD: arquivo e duração | VCTK mic1: arquivo e duração | VCTK mic2: arquivo e duração |',
              '|---:|---|---|---|']
    for index in range(1, 6):
        entries = []
        for _, path, _ in audio_paths(manifest, index):
            info = sf.info(path)
            entries.append(f'{path.name} · {info.frames / info.samplerate:.2f} s')
        lines.append(f'| {index} | ' + ' | '.join(entries) + ' |')
    lines.append('')
    for index, figure in enumerate(figures, 1):
        lines += [f'### Gravação ilustrativa {index}', '',
                  f'![Etapas da gravação ilustrativa {index}](figuras/relatorio_8k/{figure.name})', '']
    lines += ['## Fontes dos corpora', '',
              '- [BrSD, página dos autores](https://sites.google.com/view/brsduem).',
              '- [VCTK 0.92, University of Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3443).', '']
    DOC.write_text('\n'.join(lines), encoding='utf-8')


def main():
    lengths, profiles, results = read_inputs()
    sizes = count_model_parameters(lengths)
    counts, chosen, speaker_counts = write_splits()
    manifest = json.loads((FEATURES / 'vctk_manifesto.json').read_text())
    limits = {'brsd': lengths['brsd']['menor']['quadros'] * .016,
              'vctk': lengths['vctk_combinado']['menor']['quadros'] * .016}
    figures = [figure_for(manifest, number, limits) for number in range(1, 6)]
    render(lengths, profiles, results, sizes, counts, chosen, speaker_counts, figures, manifest)
    subprocess.run([sys.executable, str(ROOT / 'gerar_pdf.py'), str(DOC)], check=True)
    print(DOC)
    print(DOC.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
