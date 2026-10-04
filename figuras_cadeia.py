#!/usr/bin/env python3
"""Figuras de cada estágio da cadeia de pré-processamento, com unidades declaradas.

As figuras saem de uma gravação real percorrendo as **mesmas funções** que o
pipeline usa (``sr.preprocessing.signal``), e não de uma reimplementação: o que a
apresentação mostra é o que o experimento fez.

O perfil ilustrado é ``configs/brsd_vad.env`` — 48 kHz de origem, 16 kHz de destino,
VAD ligado, janela de 512 amostras. É o único perfil cujo áudio de origem ainda está
em disco e que exercita **todos** os estágios, inclusive o VAD, e a sua cadeia de
pré-processamento é idêntica à do perfil de referência do VCTK. O perfil de referência
do BrSD difere apenas em taxa de destino (8 kHz) e janela (256 amostras).

Uso::

    .venv/bin/python figuras_cadeia.py
    .venv/bin/python figuras_cadeia.py --gravacao 2 --saida docs/figuras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, freqz, sosfreqz

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from sr.preprocessing import signal as dsp  # noqa: E402

#: Parâmetros do perfil ilustrado, espelhando ``configs/brsd_vad.env``.
TAXA_ORIGEM = 48_000
TAXA_DESTINO = 16_000
VAD_TOP_DB = 30
COEF_PREENFASE = 0.97
JANELA = 512
NUM_MFCCS = 40

#: Trecho ampliado, em segundos, para as figuras que precisam de detalhe temporal.
RECORTE = (20.0, 20.12)

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'font.size': 9,
    'axes.grid': True, 'grid.alpha': 0.3, 'axes.titlesize': 10,
})

AZUL, LARANJA, VERMELHO = 'steelblue', 'darkorange', 'crimson'


def espectro_db(x: np.ndarray, taxa: int) -> tuple[np.ndarray, np.ndarray]:
    """Espectro de magnitude em dB relativos ao máximo do próprio sinal.

    A referência é o máximo porque o interesse aqui é a **forma** do espectro e a
    profundidade da atenuação, não o nível absoluto — que depende do ganho de
    gravação e não é comparável entre estágios.

    Args:
        x: Sinal a analisar.
        taxa: Taxa de amostragem, em Hz.

    Returns:
        Par ``(frequências em kHz, magnitude em dB rel. ao máximo)``.
    """
    magnitude = np.abs(np.fft.rfft(x))
    frequencia = np.fft.rfftfreq(len(x), d=1 / taxa)
    referencia = magnitude.max() if magnitude.max() > 0 else 1.0
    return frequencia / 1000, 20 * np.log10(np.maximum(magnitude / referencia, 1e-12))


def figura_carregamento(audio: np.ndarray, saida: Path) -> None:
    """Sinal como sai do disco: domínio do tempo e espectro, a 48 kHz."""
    figura, eixos = plt.subplots(2, 1, figsize=(9, 4.6))
    tempo = np.arange(len(audio)) / TAXA_ORIGEM

    eixos[0].plot(tempo, audio, color=AZUL, linewidth=0.3)
    eixos[0].set_title(f'Sinal carregado — {TAXA_ORIGEM/1000:.0f} kHz, mono, '
                       f'{len(audio):,} amostras, {tempo[-1]:.1f} s'.replace(',', '.'))
    eixos[0].set_xlabel('Tempo (s)')
    eixos[0].set_ylabel('Amplitude\n(escala cheia, adimensional)')
    eixos[0].set_xlim(0, tempo[-1])

    frequencia, db = espectro_db(audio, TAXA_ORIGEM)
    eixos[1].plot(frequencia, db, color=AZUL, linewidth=0.4)
    eixos[1].set_title('Espectro de magnitude do sinal carregado')
    eixos[1].set_xlabel('Frequência (kHz)')
    eixos[1].set_ylabel('Magnitude\n(dB rel. ao máximo)')
    eixos[1].set_xlim(0, TAXA_ORIGEM / 2000)
    eixos[1].set_ylim(-120, 5)

    figura.tight_layout()
    figura.savefig(saida)
    plt.close(figura)


def figura_vad(audio: np.ndarray, saida: Path) -> None:
    """Detecção de atividade vocal: o que é mantido, o que é descartado."""
    import librosa

    intervalos = librosa.effects.split(audio, top_db=VAD_TOP_DB)
    voz = dsp.remove_silence(audio, VAD_TOP_DB)
    silencio = dsp.extract_silence(audio, VAD_TOP_DB)
    retido = len(voz) / len(audio)

    figura, eixos = plt.subplots(2, 1, figsize=(9, 4.6))
    tempo = np.arange(len(audio)) / TAXA_ORIGEM

    eixos[0].plot(tempo, audio, color=AZUL, linewidth=0.3)
    for inicio, fim in intervalos:
        eixos[0].axvspan(inicio / TAXA_ORIGEM, fim / TAXA_ORIGEM,
                         color=LARANJA, alpha=0.25, linewidth=0)
    eixos[0].set_title(f'VAD — {len(intervalos)} intervalos com voz (faixas laranja), '
                       f'limiar de {VAD_TOP_DB} dB abaixo do pico')
    eixos[0].set_xlabel('Tempo (s)')
    eixos[0].set_ylabel('Amplitude\n(escala cheia)')
    eixos[0].set_xlim(0, tempo[-1])

    tempo_voz = np.arange(len(voz)) / TAXA_ORIGEM
    eixos[1].plot(tempo_voz, voz, color=LARANJA, linewidth=0.3)
    eixos[1].set_title(f'Sinal concatenado após o VAD — {retido:.1%} das amostras '
                       f'retidas; {len(silencio)/TAXA_ORIGEM:.1f} s descartados')
    eixos[1].set_xlabel('Tempo (s)')
    eixos[1].set_ylabel('Amplitude\n(escala cheia)')
    eixos[1].set_xlim(0, tempo[-1])

    figura.tight_layout()
    figura.savefig(saida)
    plt.close(figura)


def figura_antialias(audio: np.ndarray, filtrado: np.ndarray, saida: Path) -> None:
    """Resposta do Butterworth e o efeito dele sobre o espectro.

    Mostra as duas magnitudes que importam: a de uma passagem do filtro e a efetiva
    de ``sosfiltfilt``, que filtra para frente e para trás e portanto eleva a
    resposta ao quadrado — o dobro da atenuação em dB, e fase exatamente nula.
    """
    corte = dsp.ANTIALIAS_CUTOFF_RATIO * TAXA_DESTINO
    sos = butter(dsp.ANTIALIAS_ORDER, corte, btype='low', fs=TAXA_ORIGEM, output='sos')
    frequencia, resposta = sosfreqz(sos, worN=8192, fs=TAXA_ORIGEM)
    uma = 20 * np.log10(np.maximum(np.abs(resposta), 1e-12))

    figura, eixos = plt.subplots(2, 1, figsize=(9, 4.8))

    eixos[0].plot(frequencia / 1000, uma, color=AZUL,
                  label=f'uma passagem (ordem {dsp.ANTIALIAS_ORDER})')
    eixos[0].plot(frequencia / 1000, 2 * uma, color=LARANJA,
                  label='efetiva de sosfiltfilt (ida e volta)')
    eixos[0].axvline(corte / 1000, color=VERMELHO, linestyle='--',
                     label=f'corte {corte/1000:.1f} kHz (0,45 × {TAXA_DESTINO/2000:.0f} kHz)')
    eixos[0].axvline(TAXA_DESTINO / 2000, color='black', linestyle=':',
                     label=f'Nyquist do destino {TAXA_DESTINO/2000:.0f} kHz')
    eixos[0].set_title('Resposta em magnitude do filtro anti-aliasing Butterworth')
    eixos[0].set_xlabel('Frequência (kHz)')
    eixos[0].set_ylabel('Ganho |H(f)| (dB)')
    eixos[0].set_xlim(0, 12)
    eixos[0].set_ylim(-160, 6)
    eixos[0].legend(fontsize=7, loc='lower left')

    for sinal, cor, rotulo in ((audio, AZUL, 'antes'), (filtrado, LARANJA, 'depois')):
        f, db = espectro_db(sinal, TAXA_ORIGEM)
        eixos[1].plot(f, db, color=cor, linewidth=0.4, label=rotulo, alpha=0.85)
    eixos[1].axvline(TAXA_DESTINO / 2000, color='black', linestyle=':',
                     label=f'Nyquist do destino {TAXA_DESTINO/2000:.0f} kHz')
    eixos[1].set_title('Espectro do sinal antes e depois da filtragem')
    eixos[1].set_xlabel('Frequência (kHz)')
    eixos[1].set_ylabel('Magnitude\n(dB rel. ao máximo)')
    eixos[1].set_xlim(0, TAXA_ORIGEM / 2000)
    eixos[1].set_ylim(-140, 5)
    eixos[1].legend(fontsize=7, loc='upper right')

    figura.tight_layout()
    figura.savefig(saida)
    plt.close(figura)


def figura_decimacao(filtrado: np.ndarray, reamostrado: np.ndarray, saida: Path) -> None:
    """Decimação polifásica: a banda que sobra e a que foi descartada."""
    fator = TAXA_ORIGEM // TAXA_DESTINO

    figura, eixos = plt.subplots(2, 1, figsize=(9, 4.6))

    f_antes, db_antes = espectro_db(filtrado, TAXA_ORIGEM)
    f_depois, db_depois = espectro_db(reamostrado, TAXA_DESTINO)
    eixos[0].plot(f_antes, db_antes, color=AZUL, linewidth=0.4,
                  label=f'filtrado, {TAXA_ORIGEM/1000:.0f} kHz', alpha=0.8)
    eixos[0].plot(f_depois, db_depois, color=LARANJA, linewidth=0.4,
                  label=f'decimado, {TAXA_DESTINO/1000:.0f} kHz (fator 1/{fator})')
    eixos[0].axvline(TAXA_DESTINO / 2000, color='black', linestyle=':',
                     label=f'Nyquist {TAXA_DESTINO/2000:.0f} kHz')
    eixos[0].set_title('Espectro antes e depois da decimação polifásica (resample_poly)')
    eixos[0].set_xlabel('Frequência (kHz)')
    eixos[0].set_ylabel('Magnitude\n(dB rel. ao máximo)')
    eixos[0].set_xlim(0, TAXA_ORIGEM / 2000)
    eixos[0].set_ylim(-140, 5)
    eixos[0].legend(fontsize=7, loc='upper right')

    inicio, fim = RECORTE[0], RECORTE[0] + 0.008
    a = slice(int(inicio * TAXA_ORIGEM), int(fim * TAXA_ORIGEM))
    b = slice(int(inicio * TAXA_DESTINO), int(fim * TAXA_DESTINO))
    eixos[1].plot(1000 * (np.arange(a.start, a.stop) / TAXA_ORIGEM - inicio), filtrado[a],
                  color=AZUL, linewidth=0.9, marker='.', markersize=4,
                  label=f'{TAXA_ORIGEM/1000:.0f} kHz  ({a.stop - a.start} amostras)',
                  zorder=2)
    eixos[1].plot(1000 * (np.arange(b.start, b.stop) / TAXA_DESTINO - inicio), reamostrado[b],
                  color=LARANJA, linestyle='none', marker='o', markersize=7,
                  markerfacecolor='none', markeredgewidth=1.3,
                  label=f'{TAXA_DESTINO/1000:.0f} kHz  ({b.stop - b.start} amostras)',
                  zorder=3)
    eixos[1].set_title(f'Mesmo trecho de {(fim-inicio)*1000:.0f} ms nas duas taxas: '
                       f'uma amostra a cada {fator} sobrevive à decimação')
    eixos[1].set_xlabel(f'Tempo desde {inicio:.3f} s da gravação (ms)')
    eixos[1].set_ylabel('Amplitude\n(escala cheia)')
    eixos[1].set_xlim(0, 1000 * (fim - inicio))
    eixos[1].legend(fontsize=7)

    figura.tight_layout()
    figura.savefig(saida)
    plt.close(figura)


def figura_preenfase(reamostrado: np.ndarray, enfatizado: np.ndarray, saida: Path) -> None:
    """Pré-ênfase: filtro FIR de primeira ordem e o seu efeito na inclinação."""
    frequencia, resposta = freqz([1.0, -COEF_PREENFASE], [1.0], worN=4096, fs=TAXA_DESTINO)
    ganho = 20 * np.log10(np.maximum(np.abs(resposta), 1e-12))

    figura, eixos = plt.subplots(2, 1, figsize=(9, 4.6))

    eixos[0].plot(frequencia / 1000, ganho, color=AZUL)
    eixos[0].axhline(0, color='black', linewidth=0.6, linestyle=':')
    eixos[0].set_title(f'Resposta da pré-ênfase  H(z) = 1 − {COEF_PREENFASE} z⁻¹  '
                       '— passa-altas de primeira ordem, ≈ +6 dB por oitava')
    eixos[0].set_xlabel('Frequência (kHz)')
    eixos[0].set_ylabel('Ganho |H(f)| (dB)')
    eixos[0].set_xlim(0, TAXA_DESTINO / 2000)

    for sinal, cor, rotulo in ((reamostrado, AZUL, 'antes'), (enfatizado, LARANJA, 'depois')):
        f, db = espectro_db(sinal, TAXA_DESTINO)
        eixos[1].plot(f, db, color=cor, linewidth=0.4, label=rotulo, alpha=0.85)
    eixos[1].set_title('Espectro antes e depois da pré-ênfase: a região aguda sobe '
                       'em relação à grave')
    eixos[1].set_xlabel('Frequência (kHz)')
    eixos[1].set_ylabel('Magnitude\n(dB rel. ao máximo)')
    eixos[1].set_xlim(0, TAXA_DESTINO / 2000)
    eixos[1].set_ylim(-100, 5)
    eixos[1].legend(fontsize=7, loc='upper right')

    figura.tight_layout()
    figura.savefig(saida)
    plt.close(figura)


def figura_janelamento(enfatizado: np.ndarray, saida: Path) -> None:
    """Janela de Hann, sobreposição de 50% e o banco de filtros mel."""
    import librosa

    salto = JANELA // 2
    janela = np.hanning(JANELA)
    duracao_ms = 1000 * JANELA / TAXA_DESTINO

    figura, eixos = plt.subplots(2, 1, figsize=(9, 4.8))

    inicio = int(RECORTE[0] * TAXA_DESTINO)
    amostras = np.arange(JANELA * 2)
    eixos[0].plot(1000 * amostras / TAXA_DESTINO,
                  enfatizado[inicio:inicio + JANELA * 2],
                  color='0.6', linewidth=0.6, label='sinal pré-enfatizado')
    for k, cor in enumerate((AZUL, LARANJA, VERMELHO)):
        deslocamento = k * salto
        eixos[0].plot(1000 * (np.arange(JANELA) + deslocamento) / TAXA_DESTINO,
                      janela * np.abs(enfatizado[inicio:inicio + JANELA * 2]).max(),
                      color=cor, linewidth=1.2,
                      label=f'quadro {k + 1}' if k < 3 else None)
    eixos[0].set_title(f'Janelamento de Hann — {JANELA} amostras = {duracao_ms:.0f} ms, '
                       f'salto de {salto} amostras = {duracao_ms/2:.0f} ms (50% de sobreposição)')
    eixos[0].set_xlabel('Tempo dentro do trecho (ms)')
    eixos[0].set_ylabel('Amplitude\n(escala cheia)')
    eixos[0].set_xlim(0, 1000 * JANELA * 2 / TAXA_DESTINO)
    eixos[0].legend(fontsize=7, ncol=4, loc='upper right')

    banco = librosa.filters.mel(sr=TAXA_DESTINO, n_fft=JANELA)
    frequencias = np.fft.rfftfreq(JANELA, d=1 / TAXA_DESTINO) / 1000
    for i in range(0, banco.shape[0], 4):
        eixos[1].plot(frequencias, banco[i], linewidth=0.7)
    eixos[1].set_title(f'Banco de filtros mel — {banco.shape[0]} filtros triangulares '
                       'sobre a escala Slaney (1 em cada 4 desenhado)')
    eixos[1].set_xlabel('Frequência (kHz)')
    eixos[1].set_ylabel('Ganho do filtro\n(normalizado por área, 1/Hz)')
    eixos[1].set_xlim(0, TAXA_DESTINO / 2000)

    figura.tight_layout()
    figura.savefig(saida)
    plt.close(figura)


def figura_mfcc(enfatizado: np.ndarray, saida: Path) -> None:
    """Matriz de coeficientes cepstrais, com o eixo do tempo em segundos.

    O coeficiente 0 vai em um painel próprio. Ele é a energia do quadro e tem ordem
    de grandeza muito maior que os demais; deixá-lo no mesmo mapa de cores comprime
    todo o resto em uma faixa indistinguível.
    """
    mfccs = dsp.extract_mfccs(enfatizado, TAXA_DESTINO, NUM_MFCCS, JANELA)
    salto = JANELA // 2
    duracao = mfccs.shape[1] * salto / TAXA_DESTINO

    figura, eixos = plt.subplots(2, 1, figsize=(9, 4.4), height_ratios=(1, 2.6))

    tempo = np.arange(mfccs.shape[1]) * salto / TAXA_DESTINO
    eixos[0].plot(tempo, mfccs[0], color=AZUL, linewidth=0.4)
    eixos[0].set_title('Coeficiente 0 — a energia do quadro, em escala própria')
    eixos[0].set_xlabel('Tempo (s)')
    eixos[0].set_ylabel('c₀ (dB · DCT)')
    eixos[0].set_xlim(0, duracao)

    restante = mfccs[1:]
    limite = np.percentile(np.abs(restante), 99)
    imagem = eixos[1].imshow(restante, aspect='auto', origin='lower', cmap='viridis',
                             interpolation='nearest', vmin=-limite, vmax=limite,
                             extent=(0, duracao, 0.5, NUM_MFCCS - 0.5))
    barra = figura.colorbar(imagem, ax=eixos[1], pad=0.01)
    barra.set_label('Amplitude cepstral\n(dB transformados pela DCT-II ortonormal)')
    eixos[1].set_title(f'Coeficientes 1 a {NUM_MFCCS - 1} — matriz completa é '
                       f'{mfccs.shape[0]} × {mfccs.shape[1]} (coeficientes × quadros), '
                       f'{duracao:.1f} s; cores saturadas no percentil 99')
    eixos[1].set_xlabel('Tempo (s)')
    eixos[1].set_ylabel('Índice do coeficiente\n(adimensional)')
    eixos[1].grid(False)

    figura.tight_layout()
    figura.savefig(saida)
    plt.close(figura)


def main() -> None:
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument('--gravacao', type=int, default=41,
                            help='índice do arquivo do BrSD a ilustrar')
    analisador.add_argument('--saida', type=Path, default=ROOT / 'docs' / 'figuras')
    analisador.add_argument('--audio', type=Path,
                            default=Path.home() / 'datasets' / 'brsd' / 'utterances')
    argumentos = analisador.parse_args()

    import librosa

    origem = argumentos.audio / f'{argumentos.gravacao}.wav'
    if not origem.is_file():
        sys.exit(f'Gravação não encontrada: {origem}')

    argumentos.saida.mkdir(parents=True, exist_ok=True)

    # A cadeia, exatamente na ordem do pipeline.
    audio, _ = librosa.load(origem, sr=TAXA_ORIGEM)
    voz = dsp.remove_silence(audio, VAD_TOP_DB)
    filtrado = dsp.antialias_filter(voz, TAXA_ORIGEM, TAXA_DESTINO)
    reamostrado = dsp.resample(filtrado, TAXA_ORIGEM, TAXA_DESTINO)
    enfatizado = dsp.pre_emphasis(reamostrado, COEF_PREENFASE)

    figuras = (
        ('01_carregamento.png', lambda p: figura_carregamento(audio, p)),
        ('02_vad.png', lambda p: figura_vad(audio, p)),
        ('03_antialias.png', lambda p: figura_antialias(voz, filtrado, p)),
        ('04_decimacao.png', lambda p: figura_decimacao(filtrado, reamostrado, p)),
        ('05_preenfase.png', lambda p: figura_preenfase(reamostrado, enfatizado, p)),
        ('06_janelamento.png', lambda p: figura_janelamento(enfatizado, p)),
        ('07_mfcc.png', lambda p: figura_mfcc(enfatizado, p)),
    )
    for nome, desenhar in figuras:
        destino = argumentos.saida / nome
        desenhar(destino)
        print(f'{destino}  ({destino.stat().st_size / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
