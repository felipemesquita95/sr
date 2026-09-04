"""Figuras que ilustram a cadeia de processamento de sinal.

São as figuras destinadas ao texto do trabalho — o sinal no tempo, o espectro em
cada estágio da cadeia e a matriz de coeficientes resultante. Ficam separadas do
pipeline porque gerá-las é caro e desnecessário no processamento em massa: o uso
esperado é produzi-las para um punhado de exemplos representativos.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # Backend sem display: o pipeline roda sem sessão gráfica.
import matplotlib.pyplot as plt
import numpy as np


def plot_waveform(audio: np.ndarray, sampling_rate: int, output: Path, title: str) -> None:
    """Desenha o sinal no domínio do tempo.

    Args:
        audio: Sinal a desenhar.
        sampling_rate: Taxa de amostragem, usada para converter índices em segundos.
        output: Caminho do arquivo de imagem a gravar.
        title: Título da figura.
    """
    time = np.arange(len(audio)) / sampling_rate

    plt.figure(figsize=(10, 4))
    plt.plot(time, audio, color='steelblue', linewidth=0.5)
    plt.title(title)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.xlim(0, time[-1] if len(time) else 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_spectrum(audio: np.ndarray, sampling_rate: int, output: Path, title: str) -> None:
    """Desenha o espectro de magnitude do sinal.

    Mostra apenas as frequências não negativas: para um sinal real, o espectro é
    simétrico e a metade negativa não acrescenta informação.

    Args:
        audio: Sinal a analisar.
        sampling_rate: Taxa de amostragem, em Hz.
        output: Caminho do arquivo de imagem a gravar.
        title: Título da figura.
    """
    spectrum = np.abs(np.fft.rfft(audio))
    frequency = np.fft.rfftfreq(len(audio), d=1 / sampling_rate)

    plt.figure(figsize=(10, 4))
    plt.plot(frequency / 1000, spectrum, color='steelblue', linewidth=0.6)
    plt.title(title)
    plt.xlabel('Frequência (kHz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, frequency[-1] / 1000 if len(frequency) else 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_mfccs(mfccs: np.ndarray, output: Path, title: str) -> None:
    """Desenha a matriz de coeficientes cepstrais como mapa de calor.

    Args:
        mfccs: Matriz de forma ``(num_coeficientes, num_quadros)``.
        output: Caminho do arquivo de imagem a gravar.
        title: Título da figura.
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(mfccs, aspect='auto', origin='lower', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Magnitude')
    plt.title(title)
    plt.xlabel('Quadro')
    plt.ylabel('Coeficiente cepstral')
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def plot_duration_distribution(frame_counts: list[int], output: Path) -> None:
    """Desenha o histograma de duração dos enunciados, medida em quadros.

    A dispersão desta distribuição determina o custo do alinhamento por padding: se
    os enunciados variam muito de duração, o comprimento comum acaba dominado pelos
    mais longos e os curtos ficam majoritariamente preenchidos por repetição.

    Args:
        frame_counts: Número de quadros de cada enunciado do conjunto.
        output: Caminho do arquivo de imagem a gravar.
    """
    if not frame_counts:
        return

    counts = np.asarray(frame_counts)

    plt.figure(figsize=(10, 5))
    plt.hist(counts, bins=40, color='steelblue', edgecolor='white')
    plt.axvline(counts.mean(), color='crimson', linestyle='--',
                label=f'média {counts.mean():.0f} quadros')
    plt.axvline(np.percentile(counts, 95), color='darkorange', linestyle=':',
                label=f'p95 {np.percentile(counts, 95):.0f} quadros')
    plt.title('Distribuição de duração dos enunciados')
    plt.xlabel('Número de quadros')
    plt.ylabel('Número de enunciados')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def plot_utterances_per_speaker(counts: dict[int, int], output: Path) -> None:
    """Desenha o número de enunciados disponível por locutor.

    Desequilíbrio entre classes enviesa a acurácia global; a figura documenta se o
    conjunto está balanceado.

    Args:
        counts: Mapeamento de índice de locutor para contagem de enunciados.
        output: Caminho do arquivo de imagem a gravar.
    """
    if not counts:
        return

    speakers = sorted(counts)
    values = [counts[s] for s in speakers]

    plt.figure(figsize=(max(10, len(speakers) * 0.12), 5))
    plt.bar(speakers, values, color='darkorange')
    plt.title(f'Enunciados por locutor ({sum(values)} no total, {len(speakers)} locutores)')
    plt.xlabel('Locutor')
    plt.ylabel('Número de enunciados')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
