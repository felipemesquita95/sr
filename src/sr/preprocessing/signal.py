"""Operações de processamento de sinal, sem efeitos colaterais.

Todas as funções deste módulo são puras: recebem e devolvem arrays, sem tocar em
disco nem depender de configuração global. Isso as torna verificáveis isoladamente
e permite descrever o pipeline no texto do trabalho como uma composição de
transformações bem definidas.

A ordem canônica da cadeia é:

    carregar → (VAD) → filtro anti-aliasing → decimação → pré-ênfase → MFCC

A pré-ênfase vem **depois** da decimação porque é um filtro de primeira ordem
definido em relação à taxa de amostragem final; aplicá-la antes daria a ela um
ponto de corte diferente do pretendido.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

#: Ordem do filtro Butterworth anti-aliasing.
ANTIALIAS_ORDER = 8

#: Fração da taxa de destino usada como corte (90% da nova frequência de Nyquist).
ANTIALIAS_CUTOFF_RATIO = 0.45


def remove_silence(audio: np.ndarray, top_db: int) -> np.ndarray:
    """Concatena apenas os trechos detectados como voz.

    Args:
        audio: Sinal de entrada.
        top_db: Limiar em dB abaixo do pico; trechos mais baixos são descartados.

    Returns:
        Sinal contendo somente os intervalos com atividade vocal. Se nenhum trecho
        for detectado, devolve o sinal original inalterado.
    """
    import librosa  # importado tardiamente: librosa é pesado e nem todo uso o exige.

    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return audio
    return np.concatenate([audio[start:end] for start, end in intervals])


def extract_silence(audio: np.ndarray, top_db: int) -> np.ndarray:
    """Concatena apenas os trechos **não** detectados como voz.

    É o complemento exato de :func:`remove_silence`, e a base do diagnóstico de
    canal: se a identidade do locutor puder ser predita a partir deste sinal, ela
    não está sendo lida da fala.

    Args:
        audio: Sinal de entrada.
        top_db: Limiar em dB abaixo do pico, idêntico ao usado na detecção de voz.

    Returns:
        Sinal contendo somente os intervalos sem atividade vocal.
    """
    import librosa

    voiced = librosa.effects.split(audio, top_db=top_db)
    if len(voiced) == 0:
        return np.array([], dtype=audio.dtype)

    mask = np.ones(len(audio), dtype=bool)
    for start, end in voiced:
        mask[start:end] = False
    return audio[mask]


def antialias_filter(audio: np.ndarray, sampling_rate: int, target_rate: int) -> np.ndarray:
    """Aplica o filtro passa-baixas que precede a decimação.

    O corte fica em 45% da nova frequência de Nyquist, deixando banda de guarda para
    o rolloff do filtro. Sem esta etapa, componentes acima de ``target_rate / 2``
    seriam rebatidos para dentro da banda útil pela decimação, contaminando os
    coeficientes cepstrais com energia espúria.

    Usa-se filtragem de fase zero (``sosfiltfilt``): a filtragem direta introduziria
    atraso de fase dependente da frequência, deslocando os formantes uns em relação
    aos outros ao longo do tempo.

    Args:
        audio: Sinal na taxa original.
        sampling_rate: Taxa de amostragem do sinal de entrada, em Hz.
        target_rate: Taxa de amostragem pretendida após a decimação, em Hz.

    Returns:
        Sinal filtrado, ainda na taxa original.
    """
    if target_rate >= sampling_rate:
        return audio

    cutoff = ANTIALIAS_CUTOFF_RATIO * target_rate
    sos = butter(ANTIALIAS_ORDER, cutoff, btype='low', fs=sampling_rate, output='sos')
    return sosfiltfilt(sos, audio)


def resample(audio: np.ndarray, sampling_rate: int, target_rate: int) -> np.ndarray:
    """Reamostra o sinal para a taxa alvo por reamostragem polifásica.

    Prefere-se ``resample_poly`` ao método baseado em FFT: as taxas envolvidas têm
    razão racional exata (48 kHz para 8 kHz é 1/6; para 16 kHz é 1/3), e o método
    por FFT pressupõe periodicidade do sinal, produzindo artefatos nas bordas de
    gravações que não começam e terminam no mesmo valor.

    Args:
        audio: Sinal de entrada.
        sampling_rate: Taxa de amostragem atual, em Hz.
        target_rate: Taxa de amostragem desejada, em Hz.

    Returns:
        Sinal reamostrado.
    """
    if target_rate == sampling_rate:
        return audio

    from math import gcd

    divisor = gcd(int(target_rate), int(sampling_rate))
    return resample_poly(audio, int(target_rate) // divisor, int(sampling_rate) // divisor)


def pre_emphasis(audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """Aplica o filtro de pré-ênfase de primeira ordem ``y[n] = x[n] - a·x[n-1]``.

    Compensa a queda de aproximadamente 6 dB por oitava do espectro da fala,
    equilibrando a energia entre as regiões grave e aguda antes da análise cepstral.

    Args:
        audio: Sinal de entrada.
        coef: Coeficiente ``a``. O valor ``0.0`` desativa a pré-ênfase, permitindo
            executar a ablação sem alterar o restante da cadeia.

    Returns:
        Sinal pré-enfatizado, com o mesmo comprimento da entrada.
    """
    if coef == 0.0:
        return audio
    return np.append(audio[0], audio[1:] - coef * audio[:-1])


def extract_mfccs(
    audio: np.ndarray,
    sampling_rate: int,
    num_mfccs: int,
    frame_size: int,
) -> np.ndarray:
    """Extrai a matriz de coeficientes cepstrais de frequência mel.

    O salto entre quadros é metade da janela, ou seja, 50% de sobreposição — valor
    convencional, que evita perder transições que caiam na fronteira entre quadros.

    Args:
        audio: Sinal já pré-processado.
        sampling_rate: Taxa de amostragem do sinal, em Hz.
        num_mfccs: Número de coeficientes por quadro.
        frame_size: Tamanho da janela de análise, em amostras.

    Returns:
        Matriz de forma ``(num_mfccs, num_quadros)``.

    Note:
        Valores altos de ``num_mfccs`` retêm estrutura espectral fina — harmônicos e
        coloração do canal — além do envelope do trato vocal. Isso é relevante para a
        interpretação dos resultados: coeficientes de ordem alta carregam informação
        do equipamento de gravação, não apenas do locutor.
    """
    import librosa

    return librosa.feature.mfcc(
        y=audio,
        sr=sampling_rate,
        n_mfcc=num_mfccs,
        n_fft=frame_size,
        hop_length=frame_size // 2,
    )
