"""Assinatura de canal: o resumo de um recorte de sinal em estatísticas cepstrais.

Este módulo isola o que o diagnóstico de canal precisa saber sobre um sinal, e o
separa de *quando* esse cálculo acontece. A distinção importa por uma razão
operacional: no VCTK o áudio existe apenas transitoriamente — é extraído do zip,
convertido em features e apagado — de modo que qualquer medida que dependa da forma
de onda tem de ser tomada naquela única passada, ou não poderá ser tomada depois
sem decodificar o corpus inteiro de novo.

A assinatura é deliberadamente pobre: média e desvio de cada coeficiente cepstral,
sem qualquer estrutura temporal. Ela descreve a coloração espectral média do recorte
e a sua variabilidade, que é o que caracteriza um canal de gravação. Se a identidade
do locutor puder ser predita a partir disso — ainda mais a partir do recorte **sem
fala** — ela não está sendo lida da voz.
"""

from __future__ import annotations

import numpy as np

#: Recortes de sinal que o diagnóstico sabe extrair.
#:
#: ``silence`` é o controle negativo: se o locutor for identificável daqui, a pista
#: está no equipamento, no ambiente ou no ruído de fundo. ``full`` **contém**
#: ``speech`` e ``silence``, então a comparação informativa é ``full`` contra
#: ``speech``, e não contra ``silence``.
CONDITIONS = ('silence', 'speech', 'full')

#: Número mínimo de amostras para que uma estimativa de assinatura seja estável.
MIN_SAMPLES = 2048


def extract_condition(audio: np.ndarray, condition: str, top_db: int) -> np.ndarray:
    """Extrai do áudio o recorte correspondente à condição pedida.

    Args:
        audio: Sinal completo da gravação.
        condition: Uma de :data:`CONDITIONS`.
        top_db: Limiar da detecção de voz, em dB abaixo do pico.

    Returns:
        O recorte pedido do sinal.

    Raises:
        ValueError: Se a condição for desconhecida.
    """
    from sr.preprocessing import signal

    if condition == 'silence':
        return signal.extract_silence(audio, top_db)
    if condition == 'speech':
        return signal.remove_silence(audio, top_db)
    if condition == 'full':
        return audio
    raise ValueError(f'Condição desconhecida: {condition!r}. Use uma de {CONDITIONS}.')


def channel_signature(
    audio: np.ndarray,
    sampling_rate: int,
    num_mfccs: int,
) -> np.ndarray | None:
    """Resume um recorte de sinal em um vetor de estatísticas cepstrais.

    O cálculo é feito na taxa de amostragem original, e não na taxa decimada da
    cadeia principal: a assinatura de canal mora justamente na parte alta do
    espectro que a decimação descarta.

    Args:
        audio: Recorte de sinal já isolado.
        sampling_rate: Taxa de amostragem do recorte, em Hz.
        num_mfccs: Número de coeficientes cepstrais.

    Returns:
        Vetor de tamanho ``2 * num_mfccs`` com médias seguidas de desvios, ou
        ``None`` se o recorte for curto demais para uma estimativa estável.
    """
    if len(audio) < MIN_SAMPLES:
        return None

    import librosa

    mfccs = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=num_mfccs)
    return np.concatenate([mfccs.mean(axis=1), mfccs.std(axis=1)])


def signatures_of(
    audio: np.ndarray,
    sampling_rate: int,
    num_mfccs: int,
    top_db: int,
    conditions: tuple[str, ...] = CONDITIONS,
) -> dict[str, np.ndarray]:
    """Calcula a assinatura de um sinal sob cada condição, em uma única passada.

    Condições cujo recorte for curto demais são simplesmente omitidas do resultado,
    de modo que o chamador distinga "não havia silêncio suficiente nesta gravação"
    de "o silêncio não identificava o locutor".

    Args:
        audio: Sinal completo da gravação.
        sampling_rate: Taxa de amostragem, em Hz.
        num_mfccs: Número de coeficientes cepstrais.
        top_db: Limiar da detecção de voz, em dB abaixo do pico.
        conditions: Condições a calcular.

    Returns:
        Mapeamento de condição para vetor de assinatura, contendo apenas as
        condições que produziram recorte utilizável.
    """
    result: dict[str, np.ndarray] = {}
    for condition in conditions:
        segment = extract_condition(audio, condition, top_db)
        signature = channel_signature(segment, sampling_rate, num_mfccs)
        if signature is not None:
            result[condition] = signature.astype(np.float32)
    return result
