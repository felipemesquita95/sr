"""Indexação dos corpora, mapeando arquivos de áudio para (locutor, enunciado).

Este módulo é a única fonte de verdade sobre *quais* gravações compõem um
experimento e *como* elas são numeradas. Tanto o pré-processamento quanto os
instrumentos de diagnóstico consomem o mesmo índice — se cada um construísse a
própria lista, uma divergência silenciosa entre eles invalidaria qualquer
comparação entre os seus resultados.

Ambos os corpora são renumerados para inteiros começando em 1, de modo que o
restante do sistema não precisa conhecer as convenções de nomes de cada um.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sr.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Recording:
    """Uma gravação do corpus, já renumerada.

    Attributes:
        speaker: Índice do locutor, de 1 a ``num_speakers``.
        utterance: Índice do enunciado dentro do locutor, começando em 1.
        path: Caminho do arquivo de áudio.
    """

    speaker: int
    utterance: int
    path: Path

    @property
    def label(self) -> int:
        """Rótulo de classificação, base zero, como esperado pelo Keras."""
        return self.speaker - 1


def build_index(settings: Settings) -> list[Recording]:
    """Constrói o índice de gravações do corpus configurado.

    Args:
        settings: Configuração do experimento; determina o corpus, o número de
            locutores e o número máximo de enunciados por locutor.

    Returns:
        Lista de gravações ordenada por locutor e depois por enunciado.

    Raises:
        FileNotFoundError: Se o diretório do corpus não existir.
        ValueError: Se o formato do corpus for desconhecido.
    """
    if settings.dataset_format == 'vctk':
        recordings = _index_vctk(settings)
    elif settings.dataset_format == 'brsd':
        recordings = _index_brsd(settings)
    else:
        raise ValueError(f'Formato de corpus desconhecido: {settings.dataset_format!r}')

    speakers = {r.speaker for r in recordings}
    logger.info(
        'Índice do corpus %s: %d gravações de %d locutores.',
        settings.dataset_format, len(recordings), len(speakers),
    )
    return recordings


def _index_brsd(settings: Settings) -> list[Recording]:
    """Indexa o BrSD, cujos arquivos são ``1.wav`` … ``400.wav`` numerados em sequência.

    A numeração é densa e sequencial por locutor: o enunciado ``u`` do locutor ``s``
    é o arquivo de índice ``(s - 1) * num_utterances + u``. Como todos os locutores
    leram os mesmos cinco textos, o índice do enunciado identifica **o texto**, e não
    uma sessão independente de gravação — um fato relevante para a interpretação dos
    resultados (ver ``docs/limitacoes.md``).
    """
    if not settings.audio_path.is_dir():
        raise FileNotFoundError(f'Diretório de áudios não encontrado: {settings.audio_path}')

    recordings: list[Recording] = []
    missing = 0
    for speaker in range(1, settings.num_speakers + 1):
        for utterance in range(1, settings.num_utterances + 1):
            index = (speaker - 1) * settings.num_utterances + utterance
            path = settings.audio_path / f'{index}.wav'
            if path.exists():
                recordings.append(Recording(speaker, utterance, path))
            else:
                missing += 1

    if missing:
        logger.warning('BrSD: %d arquivos esperados não foram encontrados.', missing)
    return recordings


def _index_vctk(settings: Settings) -> list[Recording]:
    """Indexa o VCTK, organizado em uma pasta por locutor com FLACs por microfone.

    Cada frase foi gravada simultaneamente por dois microfones (``mic1``, omnidirecional;
    ``mic2``, condensador), produzindo arquivos ``pNNN_XXX_micK.flac``. Selecionar uma
    trilha por vez é o que viabiliza o protocolo cross-mic: mesma voz, mesmo texto,
    mesmo instante, apenas o transdutor muda.

    A numeração é definida sobre a **interseção** das trilhas declaradas em
    ``vctk_mics``, e não sobre a trilha que está sendo processada no momento.

    A distinção é a diferença entre o protocolo medir o que afirma e não medir. Se
    cada trilha fosse numerada isoladamente, bastaria um locutor sem ``mic2`` para
    que todos os locutores seguintes deslocassem de um entre as duas numerações — e
    ``p280`` e ``p315`` são exatamente esse caso no VCTK 0.92, nas posições 54 e 84.
    O protocolo cross-microfone passaria a comparar pessoas diferentes, sem erro e
    sem aviso, apenas devolvendo uma acurácia mais baixa que se leria como achado.

    O mesmo vale um nível abaixo, para os enunciados: só entram os presentes em todas
    as trilhas declaradas, para que o índice de frase também designe a mesma frase.
    """
    if settings.vctk_root is None or not settings.vctk_root.is_dir():
        raise FileNotFoundError(f'Raiz do VCTK não encontrada: {settings.vctk_root}')

    tracks = settings.vctk_mics or (settings.vctk_mic,)
    if settings.vctk_mic not in tracks:
        raise ValueError(
            f'A trilha corrente {settings.vctk_mic!r} não está em vctk_mics={tracks}. '
            f'A numeração seria definida sobre trilhas que não incluem a processada.')

    def utterances_of(directory: Path, mic: str) -> set[str]:
        """Identificadores de enunciado que a trilha ``mic`` possui neste locutor."""
        suffix = f'_{mic}.flac'
        return {f.name[:-len(suffix)].split('_')[-1]
                for f in directory.iterdir() if f.name.endswith(suffix)}

    directories = sorted(d for d in settings.vctk_root.iterdir() if d.is_dir())
    complete: list[tuple[Path, list[str]]] = []
    incomplete: list[str] = []

    for directory in directories:
        shared: set[str] | None = None
        for mic in tracks:
            present = utterances_of(directory, mic)
            shared = present if shared is None else (shared & present)
        if shared:
            complete.append((directory, sorted(shared)))
        else:
            incomplete.append(directory.name)

    if incomplete:
        logger.warning(
            'VCTK: %d locutores sem todas as trilhas %s foram omitidos da numeração: %s',
            len(incomplete), ','.join(tracks), ', '.join(incomplete))

    recordings: list[Recording] = []
    for speaker, (directory, shared) in enumerate(complete[:settings.num_speakers], start=1):
        for utterance, identifier in enumerate(shared[:settings.num_utterances], start=1):
            path = directory / f'{directory.name}_{identifier}_{settings.vctk_mic}.flac'
            recordings.append(Recording(speaker, utterance, path))

    return recordings
