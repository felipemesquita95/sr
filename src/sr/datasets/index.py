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

    Locutores sem a trilha pedida são omitidos, e os índices são reatribuídos de forma
    densa — evita classes vazias, que distorceriam as métricas macro.
    """
    if settings.vctk_root is None or not settings.vctk_root.is_dir():
        raise FileNotFoundError(f'Raiz do VCTK não encontrada: {settings.vctk_root}')

    suffix = f'_{settings.vctk_mic}.flac'
    speaker_dirs = sorted(
        d for d in settings.vctk_root.iterdir()
        if d.is_dir() and any(f.name.endswith(suffix) for f in d.iterdir())
    )

    skipped = len([d for d in settings.vctk_root.iterdir() if d.is_dir()]) - len(speaker_dirs)
    if skipped:
        logger.warning(
            'VCTK: %d locutores sem a trilha %s foram omitidos da numeração.',
            skipped, settings.vctk_mic,
        )

    recordings: list[Recording] = []
    for speaker, directory in enumerate(speaker_dirs[:settings.num_speakers], start=1):
        files = sorted(f for f in directory.iterdir() if f.name.endswith(suffix))
        for utterance, path in enumerate(files[:settings.num_utterances], start=1):
            recordings.append(Recording(speaker, utterance, path))

    return recordings
