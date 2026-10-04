"""Orquestração do pré-processamento: do corpus indexado às features em disco.

O subsistema percorre o índice do corpus, aplica a cadeia de processamento a cada
gravação e persiste a matriz de coeficientes resultante. A persistência é o que
permite treinar muitas arquiteturas sobre exatamente as mesmas features, sem
reprocessar o áudio a cada execução — e é também o que garante que uma comparação
entre arquiteturas isole a arquitetura como única variável.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from sr.config import Settings
from sr.datasets import Recording, build_index
from sr.preprocessing import signal, visualization

logger = logging.getLogger(__name__)

#: Nome do arquivo de features gravado para cada enunciado.
MFCC_FILENAME = 'mfccs.npy'

#: Nome do arquivo de derivadas de primeira ordem (gravado apenas sob demanda).
DELTA_FILENAME = 'delta.npy'

#: Nome do arquivo de derivadas de segunda ordem (gravado apenas sob demanda).
DELTA_DELTA_FILENAME = 'delta_delta.npy'

#: Subdiretório das figuras de nível de conjunto.
SUMMARY_DIRNAME = '_resumo'


class PreprocessingSubsystem:
    """Converte o corpus de áudio em matrizes de coeficientes cepstrais.

    Args:
        settings: Configuração do experimento.
        compute_deltas: Se verdadeiro, também calcula e grava as derivadas de
            primeira e segunda ordem dos coeficientes. Desativado por padrão porque
            triplica o espaço em disco e, nos experimentos conduzidos, não melhorou
            a acurácia (ver ``docs/resultados.md``).
    """

    def __init__(self, settings: Settings, compute_deltas: bool = False) -> None:
        self.settings = settings
        self.compute_deltas = compute_deltas

    def run(self) -> None:
        """Processa todo o corpus, gravando as features em ``settings.features_path``.

        Gravações já processadas são detectadas pela presença do arquivo de saída e
        puladas, de modo que a execução é retomável — relevante para o VCTK, cujo
        pré-processamento completo leva horas.

        Quando o corpus não está mais em disco mas as features estão, a etapa é
        dispensada em vez de falhar. O VCTK é ingerido apagando cada áudio logo após
        convertê-lo, porque o corpus não cabe em disco junto do próprio zip; exigir a
        forma de onda para treinar sobre features já extraídas tornaria o resultado
        irreprodutível justamente na máquina onde ele foi obtido.

        Raises:
            FileNotFoundError: Se nem o corpus nem as features existirem.
        """
        try:
            recordings = build_index(self.settings)
        except FileNotFoundError:
            if not self._features_present():
                raise
            logger.warning(
                'Corpus indisponível em %s; usando as features já extraídas em %s. '
                'Nenhuma gravação será reprocessada.',
                self.settings.vctk_root or self.settings.audio_path,
                self.settings.features_path)
            return
        logger.info('Pré-processando %d gravações para %s.',
                    len(recordings), self.settings.features_path)

        processed, skipped = self.process(recordings)

        logger.info('Pré-processamento concluído: %d processadas, %d já existiam.',
                    processed, skipped)
        self._write_summary_figures()

    def _features_present(self) -> bool:
        """Indica se há features do primeiro locutor já gravadas em disco.

        A verificação é deliberadamente barata e local: basta uma evidência de que a
        extração já ocorreu para esta configuração. Conferir o conjunto inteiro
        custaria uma varredura de dezenas de milhares de diretórios, e a montagem dos
        tensores logo adiante já falha, com mensagem própria, se faltar material.
        """
        return (self.settings.features_path / '1').is_dir()

    def process(self, recordings: Sequence[Recording]) -> tuple[int, int]:
        """Aplica a cadeia a uma lista explícita de gravações, já numeradas.

        Separar isto de :meth:`run` permite que o chamador forneça o índice em vez de
        deixá-lo ser derivado do conteúdo do diretório. A ingestão incremental do VCTK
        depende disso: como o corpus é processado um locutor por vez, com o áudio
        apagado em seguida, uma numeração derivada do que está em disco naquele
        instante atribuiria o índice 1 a cada locutor sucessivamente, sobrescrevendo
        as features do anterior.

        Args:
            recordings: Gravações a processar, com locutor e enunciado já atribuídos.

        Returns:
            Par ``(processadas, já existentes)``.
        """
        total = len(recordings)
        processed = skipped = 0

        for position, recording in enumerate(recordings, start=1):
            destination = self._destination(recording)
            if (destination / MFCC_FILENAME).exists():
                skipped += 1
                continue

            destination.mkdir(parents=True, exist_ok=True)
            with_figures = self.settings.enable_plots or position <= self.settings.num_plot_examples
            self._process_one(recording, destination, with_figures)

            processed += 1
            if processed % 25 == 0 or position == total:
                logger.info('  [%d/%d] locutor %d, enunciado %d',
                            position, total, recording.speaker, recording.utterance)

        return processed, skipped

    def _destination(self, recording: Recording) -> Path:
        """Diretório de saída de uma gravação, no formato ``<features>/<locutor>/<enunciado>``."""
        return self.settings.features_path / str(recording.speaker) / str(recording.utterance)

    def _process_one(self, recording: Recording, destination: Path, with_figures: bool) -> None:
        """Aplica a cadeia completa a uma gravação e grava os resultados.

        Args:
            recording: Gravação a processar.
            destination: Diretório onde gravar features e figuras.
            with_figures: Se verdadeiro, grava também as figuras de cada estágio.
        """
        import librosa

        settings = self.settings
        # Preserva a taxa nativa: cinco WAVs do BrSD são 44,1 kHz. Converter no
        # carregamento faria uma reamostragem ocorrer antes do filtro anti-aliasing.
        audio, sampling_rate = librosa.load(recording.path, sr=None)
        if sampling_rate != settings.source_sampling_rate:
            logger.info('Taxa nativa de %s: %d Hz (perfil: %d Hz).',
                        recording.path, sampling_rate, settings.source_sampling_rate)

        if with_figures:
            visualization.plot_waveform(
                audio, sampling_rate, destination / 'sinal_original.png',
                'Sinal no domínio do tempo (original)')
            visualization.plot_spectrum(
                audio, sampling_rate, destination / 'espectro_original.png',
                'Espectro do sinal original')

        if settings.enable_vad:
            audio = signal.remove_silence(audio, settings.vad_top_db)
            if with_figures:
                visualization.plot_waveform(
                    audio, sampling_rate, destination / 'sinal_vad.png',
                    'Sinal após remoção de trechos sem voz')

        filtered = signal.antialias_filter(audio, sampling_rate, settings.target_sampling_rate)
        resampled = signal.resample(filtered, sampling_rate, settings.target_sampling_rate)
        emphasized = signal.pre_emphasis(resampled, settings.pre_emphasis_coef)

        if with_figures:
            visualization.plot_spectrum(
                filtered, sampling_rate, destination / 'espectro_filtrado.png',
                'Espectro após filtragem anti-aliasing')
            visualization.plot_spectrum(
                resampled, settings.target_sampling_rate, destination / 'espectro_reamostrado.png',
                f'Espectro após reamostragem para {settings.target_sampling_rate} Hz')
            visualization.plot_spectrum(
                emphasized, settings.target_sampling_rate, destination / 'espectro_preenfase.png',
                'Espectro após pré-ênfase')

        mfccs = signal.extract_mfccs(
            emphasized, settings.target_sampling_rate, settings.num_mfccs, settings.frame_size)
        np.save(destination / MFCC_FILENAME, mfccs.astype(np.float32))

        if with_figures:
            visualization.plot_mfccs(mfccs, destination / 'mfccs.png', 'Matriz de MFCCs')

        if self.compute_deltas:
            delta = librosa.feature.delta(mfccs)
            delta_delta = librosa.feature.delta(mfccs, order=2)
            np.save(destination / DELTA_FILENAME, delta.astype(np.float32))
            np.save(destination / DELTA_DELTA_FILENAME, delta_delta.astype(np.float32))

        del audio, filtered, resampled, emphasized
        gc.collect()

    def _write_summary_figures(self) -> None:
        """Gera as figuras que descrevem o conjunto como um todo.

        Lê apenas os cabeçalhos dos arquivos de features (``mmap_mode='r'``) para
        obter as dimensões sem carregar as matrizes na memória.
        """
        frame_counts: list[int] = []
        per_speaker: dict[int, int] = {}

        for speaker in range(1, self.settings.num_speakers + 1):
            speaker_dir = self.settings.features_path / str(speaker)
            if not speaker_dir.is_dir():
                continue
            for utterance_dir in speaker_dir.iterdir():
                mfcc_file = utterance_dir / MFCC_FILENAME
                if mfcc_file.exists():
                    frame_counts.append(int(np.load(mfcc_file, mmap_mode='r').shape[1]))
                    per_speaker[speaker] = per_speaker.get(speaker, 0) + 1

        if not frame_counts:
            logger.warning('Nenhuma feature encontrada; figuras de resumo não geradas.')
            return

        summary_dir = self.settings.features_path / SUMMARY_DIRNAME
        summary_dir.mkdir(parents=True, exist_ok=True)
        visualization.plot_duration_distribution(frame_counts, summary_dir / 'duracao.png')
        visualization.plot_utterances_per_speaker(per_speaker, summary_dir / 'enunciados_por_locutor.png')

        counts = np.asarray(frame_counts)
        logger.info(
            'Resumo do conjunto: %d enunciados, quadros mín=%d, média=%.0f, p95=%.0f, máx=%d.',
            len(counts), counts.min(), counts.mean(), np.percentile(counts, 95), counts.max(),
        )
        logger.info('Figuras de resumo em %s', summary_dir)
