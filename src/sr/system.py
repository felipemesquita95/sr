"""Orquestração do sistema de reconhecimento de locutor.

Compõe os quatro subsistemas e executa o protocolo experimental indicado pela
configuração. Os três protocolos diferem apenas em como os conjuntos são montados;
o treinamento e a avaliação são idênticos entre eles, o que é condição para que os
seus resultados sejam comparáveis entre si.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path

from sr.config import Settings
from sr.evaluation import evaluate_model, write_fold_report, write_summary
from sr.features import FeatureAdjustmentSubsystem
from sr.preprocessing import PreprocessingSubsystem
from sr.training import TrainingSubsystem

logger = logging.getLogger(__name__)


class SpeakerRecognitionSystem:
    """Sistema completo de identificação de locutor em conjunto fechado.

    Args:
        settings: Configuração do experimento, que determina o corpus, o protocolo
            e as arquiteturas a treinar.
    """

    def __init__(self, settings: Settings, resume: bool = False) -> None:
        self.settings = settings
        self.resume = resume
        self.preprocessing = PreprocessingSubsystem(settings)
        self.features = FeatureAdjustmentSubsystem(settings)
        self.training = TrainingSubsystem(settings)

    def run(self) -> None:
        """Executa o experimento descrito pela configuração.

        Despacha para o protocolo indicado: pré-processamento isolado, cruzamento de
        microfones, treino multi-microfone, ou validação cruzada convencional.
        """
        logger.info('%s', self.settings.describe())

        if self.settings.preprocess_only:
            self.preprocessing.run()
            return

        if self.settings.cross_mic:
            self.run_cross_microphone()
        elif self.settings.both_mics:
            self.run_multi_microphone()
        else:
            self.run_cross_validation()

    def run_cross_validation(self) -> None:
        """Executa a validação cruzada dentro de um mesmo canal de gravação.

        É o protocolo convencional, e o que produz os números mais altos. Os seus
        resultados devem ser lidos em conjunto com o diagnóstico de canal: dentro de
        um mesmo canal, parte da acurácia pode não ser atribuível à voz.
        """
        self.preprocessing.run()

        for architecture in self.settings.architectures:
            accuracies: list[float] = []
            f1_scores: list[float] = []

            for fold in range(1, self.settings.effective_folds + 1):
                if self.resume:
                    previous = self._completed_fold_metrics(architecture, fold)
                    if previous is not None:
                        logger.info('Reaproveitando %s, partição %d.', architecture, fold)
                        accuracies.append(previous[0])
                        f1_scores.append(previous[1])
                        continue
                split = self.features.prepare_fold(fold)
                result = self._train_and_evaluate(architecture, split, fold)
                accuracies.append(result.accuracy)
                f1_scores.append(result.f1)

            self._write_summary(architecture, accuracies, f1_scores)

    def _completed_fold_metrics(self, architecture: str, fold: int) -> tuple[float, float] | None:
        """Lê uma partição completa do mesmo protocolo, ou pede novo treino."""
        output = self._fold_directory(architecture, fold)
        if not (output / 'modelo.keras').is_file():
            return None
        try:
            division = json.loads((output / 'divisao.json').read_text())
            metrics = json.loads((output / 'metricas.json').read_text())
            expected = {
                'particao': fold,
                'validation_fold_offset': self.settings.validation_fold_offset,
                'num_folds': self.settings.num_folds,
                'validation_seed': self.settings.validation_seed,
            }
            if any(division.get(key) != value for key, value in expected.items()):
                return None
            if (metrics['num_classes'] != self.settings.num_speakers or
                    metrics['num_amostras_teste'] != division['teste'] or
                    min(division['treino'], division['validacao'], division['teste']) <= 0):
                return None
            accuracy = float(metrics['acuracia'])
            f1 = float(metrics['f1_macro'])
            if not (0 <= accuracy <= 1 and 0 <= f1 <= 1):
                return None
            return accuracy, f1
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def run_cross_microphone(self) -> None:
        """Executa o protocolo cross-mic: treina em um microfone, avalia no outro.

        Não há partições: a divisão é única e determinada pelos microfones. O
        resultado é o piso honesto de identidade vocal, imune à assinatura do canal
        de treino.
        """
        logger.info('Protocolo cross-mic: treino em %s, teste em %s.',
                    self.settings.features_path_train, self.settings.features_path_test)

        split = self.features.prepare_cross_microphone()
        for architecture in self.settings.architectures:
            result = self._train_and_evaluate(architecture, split, fold=1)
            self._write_summary(architecture, [result.accuracy], [result.f1])

    def run_multi_microphone(self) -> None:
        """Executa o protocolo multi-mic, com validação cruzada por enunciado.

        Todos os microfones entram no treino, o que descorrelaciona a assinatura do
        canal do rótulo do locutor. Comparar este resultado com o cross-mic indica
        se a exposição a múltiplos canais durante o treino de fato remove o atalho.
        """
        logger.info('Protocolo multi-mic sobre %d conjuntos de features.',
                    len(self.settings.features_paths))

        for architecture in self.settings.architectures:
            accuracies: list[float] = []
            f1_scores: list[float] = []

            for fold in range(1, self.settings.effective_folds + 1):
                split = self.features.prepare_multi_microphone(fold)
                result = self._train_and_evaluate(architecture, split, fold)
                accuracies.append(result.accuracy)
                f1_scores.append(result.f1)

            self._write_summary(architecture, accuracies, f1_scores)

    def _train_and_evaluate(self, architecture: str, split, fold: int):
        """Treina uma arquitetura em uma partição e grava o relatório correspondente.

        Args:
            architecture: Nome da arquitetura.
            split: Conjuntos já montados e normalizados.
            fold: Índice da partição.

        Returns:
            As métricas obtidas no conjunto de teste.
        """
        output = self._fold_directory(architecture, fold)
        logger.info('Treinando %r na partição %d.', architecture, fold)

        model, history = self.training.train(architecture, split, output, fold=fold)
        result = evaluate_model(model, split.test_x, split.test_y, self.settings.num_speakers)
        write_fold_report(result, history, output)
        (output / 'divisao.json').write_text(json.dumps({
            'particao': fold,
            'treino': int(len(split.train_y)),
            'validacao': int(len(split.validation_y)),
            'teste': int(len(split.test_y)),
            'validation_fold_offset': self.settings.validation_fold_offset,
            'num_folds': self.settings.num_folds,
            'validation_seed': self.settings.validation_seed,
            'quadros_por_gravacao': int(split.input_shape[-1]),
        }, indent=2, ensure_ascii=False))
        return result

    def _write_summary(self, architecture: str, accuracies: list[float],
                       f1_scores: list[float]) -> None:
        """Grava o resumo de uma arquitetura sobre as partições executadas."""
        write_summary(
            architecture, accuracies, f1_scores,
            chance_level=1.0 / self.settings.num_speakers,
            output=self.settings.models_path / architecture,
        )

    def _fold_directory(self, architecture: str, fold: int) -> Path:
        """Diretório de saída de uma partição de uma arquitetura."""
        return self.settings.models_path / architecture / f'particao{fold}'
