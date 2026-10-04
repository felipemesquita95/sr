"""Montagem dos conjuntos de treino, validação e teste a partir das features.

Este subsistema decide **quem vai para onde**, e é portanto onde vazamentos de
informação se originam. Três garantias são mantidas por construção e valem para
todos os protocolos implementados aqui:

1. **O conjunto de teste não ajusta pesos nem escolhe épocas.** A validação usada
   pelo early-stopping é separada do treino, jamais do teste. Selecionar a melhor
   época medindo no teste tornaria a acurácia reportada otimista.

2. **O comprimento padrão vem apenas do treino.** No perfil de reexecução com
   ``max_frames_cap=-1``, a largura é fixada pelo menor enunciado de todo o
   corpus, inclusive os de teste. Isso usa metadados de duração do teste, não o
   sinal nem o rótulo, e deve ser declarado ao interpretar os resultados.

3. **Média e desvio da normalização vêm apenas do treino.** As estatísticas são
   então aplicadas inalteradas a validação e teste, como ocorreria em produção,
   onde as amostras futuras não estão disponíveis para normalizar o modelo.
"""

from __future__ import annotations

import gc
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sr.config import Settings

logger = logging.getLogger(__name__)

#: Nome do arquivo de features gravado pelo subsistema de pré-processamento.
MFCC_FILENAME = 'mfccs.npy'

#: Estabilizador somado ao desvio padrão para evitar divisão por zero.
EPSILON = 1e-8


@dataclass(frozen=True)
class DataSplit:
    """Os três conjuntos de um experimento, já como tensores normalizados.

    Attributes:
        train_x: Tensor de treino, de forma ``(n, num_mfccs, num_quadros)``.
        train_y: Rótulos de treino, base zero.
        validation_x: Tensor de validação, usado pelo early-stopping.
        validation_y: Rótulos de validação.
        test_x: Tensor de teste, usado somente na avaliação final.
        test_y: Rótulos de teste.
        normalization_mean: Média por coeficiente estimada no treino, quando disponível.
        normalization_std: Desvio do treino com estabilizador, quando disponível.
    """

    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    normalization_mean: np.ndarray | None = None
    normalization_std: np.ndarray | None = None

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Forma de uma amostra, como esperado pela camada de entrada da rede."""
        return self.train_x.shape[1:]

    def describe(self) -> str:
        """Resumo dos tamanhos dos três conjuntos, para registro no log."""
        return (f'treino={self.train_x.shape} '
                f'validação={self.validation_x.shape} '
                f'teste={self.test_x.shape}')


#: Um item ainda não alinhado: a matriz de MFCCs e o rótulo do seu locutor.
Item = tuple[np.ndarray, int]

RecordingKey = tuple[int, int]


@dataclass(frozen=True)
class PairedPartition:
    """Papéis dos enunciados, compartilhados por ambas as capturas e todos os treinos.

    As chaves são ``(locutor, enunciado)`` com índices base um. A semente governa
    apenas esta divisão; nenhuma semente de inicialização da rede entra aqui.
    """

    train: tuple[RecordingKey, ...]
    validation: tuple[RecordingKey, ...]
    test: tuple[RecordingKey, ...]
    seed: int

    def __post_init__(self) -> None:
        groups = [set(self.train), set(self.validation), set(self.test)]
        if any(not group for group in groups):
            raise ValueError('A divisão pareada exige treino, validação e teste não vazios.')
        if any(len(keys) != len(group) for keys, group in
               zip((self.train, self.validation, self.test), groups)):
            raise ValueError('Enunciado duplicado na divisão pareada.')
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError('Os papéis da divisão pareada devem ser disjuntos.')


@dataclass(frozen=True)
class PairedDataSplit:
    """Um treino de origem e dois testes sobre a mesma lista de gravações.

    ``source`` contém treino, validação e teste da origem. O teste de destino
    recebe exatamente o comprimento, a média e o desvio estimados nesse treino.
    """

    source: DataSplit
    target_test_x: np.ndarray
    target_test_y: np.ndarray
    partition: PairedPartition


class FeatureAdjustmentSubsystem:
    """Monta os conjuntos de treino, validação e teste segundo o protocolo escolhido.

    Args:
        settings: Configuração do experimento.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    # ------------------------------------------------------------------
    # Leitura das features
    # ------------------------------------------------------------------

    def load_features(self, features_path: Path | None = None) -> dict[tuple[int, int], np.ndarray]:
        """Carrega da pasta de features todas as matrizes de MFCC disponíveis.

        Args:
            features_path: Diretório a ler. Se omitido, usa o do experimento; o
                parâmetro existe para os protocolos que leem de dois diretórios,
                um por microfone.

        Returns:
            Mapeamento de ``(locutor, enunciado)`` para matriz de forma
            ``(num_mfccs, num_quadros)``.

        Raises:
            FileNotFoundError: Se o diretório de features não existir.
        """
        path = features_path or self.settings.features_path
        if not path.is_dir():
            raise FileNotFoundError(
                f'Features não encontradas em {path}. Rode o pré-processamento antes.')

        features: dict[tuple[int, int], np.ndarray] = {}
        missing = 0
        for speaker in range(1, self.settings.num_speakers + 1):
            for utterance in range(1, self.settings.num_utterances + 1):
                file = path / str(speaker) / str(utterance) / MFCC_FILENAME
                if file.exists():
                    matrix = np.load(file)
                    if matrix.shape[1] >= self.settings.min_frames:
                        features[(speaker, utterance)] = matrix
                else:
                    missing += 1

        if missing:
            # Ausências são esperadas quando o número de enunciados varia por locutor.
            logger.info('%s: %d enunciados carregados, %d índices ausentes.',
                        path.name, len(features), missing)
        return features

    # ------------------------------------------------------------------
    # Alinhamento de comprimento
    # ------------------------------------------------------------------

    @staticmethod
    def pad_or_truncate(matrix: np.ndarray, num_frames: int) -> np.ndarray:
        """Ajusta o número de quadros de uma matriz para um comprimento fixo.

        Enunciados mais longos são truncados. Enunciados mais curtos são estendidos
        por **repetição** do próprio conteúdo, e não por preenchimento com zeros.

        A escolha é deliberada e específica deste problema: preencher com zeros
        introduziria um trecho de silêncio artificial cujo comprimento depende da
        duração original do enunciado. Como o silêncio é justamente o portador da
        assinatura de canal que este trabalho investiga, o preenchimento criaria uma
        pista correlacionada com o locutor — exatamente o artefato sob estudo.

        Args:
            matrix: Matriz de forma ``(num_mfccs, quadros)``.
            num_frames: Número de quadros desejado.

        Returns:
            Matriz de forma ``(num_mfccs, num_frames)``.
        """
        current = matrix.shape[1]
        if current == num_frames:
            return matrix
        if current > num_frames:
            return matrix[:, :num_frames]

        repetitions = math.ceil(num_frames / current)
        return np.tile(matrix, (1, repetitions))[:, :num_frames]

    # ------------------------------------------------------------------
    # Protocolo 1: validação cruzada dentro de um mesmo canal
    # ------------------------------------------------------------------

    def fold_roles(self, available: list[int], fold: int,
                   rng: np.random.Generator) -> dict[int, str]:
        """Define papéis por gravação sem depender dos valores dos MFCCs.

        Com deslocamento 1, grupos consecutivos fornecem teste e validação;
        os outros grupos ficam no treino. Assim cada gravação passa uma vez
        pelo teste e uma vez pela validação nas cinco partições.
        """
        if not 1 <= fold <= self.settings.num_folds:
            raise ValueError('Partição fora do intervalo configurado.')
        test_group = fold - 1
        validation_group = (test_group + self.settings.validation_fold_offset) % self.settings.num_folds
        roles = {}
        for position, utterance in enumerate(available):
            group = position % self.settings.num_folds
            roles[utterance] = ('teste' if group == test_group else
                                'validacao' if self.settings.validation_fold_offset and
                                group == validation_group else 'treino')
        if self.settings.validation_fold_offset == 0:
            candidates = [u for u in available if roles[u] == 'treino']
            if len(candidates) >= 2:
                roles[int(rng.choice(candidates))] = 'validacao'
        return roles

    def prepare_fold(self, fold: int) -> DataSplit:
        """Monta a partição ``fold`` da validação cruzada.

        Os enunciados de cada locutor são distribuídos em ``num_folds`` grupos pela
        sua posição ordenada. Na partição ``k``, o grupo ``k`` de cada locutor vai
        para teste. Por padrão, uma gravação por locutor sai do restante para
        validação. Com ``validation_fold_offset=1``, o grupo seguinte inteiro
        vai para validação e os outros três grupos ficam no treino.

        Quando o número de enunciados iguala o número de partições, o esquema recai
        no *leave-one-utterance-out* clássico. No BrSD isso tem uma propriedade
        desejável: como todos os locutores leram os mesmos cinco textos, a partição
        ``k`` retém o texto ``k`` de **todos** os locutores, de modo que o conteúdo
        linguístico do teste nunca aparece no treino. Memorização de texto fica assim
        descartada como explicação para o desempenho.

        Args:
            fold: Índice da partição, de 1 a ``num_folds``.

        Returns:
            Os três conjuntos, já alinhados e normalizados.
        """
        features = self.load_features()
        rng = np.random.default_rng(self.settings.validation_seed)

        train: list[Item] = []
        validation: list[Item] = []
        test: list[Item] = []

        for speaker in range(1, self.settings.num_speakers + 1):
            label = speaker - 1
            available = sorted(
                u for u in range(1, self.settings.num_utterances + 1)
                if (speaker, u) in features
            )
            # Na variante filtrada, preserve o papel que cada índice tinha no
            # experimento completo. Renumerar só os sobreviventes trocaria
            # treino/validação/teste entre os dois experimentos.
            role_indices = (list(range(1, self.settings.num_utterances + 1))
                            if self.settings.min_frames and self.settings.validation_fold_offset
                            else available)
            roles = self.fold_roles(role_indices, fold, rng)

            for utterance in available:
                item = (features[(speaker, utterance)], label)
                if roles[utterance] == 'teste':
                    test.append(item)
                elif roles[utterance] == 'validacao':
                    validation.append(item)
                else:
                    train.append(item)

        return self._finalize(train, validation, test, label=f'fold {fold}/{self.settings.num_folds}')

    # ------------------------------------------------------------------
    # Protocolo 2: treino em um microfone, avaliação em outro
    # ------------------------------------------------------------------

    @staticmethod
    def _split_utterances(
        train_utterances: list[int],
        test_utterances: list[int],
    ) -> tuple[list[int], list[int]]:
        """Divide os enunciados de um locutor em metades disjuntas entre as trilhas.

        O corte é feito sobre a **interseção** das duas trilhas, e não sobre cada uma
        isoladamente: se um enunciado existe só na trilha de treino, incluí-lo
        deslocaria o ponto de corte entre elas e um mesmo enunciado poderia cair nos
        dois lados. A ordem é a numérica, que é determinística e independe de sorteio.

        Args:
            train_utterances: Enunciados presentes na trilha de treino.
            test_utterances: Enunciados presentes na trilha de teste.

        Returns:
            Par ``(treino, teste)`` sem enunciado em comum.
        """
        shared = sorted(set(train_utterances) & set(test_utterances))
        middle = len(shared) // 2
        return shared[:middle], shared[middle:]

    def prepare_cross_microphone(self) -> DataSplit:
        """Monta o protocolo cross-mic: treina em um microfone, avalia no outro.

        No VCTK cada frase é captada simultaneamente por dois microfones. A voz, o
        texto e o instante são idênticos entre as duas trilhas; apenas o transdutor
        muda. Um modelo que tenha aprendido a assinatura espectral do microfone de
        treino não a encontra no de teste, e o seu desempenho desaba.

        A distância entre a acurácia dentro do mesmo microfone e a acurácia cruzando
        microfones é, portanto, uma medida direta de quanto do desempenho aparente se
        devia ao canal. O que resta é o piso honesto atribuível à voz.

        Returns:
            Os três conjuntos, com teste proveniente exclusivamente do outro microfone.
        """
        settings = self.settings
        train_features = self.load_features(settings.features_path_train)
        test_features = self.load_features(settings.features_path_test)
        rng = np.random.default_rng(settings.validation_seed)

        train: list[Item] = []
        validation: list[Item] = []
        test: list[Item] = []

        for speaker in range(1, settings.num_speakers + 1):
            label = speaker - 1
            train_utterances = sorted(
                u for u in range(1, settings.num_utterances + 1) if (speaker, u) in train_features)
            test_utterances = sorted(
                u for u in range(1, settings.num_utterances + 1) if (speaker, u) in test_features)

            if settings.cross_mic_disjoint_utterances:
                train_utterances, test_utterances = self._split_utterances(
                    train_utterances, test_utterances)

            # A validação sai do microfone de treino: medir a época ótima no microfone
            # de teste seria selecionar o modelo pelo próprio efeito sob investigação.
            reserved = min(settings.cross_mic_validation_per_speaker,
                           max(0, len(train_utterances) - 1))
            validation_set = set()
            if reserved > 0:
                validation_set = {
                    int(u) for u in rng.choice(train_utterances, size=reserved, replace=False)
                }

            for utterance in train_utterances:
                item = (train_features[(speaker, utterance)], label)
                (validation if utterance in validation_set else train).append(item)
            for utterance in test_utterances:
                test.append((test_features[(speaker, utterance)], label))

        return self._finalize(train, validation, test, label='cross-mic')

    # ------------------------------------------------------------------
    # Protocolo 3: treino com múltiplos microfones
    # ------------------------------------------------------------------

    def prepare_multi_microphone(self, fold: int) -> DataSplit:
        """Monta o protocolo multi-mic: treina com todos os microfones disponíveis.

        A partição é feita por **enunciado**, e todas as trilhas de microfone de um
        mesmo enunciado vão para o mesmo lado da divisão. Assim a mesma frase nunca
        aparece simultaneamente em treino e teste, e cada locutor é visto no treino
        através de mais de um canal — o que descorrelaciona a assinatura do microfone
        do rótulo do locutor e remove o atalho.

        Args:
            fold: Índice da partição, de 1 a ``num_folds``.

        Returns:
            Os três conjuntos, contendo todas as trilhas de microfone.
        """
        settings = self.settings
        sources = [self.load_features(path) for path in settings.features_paths]
        rng = np.random.default_rng(settings.validation_seed)

        train: list[Item] = []
        validation: list[Item] = []
        test: list[Item] = []

        for speaker in range(1, settings.num_speakers + 1):
            label = speaker - 1
            available = sorted(
                u for u in range(1, settings.num_utterances + 1)
                if any((speaker, u) in source for source in sources)
            )
            test_utterances = {
                u for position, u in enumerate(available)
                if position % settings.num_folds == (fold - 1)
            }
            candidates = [u for u in available if u not in test_utterances]
            validation_utterance = int(rng.choice(candidates)) if len(candidates) >= 2 else None

            for utterance in available:
                if utterance in test_utterances:
                    bucket = test
                elif utterance == validation_utterance:
                    bucket = validation
                else:
                    bucket = train
                # Todas as trilhas deste enunciado vão para o mesmo conjunto.
                for source in sources:
                    if (speaker, utterance) in source:
                        bucket.append((source[(speaker, utterance)], label))

        return self._finalize(train, validation, test,
                              label=f'multi-mic fold {fold}/{settings.num_folds}')

    # ------------------------------------------------------------------
    # Alinhamento, montagem e normalização
    # ------------------------------------------------------------------

    def paired_partition(
        self,
        first: dict[RecordingKey, np.ndarray],
        second: dict[RecordingKey, np.ndarray],
        *,
        seed: int,
        test_fraction: float = 0.2,
        validation_per_speaker: int = 10,
    ) -> PairedPartition:
        """Reserva os mesmos enunciados em ambas as trilhas, antes de qualquer treino.

        Usa somente pares presentes nas duas capturas, em ordem canônica antes do
        sorteio. Por locutor, reserva ``ceil(n * test_fraction)`` pares para teste,
        depois a quantidade declarada para validação. Exige ao menos um par de
        treino por classe; reduzir silenciosamente a validação mudaria a medida.
        """
        if not 0 < test_fraction < 1 or validation_per_speaker < 1:
            raise ValueError('Fração de teste e quantidade de validação inválidas.')
        shared = set(first) & set(second)
        rng = np.random.default_rng(seed)
        train, validation, test = [], [], []
        for speaker in range(1, self.settings.num_speakers + 1):
            keys = sorted(key for key in shared if key[0] == speaker)
            n_test = math.ceil(len(keys) * test_fraction)
            if len(keys) <= n_test + validation_per_speaker:
                raise ValueError(f'Locutor {speaker} sem pares suficientes para os três papéis.')
            order = rng.permutation(len(keys))
            test.extend(keys[int(i)] for i in order[:n_test])
            validation.extend(keys[int(i)] for i in
                              order[n_test:n_test + validation_per_speaker])
            train.extend(keys[int(i)] for i in order[n_test + validation_per_speaker:])
        return PairedPartition(tuple(sorted(train)), tuple(sorted(validation)),
                               tuple(sorted(test)), seed)

    def prepare_paired_microphone(
        self,
        source: dict[RecordingKey, np.ndarray],
        target: dict[RecordingKey, np.ndarray],
        partition: PairedPartition,
    ) -> PairedDataSplit:
        """Monta uma linha da matriz, conservando as estatísticas da origem.

        Os dois testes entram juntos na montagem para receber a mesma transformação.
        A validação usa exclusivamente a origem. Não há novo sorteio nesta etapa,
        nem seleção de enunciados em função da semente de treino.
        """
        if self.settings.permute_labels:
            raise ValueError('A matriz pareada exige os rótulos verdadeiros.')
        keys = set(partition.train + partition.validation + partition.test)
        if not keys <= source.keys() or not keys <= target.keys():
            raise ValueError('Todos os enunciados da divisão devem existir nas duas trilhas.')

        def items(features, selected):
            return [(features[key], key[0] - 1) for key in selected]

        combined = self._finalize(
            items(source, partition.train), items(source, partition.validation),
            items(source, partition.test) + items(target, partition.test), label='mic pareado')
        n_test = len(partition.test)
        origin = DataSplit(
            combined.train_x, combined.train_y, combined.validation_x, combined.validation_y,
            combined.test_x[:n_test], combined.test_y[:n_test],
            combined.normalization_mean, combined.normalization_std)
        return PairedDataSplit(origin, combined.test_x[n_test:], combined.test_y[n_test:], partition)

    def _permute_labels(
        self,
        train: list[Item],
        validation: list[Item],
        test: list[Item],
    ) -> tuple[list[Item], list[Item], list[Item]]:
        """Redistribui os rótulos ao acaso entre todas as gravações da partição.

        O embaralhamento é feito sobre o conjunto reunido, e não dentro de cada
        subconjunto, porque é a associação gravação↔locutor que precisa ser destruída
        — inclusive a que atravessa a fronteira entre treino e teste. Permutar cada
        lado isoladamente preservaria essa fronteira e deixaria passar exatamente o
        vazamento que o controle existe para detectar.

        A permutação é uma reordenação dos rótulos existentes, e não um sorteio novo:
        a contagem de exemplos por classe fica idêntica à do experimento verdadeiro,
        de modo que o acaso continua sendo ``1/num_speakers`` e as duas execuções são
        comparáveis. O que muda é apenas *qual* gravação recebe *qual* rótulo.

        Args:
            train: Itens de treino.
            validation: Itens de validação.
            test: Itens de teste.

        Returns:
            Os três conjuntos, com as mesmas matrizes e os rótulos redistribuídos.
        """
        sizes = (len(train), len(validation), len(test))
        labels = np.asarray([label for group in (train, validation, test) for _, label in group])

        rng = np.random.default_rng(self.settings.permutation_seed)
        rng.shuffle(labels)

        matrices = [matrix for group in (train, validation, test) for matrix, _ in group]
        shuffled = list(zip(matrices, (int(label) for label in labels)))

        first, second = sizes[0], sizes[0] + sizes[1]
        logger.warning(
            'CONTROLE DE PERMUTAÇÃO ATIVO (semente %d): %d rótulos redistribuídos. '
            'Acurácia acima do acaso aqui indica vazamento no arcabouço.',
            self.settings.permutation_seed, len(labels))
        return shuffled[:first], shuffled[first:second], shuffled[second:]

    def _finalize(
        self,
        train: list[Item],
        validation: list[Item],
        test: list[Item],
        label: str,
    ) -> DataSplit:
        """Alinha comprimentos, monta os tensores e normaliza com estatísticas do treino.

        Args:
            train: Itens de treino.
            validation: Itens de validação.
            test: Itens de teste.
            label: Identificação da partição, apenas para o log.

        Returns:
            Os três conjuntos como tensores normalizados.

        Raises:
            RuntimeError: Se o conjunto de treino estiver vazio.
        """
        if not train:
            raise RuntimeError(
                'Conjunto de treino vazio. Verifique se o pré-processamento foi executado '
                'e se num_speakers e num_utterances correspondem às features em disco.')

        if self.settings.permute_labels:
            train, validation, test = self._permute_labels(train, validation, test)

        if self.settings.max_frames_cap == -1:
            # Perfil de duração mínima: cada gravação do corpus participa da escolha.
            # Assim todas as partições do mesmo corpus usam a mesma largura.
            num_frames = min(matrix.shape[1] for group in (train, validation, test)
                             for matrix, _ in group)
        else:
            # Perfil padrão: a largura vem apenas do treino.
            num_frames = max(matrix.shape[1] for matrix, _ in train)
        if self.settings.max_frames_cap > 0:
            num_frames = min(num_frames, self.settings.max_frames_cap)

        num_mfccs = train[0][0].shape[0]
        logger.info('%s: alinhando em %d quadros (%d coeficientes).', label, num_frames, num_mfccs)

        def build(items: list[Item]) -> tuple[np.ndarray, np.ndarray]:
            # Pré-aloca e preenche no lugar: montar uma lista intermediária e depois
            # convertê-la dobraria o pico de memória, que é o recurso limitante aqui.
            x = np.empty((len(items), num_mfccs, num_frames), dtype=np.float32)
            y = np.empty(len(items), dtype=np.int64)
            for index, (matrix, item_label) in enumerate(items):
                x[index] = self.pad_or_truncate(matrix, num_frames)
                y[index] = item_label
            return x, y

        train_x, train_y = build(train)
        validation_x, validation_y = build(validation)
        test_x, test_y = build(test)

        del train, validation, test
        gc.collect()

        # Garantia 3: média e desvio por coeficiente, calculados somente sobre o treino.
        mean = train_x.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        std = (train_x.std(axis=(0, 2), keepdims=True) + EPSILON).astype(np.float32)

        for tensor in (train_x, validation_x, test_x):
            tensor -= mean
            tensor /= std

        split = DataSplit(train_x, train_y, validation_x, validation_y, test_x, test_y, mean, std)
        logger.info('%s: %s', label, split.describe())
        return split
