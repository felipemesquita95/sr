"""Configuração tipada do sistema, carregada de um arquivo de perfil.

Cada experimento é descrito por um arquivo de perfil (`configs/*.env`) contendo
pares ``CHAVE=valor``. Este módulo lê o perfil uma única vez e o expõe como uma
estrutura imutável e validada, em vez de espalhar chamadas a ``os.getenv`` pelos
subsistemas.

A motivação é reprodutibilidade: um experimento fica descrito por exatamente um
arquivo, versionável e citável no texto do trabalho.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Sequence

from dotenv import dotenv_values

#: Variável de ambiente que aponta para o arquivo de perfil a carregar.
CONFIG_ENV_VAR = 'SR_CONFIG'

#: Perfil usado quando ``SR_CONFIG`` não está definida.
DEFAULT_CONFIG = 'configs/brsd.env'


def _as_bool(value: Any) -> bool:
    """Interpreta valores textuais de perfil como booleanos."""
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def _as_path_list(value: Any) -> list[Path]:
    """Interpreta uma lista de caminhos separados por vírgula."""
    if not value:
        return []
    return [Path(p.strip()).expanduser() for p in str(value).split(',') if p.strip()]


@dataclass(frozen=True)
class Settings:
    """Parâmetros de um experimento.

    Os atributos espelham as chaves do arquivo de perfil. Todos os caminhos são
    resolvidos para absolutos na construção, de modo que o sistema independe do
    diretório de trabalho.
    """

    # ---- Identificação do experimento ------------------------------------
    #: Nome legível do experimento, usado em logs e no nome dos diretórios de saída.
    experiment_name: str = 'experimento'

    # ---- Dataset ---------------------------------------------------------
    #: Formato do corpus: ``'brsd'`` (arquivos ``{n}.wav`` numerados) ou ``'vctk'``
    #: (uma pasta por locutor, com FLACs por microfone).
    dataset_format: str = 'brsd'
    #: Diretório dos áudios do BrSD.
    audio_path: Path = Path('data/brsd/utterances')
    #: Raiz do corpus VCTK (``wav48_silence_trimmed``).
    vctk_root: Path | None = None
    #: Trilha de microfone do VCTK a utilizar (``mic1`` ou ``mic2``).
    vctk_mic: str = 'mic1'
    #: Número de locutores a considerar.
    num_speakers: int = 80
    #: Número máximo de enunciados por locutor.
    num_utterances: int = 5

    # ---- Pré-processamento ----------------------------------------------
    #: Taxa de amostragem em que os áudios são lidos.
    source_sampling_rate: int = 48_000
    #: Taxa de amostragem alvo após filtragem e decimação.
    target_sampling_rate: int = 8_000
    #: Se verdadeiro, remove trechos de não-voz antes da extração de features.
    enable_vad: bool = False
    #: Limiar da detecção de atividade vocal, em dB abaixo do pico.
    vad_top_db: int = 30
    #: Coeficiente do filtro de pré-ênfase; ``0.0`` desativa a pré-ênfase.
    pre_emphasis_coef: float = 0.97
    #: Número de coeficientes cepstrais extraídos por quadro.
    num_mfccs: int = 40
    #: Tamanho da janela de análise, em amostras. O salto é metade disso (50%).
    frame_size: int = 256

    # ---- Ajuste de features ---------------------------------------------
    #: Número de partições da validação cruzada.
    num_folds: int = 5
    #: Limita quantas partições são efetivamente executadas (0 = todas).
    max_folds: int = 0
    #: Teto para o número de quadros após padding (0 = sem teto). Trava de memória.
    max_frames_cap: int = 0
    #: Semente do sorteio determinístico do conjunto de validação.
    validation_seed: int = 42

    # ---- Protocolos de diagnóstico ---------------------------------------
    #: Executa apenas o pré-processamento e encerra.
    preprocess_only: bool = False
    #: Ativa o protocolo cross-mic (treina em um microfone, avalia em outro).
    cross_mic: bool = False
    #: Diretório de features do microfone de treino, no protocolo cross-mic.
    features_path_train: Path | None = None
    #: Diretório de features do microfone de teste, no protocolo cross-mic.
    features_path_test: Path | None = None
    #: Enunciados por locutor reservados para validação no protocolo cross-mic.
    cross_mic_validation_per_speaker: int = 10
    #: Ativa o protocolo multi-microfone (treina com todos, particionando por enunciado).
    both_mics: bool = False
    #: Diretórios de features a combinar no protocolo multi-microfone.
    features_paths: list[Path] = field(default_factory=list)

    # ---- Treinamento -----------------------------------------------------
    #: Arquiteturas a treinar, em ordem.
    architectures: tuple[str, ...] = ('cnn', 'attention', 'temporal_cnn')
    #: Número máximo de épocas (o early-stopping normalmente encerra antes).
    epochs: int = 1000
    #: Tamanho do lote.
    batch_size: int = 64
    #: Taxa de aprendizado inicial do otimizador Adam.
    learning_rate: float = 1e-3
    #: Épocas sem melhora na validação antes de interromper o treino.
    early_stopping_patience: int = 30

    # ---- Saídas ----------------------------------------------------------
    #: Diretório onde as features extraídas são persistidas.
    features_path: Path = Path('runs/features')
    #: Diretório onde modelos, métricas e figuras são gravados.
    models_path: Path = Path('runs/models')
    #: Gera figuras de diagnóstico para todos os áudios (custoso).
    enable_plots: bool = False
    #: Gera o conjunto completo de figuras apenas para os N primeiros áudios.
    num_plot_examples: int = 0

    @property
    def hop_size(self) -> int:
        """Salto entre quadros consecutivos, em amostras (50% de sobreposição)."""
        return self.frame_size // 2

    @property
    def frame_duration_ms(self) -> float:
        """Duração da janela de análise, em milissegundos."""
        return 1000.0 * self.frame_size / self.target_sampling_rate

    @property
    def chance_level(self) -> float:
        """Acurácia esperada de um classificador aleatório, em porcentagem."""
        return 100.0 / self.num_speakers

    @property
    def effective_folds(self) -> int:
        """Número de partições que serão efetivamente executadas."""
        if self.max_folds <= 0:
            return self.num_folds
        return min(self.num_folds, self.max_folds)

    def describe(self) -> str:
        """Resumo de uma linha por parâmetro, para registro no log do experimento."""
        lines = [f'Experimento: {self.experiment_name}']
        for f in fields(self):
            if f.name == 'experiment_name':
                continue
            lines.append(f'  {f.name} = {getattr(self, f.name)}')
        return '\n'.join(lines)


#: Conversores por campo, para os tipos que não são simplesmente ``str``.
_CONVERTERS = {
    'audio_path': lambda v: Path(v).expanduser(),
    'vctk_root': lambda v: Path(v).expanduser(),
    'features_path': lambda v: Path(v).expanduser(),
    'features_path_train': lambda v: Path(v).expanduser(),
    'features_path_test': lambda v: Path(v).expanduser(),
    'models_path': lambda v: Path(v).expanduser(),
    'features_paths': _as_path_list,
    'architectures': lambda v: tuple(a.strip() for a in str(v).split(',') if a.strip()),
    'enable_vad': _as_bool,
    'preprocess_only': _as_bool,
    'cross_mic': _as_bool,
    'both_mics': _as_bool,
    'enable_plots': _as_bool,
}


def _convert(name: str, raw: str, declared_type: Any) -> Any:
    """Converte um valor textual do perfil para o tipo declarado no dataclass.

    Com ``from __future__ import annotations`` as anotações chegam como strings,
    então a comparação é feita sobre o nome do tipo e não sobre o próprio objeto.
    """
    if name in _CONVERTERS:
        return _CONVERTERS[name](raw)
    type_name = declared_type if isinstance(declared_type, str) else getattr(
        declared_type, '__name__', str(declared_type)
    )
    if type_name == 'int':
        return int(raw)
    if type_name == 'float':
        return float(raw)
    if type_name == 'bool':
        return _as_bool(raw)
    return raw


def load_settings(config_path: str | os.PathLike | None = None) -> Settings:
    """Carrega o perfil de experimento e devolve as configurações validadas.

    A precedência é: argumento explícito → variável ``SR_CONFIG`` → ``DEFAULT_CONFIG``.
    Valores presentes no ambiente do processo sobrescrevem os do arquivo, o que
    permite variações pontuais sem editar o perfil::

        NUM_MFCCS=13 SR_CONFIG=configs/vctk.env python experiments/run_experiment.py

    Args:
        config_path: Caminho do perfil. Se omitido, usa ``SR_CONFIG``.

    Returns:
        Estrutura imutável com todos os parâmetros do experimento.

    Raises:
        FileNotFoundError: Se o perfil indicado não existir.
        ValueError: Se algum valor não puder ser convertido para o tipo esperado.
    """
    path = Path(config_path or os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG))
    if not path.is_file():
        raise FileNotFoundError(
            f'Perfil de configuração não encontrado: {path}. '
            f'Defina {CONFIG_ENV_VAR} apontando para um arquivo em configs/.'
        )

    raw: dict[str, str] = {k: v for k, v in dotenv_values(path).items() if v is not None}
    # O ambiente do processo tem precedência sobre o arquivo.
    for key in list(raw):
        if key in os.environ:
            raw[key] = os.environ[key]

    kwargs: dict[str, Any] = {}
    for f in fields(Settings):
        key = f.name.upper()
        if key not in raw:
            continue
        try:
            kwargs[f.name] = _convert(f.name, raw[key], f.type)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Valor inválido para {key} em {path}: {raw[key]!r}') from exc

    settings = Settings(**kwargs)
    _validate(settings)
    return settings


def _validate(settings: Settings) -> None:
    """Verifica invariantes que, se violadas, produziriam resultados sem sentido.

    Falhar aqui é preferível a treinar por horas e só então descobrir que o
    experimento estava mal especificado.
    """
    if settings.num_speakers < 2:
        raise ValueError('num_speakers deve ser ao menos 2 para haver classificação.')
    if settings.num_folds < 2:
        raise ValueError('num_folds deve ser ao menos 2.')
    if settings.frame_size <= 0 or settings.frame_size & (settings.frame_size - 1):
        raise ValueError('frame_size deve ser uma potência de dois (exigência da FFT).')
    if settings.target_sampling_rate > settings.source_sampling_rate:
        raise ValueError(
            'target_sampling_rate não pode exceder source_sampling_rate: '
            'reamostragem para cima não cria informação.'
        )
    if settings.dataset_format not in ('brsd', 'vctk'):
        raise ValueError(f'dataset_format desconhecido: {settings.dataset_format!r}')
    if settings.dataset_format == 'vctk' and settings.vctk_root is None:
        raise ValueError('dataset_format=vctk exige VCTK_ROOT.')
    if settings.cross_mic and not (settings.features_path_train and settings.features_path_test):
        raise ValueError('cross_mic exige FEATURES_PATH_TRAIN e FEATURES_PATH_TEST.')
    if settings.both_mics and len(settings.features_paths) < 2:
        raise ValueError('both_mics exige ao menos dois diretórios em FEATURES_PATHS.')
