"""Acesso aos artefatos e reconstrução das divisões para inspeção.

A leitura independe de widgets e do Keras para ser testável sem janela.
A inspeção reutiliza as regras de particionamento do sistema; manter uma
segunda implementação poderia mostrar uma divisão diferente da usada no treino.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from sr.config import load_settings
from sr.features import FeatureAdjustmentSubsystem

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / 'runs'
FIGURES = ('sinal_original.png', 'espectro_original.png', 'espectro_filtrado.png',
           'espectro_reamostrado.png', 'espectro_preenfase.png', 'mfccs.png')


def numeric_directories(path: Path) -> list[Path]:
    """Restringe o índice aos identificadores numéricos dos artefatos.

    Pastas auxiliares como ``_resumo`` não representam locutores ou enunciados.

    Args:
        path: Diretório cujos filhos serão examinados.

    Returns:
        Subdiretórios em ordem numérica; lista vazia se a raiz não existir.
    """
    if not path.is_dir():
        return []
    return sorted((p for p in path.iterdir() if p.is_dir() and p.name.isdigit()),
                  key=lambda p: int(p.name))


def tracks(root: Path = RUNS / 'features') -> list[Path]:
    """Descobre trilhas disponíveis sem depender de uma lista de perfis fixa.

    Args:
        root: Raiz dos diretórios de features.

    Returns:
        Pastas não auxiliares com ao menos um subdiretório numérico.
    """
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and not p.name.startswith('_') and numeric_directories(p))


def figure(path: Path, name: str) -> Path | None:
    """Representa a ausência de figura como dado para o aviso da página.

    Args:
        path: Diretório da gravação.
        name: Nome do PNG persistido.

    Returns:
        Caminho existente ou ``None``; não tenta reconstruir o sinal.
    """
    target = path / name
    return target if target.is_file() else None


def version(path: Path) -> tuple[int, int] | None:
    """Obtém uma assinatura barata para invalidar entradas de cache.

    A chave por ``(mtime_ns, tamanho)`` detecta mudanças usuais sem ler o
    conteúdo. Não é um hash: alterações que preservem ambos exigem atualização
    explícita pelo usuário.

    Args:
        path: Artefato a consultar.

    Returns:
        Par de data de modificação e tamanho, ou ``None`` se ausente.
    """
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except FileNotFoundError:
        return None


@lru_cache(maxsize=256)
def _json(path: Path, stamp) -> dict:
    """Mantém a versão do arquivo na chave do cache de leitura.

    Args:
        path: JSON persistido.
        stamp: Assinatura de versão; participa da chave sem entrar na leitura.

    Returns:
        Conteúdo decodificado do artefato.
    """
    return json.loads(path.read_text(encoding='utf-8'))


def read_json(path: Path, *, live: bool = False) -> dict:
    """Tolera artefatos ausentes ou em gravação sem interromper a inspeção.

    O progresso deve ser solicitado com ``live=True``: ele é escrito durante
    o treino e fica fora do cache. Uma leitura parcial retorna ausência de dados
    para que a próxima consulta do timer possa tentar novamente.

    Args:
        path: Caminho do JSON.
        live: Ignora o cache para arquivos acompanhados durante o treino.

    Returns:
        Conteúdo do JSON, ou dicionário vazio se a leitura ou decodificação falhar.
    """
    try:
        if live:
            return json.loads(path.read_text(encoding='utf-8'))
        return _json(path, version(path))
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=32)
def _mfcc(path: Path, stamp) -> np.ndarray:
    result = np.load(path, allow_pickle=False)
    if result.ndim != 2 or not all(result.shape):
        raise ValueError(f'Matriz MFCC inválida: {path}')
    return result


def mfcc(path: Path) -> np.ndarray:
    """Reutiliza matrizes entre páginas sem conservar versões antigas do arquivo.

    Args:
        path: Arquivo ``mfccs.npy`` da gravação.

    Returns:
        Matriz de coeficientes por quadro, compartilhada pelo cache; não a altere.

    Raises:
        OSError: Se o arquivo não puder ser lido.
        ValueError: Se o conteúdo não for uma matriz bidimensional não vazia.
    """
    return _mfcc(path, version(path))


@lru_cache(maxsize=32)
def _signatures(path: Path, stamp) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in ('silence', 'speech', 'full') if name in archive}


def signatures(path: Path) -> dict[str, np.ndarray]:
    """Preserva a cobertura real das condições de assinatura de canal.

    Condições ausentes não recebem curvas artificiais; a página pode informar
    exatamente o que não foi persistido.

    Args:
        path: Arquivo ``assinaturas.npz``.

    Returns:
        Condições presentes entre ``silence``, ``speech`` e ``full``, ou vazio se ausente.
    """
    return _signatures(path, version(path)) if path.is_file() else {}


def manifest_maps(manifest: dict) -> tuple[dict[int, str], dict[str, int]]:
    """Exige correspondência unívoca entre índice interno e nome do locutor.

    Args:
        manifest: Manifesto com o campo ``locutores`` indexado por números textuais.

    Returns:
        Mapas índice → nome e nome → índice.

    Raises:
        ValueError: Se os índices não forem inteiros ou houver nomes duplicados.
    """
    forward = {int(k): v for k, v in manifest.get('locutores', {}).items()}
    reverse = {v: k for k, v in forward.items()}
    if len(reverse) != len(forward):
        raise ValueError('Manifesto contém nomes de locutores duplicados.')
    return forward, reverse


@lru_cache(maxsize=12)
def inventory(track: Path, require_vad: bool = False) -> dict[int, dict[int, bool]]:
    """Distingue gravações inspecionáveis daquelas com todas as figuras iniciais.

    Só enunciados com MFCC entram na navegação. O cache do inventário não
    acompanha versões de cada diretório; a atualização explícita o invalida.

    Args:
        track: Raiz da trilha.
        require_vad: Exige também ``sinal_vad.png`` para considerar as figuras completas.

    Returns:
        Mapa locutor → enunciado → presença do conjunto de figuras.
    """
    result = {}
    for speaker in numeric_directories(track):
        samples = {}
        for utterance in numeric_directories(speaker):
            names = {p.name for p in utterance.iterdir()}
            if 'mfccs.npy' in names:
                required = set(FIGURES) | ({'sinal_vad.png'} if require_vad else set())
                samples[int(utterance.name)] = required.issubset(names)
        if samples:
            result[int(speaker.name)] = samples
    return result


def profiles() -> dict[Path, object]:
    """Carrega os perfis locais com a mesma precedência usada pelo experimento.

    Sobrescritas do ambiente também se aplicam à inspeção dos perfis.

    Returns:
        Mapa do caminho de cada arquivo ``configs/*.env`` para suas configurações.

    Raises:
        ValueError: Se algum perfil contiver uma configuração inválida.
    """
    return {p: load_settings(p) for p in sorted((ROOT / 'configs').glob('*.env'))}


def track_profile(track: Path, available: dict):
    # A comparação de nomes também permite inspecionar cópias locais do repositório.
    """Prefere o perfil de uma trilha isolada ao inferir seus parâmetros.

    A comparação pelo nome permite inspecionar cópias do repositório em outro
    caminho. Na falta de um perfil isolado, aceita um perfil que use a trilha
    como origem ou destino do cruzamento.

    Args:
        track: Diretório da trilha inspecionada.
        available: Mapa de caminhos de perfil para configurações.

    Returns:
        Par caminho/configuração escolhido, ou ``(None, None)``.
    """
    for path, settings in available.items():
        if settings.features_path.name == track.name and not settings.cross_mic and not settings.both_mics:
            return path, settings
    for path, settings in available.items():
        sources = [settings.features_path, settings.features_path_train, settings.features_path_test]
        if any(p and p.name == track.name for p in sources):
            return path, settings
    return None, None


def metrics(root: Path = RUNS / 'models') -> tuple[list[dict], list[str]]:
    """Separa resultados utilizáveis dos relatórios incompletos.

    Os diretórios fornecem a procedência que permite agrupar resultados por
    experimento, arquitetura e partição sem inventar medições ausentes.

    Args:
        root: Raiz de modelos e relatórios.

    Returns:
        Par de registros com procedência e caminhos de métricas ilegíveis ou incompletas.
    """
    rows, errors = [], []
    for path in sorted(root.glob('*/*/particao*/metricas.json')):
        data = read_json(path)
        required = ('acuracia', 'f1_macro', 'acaso', 'num_classes', 'num_amostras_teste')
        if not all(k in data for k in required):
            errors.append(str(path))
            continue
        rows.append(dict(data, experimento=path.parents[2].name,
                         arquitetura=path.parents[1].name, particao=path.parent.name,
                         diretorio=str(path.parent)))
    return rows, errors


@dataclass(frozen=True)
class FeatureRef:
    """Descreve uma gravação sem manter sua matriz inteira na memória.

    A referência substitui a matriz durante a inspeção das divisões reais.

    Attributes:
        path: Arquivo MFCC.
        speaker: Locutor em base um.
        utterance: Enunciado em base um.
        shape: Número de coeficientes e quadros.
    """
    path: Path
    speaker: int
    utterance: int
    shape: tuple[int, int]


@lru_cache(maxsize=12)
def references(path: Path, speakers: int, utterances: int) -> dict:
    """Inspeciona formas dos MFCCs sem carregar o corpus inteiro na memória.

    O mapeamento de memória permite consultar apenas o cabeçalho. Assim como
    o inventário, este cache depende de atualização explícita após nova ingestão.

    Args:
        path: Raiz da trilha.
        speakers: Maior índice de locutor incluído.
        utterances: Maior índice de enunciado incluído por locutor.

    Returns:
        Referências indexadas por ``(locutor, enunciado)``.

    Raises:
        ValueError: Se alguma matriz tiver forma inválida.
    """
    result = {}
    for speaker, samples in inventory(path).items():
        if speaker > speakers:
            continue
        for utterance in samples:
            if utterance > utterances:
                continue
            file = path / str(speaker) / str(utterance) / 'mfccs.npy'
            array = np.load(file, mmap_mode='r', allow_pickle=False)
            shape = array.shape
            del array
            if len(shape) != 2 or not all(shape):
                raise ValueError(f'Matriz MFCC inválida: {file}')
            result[speaker, utterance] = FeatureRef(file, speaker, utterance, shape)
    return result


class _InspectSplit(FeatureAdjustmentSubsystem):
    """Reutiliza as regras de divisão sem materializar os tensores.

    A substituição da carga por referências e da finalização por listas mantém
    a inspeção alinhada ao sistema com memória proporcional ao índice.

    Args:
        settings: Configuração que determina o protocolo e a divisão.
    """
    def load_features(self, features_path=None):
        """Fornece metadados no lugar das matrizes para o particionador real.

        Args:
            features_path: Trilha alternativa; se omitida, usa a configurada.

        Returns:
            Referências indexadas por locutor e enunciado.
        """
        return references(features_path or self.settings.features_path,
                          self.settings.num_speakers, self.settings.num_utterances)

    def _finalize(self, train, validation, test, label):
        if not train:
            raise ValueError('Sem gravações de treino para o perfil escolhido.')
        return {'Treino': [r for r, _ in train], 'Validação': [r for r, _ in validation],
                'Teste': [r for r, _ in test]}


def inspect_split(settings, fold: int) -> tuple[dict, int]:
    """Reconstrói os destinos das gravações segundo o protocolo configurado.

    O comprimento comum vem apenas do treino, como na preparação dos tensores;
    usar o máximo de todos os conjuntos esconderia uma via de vazamento.

    Args:
        settings: Perfil atual; não representa um snapshot de execuções antigas.
        fold: Partição em base um; não se aplica ao protocolo cross-microfone.

    Returns:
        Par de referências por conjunto e comprimento comum limitado pelo teto.

    Raises:
        ValueError: Se as features não permitirem formar os conjuntos exigidos.
    """
    subsystem = _InspectSplit(settings)
    if settings.cross_mic:
        split = subsystem.prepare_cross_microphone()
    elif settings.both_mics:
        split = subsystem.prepare_multi_microphone(fold)
    else:
        split = subsystem.prepare_fold(fold)
    maximum = max(r.shape[1] for r in split['Treino'])
    frames = min(maximum, settings.max_frames_cap) if settings.max_frames_cap > 0 else maximum
    return split, frames


def normalization(train: list[FeatureRef], frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Calcula estatísticas do treino com memória de uma gravação por vez.

    Duas passagens permitem obter média e variância após o alinhamento real
    sem empilhar todos os tensores. Validação e teste não participam do cálculo.

    Args:
        train: Referências não vazias do conjunto de treino.
        frames: Comprimento comum determinado pelo treino.

    Returns:
        Média e desvio por coeficiente em float32, com o estabilizador usado pelo sistema.
    """
    mean = np.zeros(train[0].shape[0], dtype=np.float64)
    count = len(train) * frames
    for ref in train:
        x = FeatureAdjustmentSubsystem.pad_or_truncate(np.load(ref.path), frames)
        mean += x.sum(axis=1, dtype=np.float64)
    mean /= count
    variance = np.zeros_like(mean)
    for ref in train:
        x = FeatureAdjustmentSubsystem.pad_or_truncate(np.load(ref.path), frames)
        variance += ((x - mean[:, None]) ** 2).sum(axis=1)
    return mean.astype(np.float32), (np.sqrt(variance / count) + 1e-8).astype(np.float32)


def clear_cache():
    """Permite enxergar uma nova ingestão sem reabrir a janela.

    Além das leituras versionadas, descarta inventários e referências cujas
    chaves não incluem a versão de todos os arquivos da trilha.
    """
    for function in (_json, _mfcc, _signatures, inventory, references):
        function.cache_clear()
