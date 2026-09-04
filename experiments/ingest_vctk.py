#!/usr/bin/env python3
"""Ingestão do VCTK: do arquivo zip às features, sem extrair o corpus inteiro.

O zip do VCTK 0.92 tem 11 GB e o corpus extraído tem porte semelhante. Manter os
dois ao mesmo tempo não cabe em uma máquina com 15 GB livres. Este script resolve
o problema processando **um locutor por vez**: extrai só os arquivos daquele
locutor, converte em MFCCs pela mesma cadeia do subsistema de pré-processamento, e
apaga o áudio antes de passar ao próximo. O pico de disco fica no zip mais um
locutor mais as features acumuladas.

Numeração canônica
------------------
O ponto delicado não é o espaço, é a numeração.

``sr.datasets.index`` numera locutores e enunciados pela **posição** na listagem do
diretório, filtrada pela trilha de microfone pedida. As duas trilhas são indexadas
de forma independente. Basta um locutor sem ``mic2`` para que todos os locutores
seguintes desloquem de um entre as duas numerações; basta um enunciado presente em
uma trilha e ausente na outra para que as frases desloquem dentro do locutor.

A consequência é silenciosa e grave: o protocolo cross-mic passaria a comparar
pessoas ou frases diferentes, em vez de "mesma voz, mesmo texto, mesmo instante,
apenas o transdutor muda". Não haveria erro nem aviso — apenas uma acurácia mais
baixa, indistinguível de um achado.

Este script elimina a possibilidade por construção. A numeração é derivada dos
**nomes dos arquivos** dentro do zip, restrita à interseção entre as trilhas
pedidas, e gravada em um manifesto JSON auditável. As duas trilhas recebem a mesma
numeração porque recebem exatamente a mesma lista.

Uso::

    # baixa o zip (retomável) e processa as duas trilhas
    python experiments/ingest_vctk.py --download

    # só o plano, sem processar nada: mostra o que seria feito
    python experiments/ingest_vctk.py --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from sr.config import Settings, load_settings  # noqa: E402
from sr.datasets import Recording  # noqa: E402
from sr.diagnostics import signatures_of  # noqa: E402
from sr.preprocessing import PreprocessingSubsystem  # noqa: E402

logger = logging.getLogger('ingest_vctk')

#: Nome do arquivo de assinaturas de canal gravado ao lado de cada matriz de MFCCs.
SIGNATURE_FILENAME = 'assinaturas.npz'

#: Endereço oficial do corpus, conforme o registro de download de 14/06/2026.
VCTK_URL = ('https://datashare.ed.ac.uk/bitstream/handle/10283/3443/'
            'VCTK-Corpus-0.92.zip')

#: Tamanho esperado do zip, em bytes, para conferência após o download.
VCTK_ZIP_BYTES = 11_747_302_977

#: Membros de áudio do zip: ``.../wav48_silence_trimmed/pNNN/pNNN_YYY_micK.flac``.
#: O identificador do locutor é repetido no nome do arquivo, e a retrorreferência
#: garante que não se case um arquivo alojado no diretório de outro locutor.
MEMBER_PATTERN = re.compile(
    r'wav48_silence_trimmed/(?P<speaker>[ps]\d+)/'
    r'(?P=speaker)_(?P<utterance>\d+)_(?P<mic>mic\d)\.flac$'
)

#: Margem de disco livre abaixo da qual a ingestão para, em GB.
DEFAULT_MIN_FREE_GB = 1.5


# ----------------------------------------------------------------------
# Aquisição do zip
# ----------------------------------------------------------------------

def download(zip_path: Path) -> None:
    """Baixa o corpus, retomando um download parcial se houver.

    Args:
        zip_path: Destino do arquivo.

    Raises:
        RuntimeError: Se o ``curl`` falhar.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info('Baixando %s para %s (retomável).', VCTK_URL, zip_path)

    result = subprocess.run(
        ['curl', '-L', '-C', '-', '--fail', '--retry', '5', '-o', str(zip_path), VCTK_URL])
    if result.returncode != 0:
        raise RuntimeError(
            f'curl terminou com código {result.returncode}. O download é retomável: '
            f'rode de novo para continuar de onde parou.')


def verify_zip(zip_path: Path) -> None:
    """Confere o tamanho do arquivo baixado contra o esperado.

    Um zip truncado abriria normalmente e só falharia na leitura de algum membro,
    horas depois de a ingestão começar.
    """
    size = zip_path.stat().st_size
    if size != VCTK_ZIP_BYTES:
        logger.warning(
            'Tamanho do zip é %d bytes, esperado %d. Se o download foi interrompido, '
            'rode com --download para retomá-lo.', size, VCTK_ZIP_BYTES)
    else:
        logger.info('Zip íntegro: %d bytes, como esperado.', size)


# ----------------------------------------------------------------------
# Catálogo e plano de ingestão
# ----------------------------------------------------------------------

def scan(zip_path: Path) -> dict[str, dict[str, dict[str, str]]]:
    """Lê o índice do zip e cataloga os arquivos de áudio por locutor e enunciado.

    Lê apenas o diretório central do zip, sem descomprimir nada.

    Args:
        zip_path: Caminho do zip do corpus.

    Returns:
        Mapeamento ``locutor -> enunciado -> microfone -> nome do membro``.
    """
    catalog: dict[str, dict[str, dict[str, str]]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()

    for name in names:
        match = MEMBER_PATTERN.search(name)
        if match:
            speaker = match.group('speaker')
            utterance = match.group('utterance')
            catalog.setdefault(speaker, {}).setdefault(utterance, {})[match.group('mic')] = name

    logger.info('Catálogo: %d locutores, %d arquivos de áudio no zip.',
                len(catalog), sum(len(m) for u in catalog.values() for m in u.values()))
    return catalog


@dataclasses.dataclass(frozen=True)
class Plan:
    """A numeração canônica e o que ela exclui.

    Attributes:
        mics: Trilhas de microfone incluídas.
        speakers: Pares ``(índice, nome)``, com índice base 1.
        utterances: Para cada nome de locutor, pares ``(índice, identificador)``.
        dropped_speakers: Locutores excluídos por não terem todas as trilhas.
        dropped_utterances: Contagem de enunciados excluídos por locutor.
    """

    mics: tuple[str, ...]
    speakers: list[tuple[int, str]]
    utterances: dict[str, list[tuple[int, str]]]
    dropped_speakers: list[str]
    dropped_utterances: dict[str, int]

    def total_recordings(self) -> int:
        """Número de gravações a processar, somando todas as trilhas."""
        return sum(len(self.utterances[name]) for _, name in self.speakers) * len(self.mics)


def build_plan(
    catalog: dict[str, dict[str, dict[str, str]]],
    mics: tuple[str, ...],
    num_speakers: int,
    num_utterances: int,
) -> Plan:
    """Deriva a numeração canônica, compartilhada por todas as trilhas.

    Só entram no plano os locutores que possuem **todas** as trilhas pedidas, e
    dentro deles apenas os enunciados presentes em todas elas. É essa interseção que
    garante que o índice ``(locutor, enunciado)`` designe a mesma pessoa dizendo a
    mesma frase em qualquer trilha — condição sem a qual o protocolo cross-mic não
    mede o que afirma medir.

    Args:
        catalog: Saída de :func:`scan`.
        mics: Trilhas a incluir.
        num_speakers: Número máximo de locutores.
        num_utterances: Número máximo de enunciados por locutor.

    Returns:
        O plano de ingestão.
    """
    complete: list[str] = []
    dropped_speakers: list[str] = []
    shared: dict[str, list[str]] = {}
    dropped_utterances: dict[str, int] = {}

    for speaker in sorted(catalog):
        by_utterance = catalog[speaker]
        present = sorted(u for u, tracks in by_utterance.items()
                         if all(mic in tracks for mic in mics))
        if not present:
            dropped_speakers.append(speaker)
            continue

        missing = len(by_utterance) - len(present)
        if missing:
            dropped_utterances[speaker] = missing

        complete.append(speaker)
        shared[speaker] = present[:num_utterances]

    selected = complete[:num_speakers]

    if dropped_speakers:
        logger.warning(
            '%d locutores sem todas as trilhas %s foram excluídos: %s',
            len(dropped_speakers), ','.join(mics), ', '.join(dropped_speakers))
    if dropped_utterances:
        total = sum(dropped_utterances.values())
        logger.warning(
            '%d enunciados presentes em apenas parte das trilhas foram excluídos '
            '(%d locutores afetados).', total, len(dropped_utterances))
    if len(complete) < num_speakers:
        logger.warning('Pedidos %d locutores, disponíveis %d.', num_speakers, len(complete))

    return Plan(
        mics=mics,
        speakers=[(index, name) for index, name in enumerate(selected, start=1)],
        utterances={name: [(i, u) for i, u in enumerate(shared[name], start=1)]
                    for name in selected},
        dropped_speakers=dropped_speakers,
        dropped_utterances=dropped_utterances,
    )


def write_manifest(plan: Plan, zip_path: Path, destination: Path) -> None:
    """Grava o manifesto que documenta a numeração aplicada.

    O manifesto é o que torna a numeração auditável depois do fato: sem ele, saber
    qual pessoa é o locutor 57 exigiria reproduzir a ordenação de cabeça.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({
        'zip': str(zip_path),
        'microfones': list(plan.mics),
        'num_locutores': len(plan.speakers),
        'locutores': {str(index): name for index, name in plan.speakers},
        'enunciados': {
            name: {str(index): utterance for index, utterance in plan.utterances[name]}
            for _, name in plan.speakers
        },
        'excluidos': {
            'locutores_sem_todas_as_trilhas': plan.dropped_speakers,
            'enunciados_parciais_por_locutor': plan.dropped_utterances,
        },
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    logger.info('Manifesto da numeração gravado em %s', destination)


# ----------------------------------------------------------------------
# Ingestão
# ----------------------------------------------------------------------

def free_gigabytes(path: Path) -> float:
    """Espaço livre na partição que contém ``path``, em GB."""
    return shutil.disk_usage(path).free / 1e9


def extract_speaker(
    archive: zipfile.ZipFile,
    catalog: dict[str, dict[str, dict[str, str]]],
    plan: Plan,
    speaker_name: str,
    workdir: Path,
) -> dict[str, list[Recording]]:
    """Extrai os arquivos de um locutor e devolve as gravações já numeradas.

    Args:
        archive: Zip aberto.
        catalog: Catálogo completo.
        plan: Numeração canônica.
        speaker_name: Nome do locutor no corpus, por exemplo ``'p225'``.
        workdir: Diretório temporário onde depositar o áudio.

    Returns:
        Mapeamento de trilha para a lista de gravações daquele locutor.
    """
    speaker_index = next(i for i, name in plan.speakers if name == speaker_name)
    workdir.mkdir(parents=True, exist_ok=True)
    by_mic: dict[str, list[Recording]] = {mic: [] for mic in plan.mics}

    for utterance_index, utterance in plan.utterances[speaker_name]:
        for mic in plan.mics:
            member = catalog[speaker_name][utterance][mic]
            target = workdir / f'{speaker_name}_{utterance}_{mic}.flac'
            with archive.open(member) as source, open(target, 'wb') as sink:
                shutil.copyfileobj(source, sink)
            by_mic[mic].append(Recording(speaker_index, utterance_index, target))

    return by_mic


def write_signatures(
    recordings: list[Recording],
    settings: Settings,
    features_path: Path,
) -> int:
    """Calcula e grava a assinatura de canal de cada gravação, sob todas as condições.

    Isto acontece aqui, e não no diagnóstico, por uma razão que não é de organização:
    o áudio do VCTK existe apenas enquanto o seu locutor está sendo processado. Uma
    medida que dependa da forma de onda ou é tomada agora, ou exigiria decodificar o
    corpus inteiro outra vez. O custo é desprezível — o arquivo tem alguns kilobytes
    contra as dezenas de kilobytes da matriz de coeficientes.

    Args:
        recordings: Gravações do locutor corrente, já numeradas.
        settings: Configuração, de onde vêm taxa, número de coeficientes e limiar.
        features_path: Raiz das features da trilha.

    Returns:
        Número de gravações para as quais alguma assinatura foi gravada.
    """
    import librosa

    written = 0
    for recording in recordings:
        destination = features_path / str(recording.speaker) / str(recording.utterance)
        if (destination / SIGNATURE_FILENAME).exists():
            continue

        audio, _ = librosa.load(recording.path, sr=settings.source_sampling_rate)
        signatures = signatures_of(
            audio, settings.source_sampling_rate, settings.num_mfccs, settings.vad_top_db)
        if not signatures:
            continue

        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / SIGNATURE_FILENAME, **signatures)
        written += 1

    return written


def ingest(
    zip_path: Path,
    settings: Settings,
    plan: Plan,
    catalog: dict[str, dict[str, dict[str, str]]],
    features_root: Path,
    workdir: Path,
    min_free_gb: float,
    with_signatures: bool = True,
    prefix: str = 'vctk',
) -> int:
    """Percorre o plano, processando e descartando o áudio locutor a locutor.

    Args:
        zip_path: Caminho do zip.
        settings: Configuração base, de onde vêm os parâmetros de DSP.
        plan: Numeração canônica.
        catalog: Catálogo completo.
        features_root: Raiz sob a qual criar ``vctk_<mic>``.
        workdir: Diretório temporário para o áudio extraído.
        min_free_gb: Piso de disco livre; abaixo dele a ingestão para.

    Returns:
        Número de gravações processadas.
    """
    subsystems = {
        mic: PreprocessingSubsystem(
            dataclasses.replace(settings, vctk_mic=mic,
                                features_path=features_root / f'{prefix}_{mic}'))
        for mic in plan.mics
    }

    total_processed = 0
    for position, (_, speaker_name) in enumerate(plan.speakers, start=1):
        free = free_gigabytes(features_root)
        if free < min_free_gb:
            logger.error(
                'Disco livre em %.1f GB, abaixo do piso de %.1f GB. Ingestão interrompida '
                'no locutor %s (%d de %d). O script é retomável: libere espaço e rode de novo.',
                free, min_free_gb, speaker_name, position, len(plan.speakers))
            break

        with zipfile.ZipFile(zip_path) as archive:
            by_mic = extract_speaker(archive, catalog, plan, speaker_name, workdir)

        try:
            signed = 0
            for mic, recordings in by_mic.items():
                processed, skipped = subsystems[mic].process(recordings)
                total_processed += processed
                if with_signatures:
                    signed += write_signatures(
                        recordings, settings, subsystems[mic].settings.features_path)
            logger.info('[%d/%d] %s: %d enunciados por trilha, %d assinaturas, %.1f GB livres.',
                        position, len(plan.speakers), speaker_name,
                        len(plan.utterances[speaker_name]), signed, free)
        finally:
            # O áudio existe apenas durante o processamento deste locutor.
            shutil.rmtree(workdir, ignore_errors=True)

    for mic, subsystem in subsystems.items():
        logger.info('Gerando figuras de resumo da trilha %s.', mic)
        subsystem._write_summary_figures()  # noqa: SLF001

    return total_processed


def main() -> int:
    """Ponto de entrada.

    Returns:
        ``0`` em caso de sucesso, ``1`` em caso de erro.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', '-c', default='configs/vctk.env',
                        help='Perfil de onde vêm os parâmetros de DSP.')
    parser.add_argument('--zip', dest='zip_path',
                        default='/home/lsmsqt/datasets/vctk/VCTK-Corpus-0.92.zip',
                        help='Caminho do zip do corpus.')
    parser.add_argument('--download', action='store_true',
                        help='Baixa o zip antes de ingerir, retomando se parcial.')
    parser.add_argument('--mics', default='mic1,mic2',
                        help='Trilhas a ingerir, separadas por vírgula.')
    parser.add_argument('--speakers', type=int, default=None,
                        help='Máximo de locutores (padrão: o do perfil).')
    parser.add_argument('--utterances', type=int, default=None,
                        help='Máximo de enunciados por locutor (padrão: o do perfil).')
    parser.add_argument('--features-root', default=None,
                        help='Raiz das features (padrão: o pai do FEATURES_PATH do perfil).')
    parser.add_argument('--workdir', default=None,
                        help='Diretório temporário do áudio (padrão: sob a raiz das features).')
    parser.add_argument('--min-free-gb', type=float, default=DEFAULT_MIN_FREE_GB,
                        help='Piso de disco livre, em GB.')
    parser.add_argument('--prefix', default='vctk',
                        help='Prefixo dos diretórios de features, um por trilha. Serve para '
                             'manter variantes lado a lado — por exemplo uma com detecção de '
                             'atividade vocal e outra sem, que só podem ser comparadas se '
                             'ambas existirem.')
    parser.add_argument('--no-signatures', action='store_true',
                        help='Não calcula as assinaturas de canal durante a ingestão. '
                             'Só use se não pretender rodar o diagnóstico: o áudio é '
                             'apagado, e refazê-las exige decodificar o corpus de novo.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Monta e grava o plano, sem extrair nem processar.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')

    try:
        settings = load_settings(args.config)
    except (FileNotFoundError, ValueError) as error:
        logger.error('%s', error)
        return 1

    zip_path = Path(args.zip_path).expanduser()
    if args.download:
        try:
            download(zip_path)
        except RuntimeError as error:
            logger.error('%s', error)
            return 1
    if not zip_path.is_file():
        logger.error('Zip não encontrado: %s. Use --download para baixá-lo.', zip_path)
        return 1
    verify_zip(zip_path)

    mics = tuple(m.strip() for m in args.mics.split(',') if m.strip())
    features_root = Path(args.features_root).expanduser() if args.features_root \
        else settings.features_path.parent
    workdir = Path(args.workdir).expanduser() if args.workdir \
        else features_root / '_extracao_vctk'

    catalog = scan(zip_path)
    if not catalog:
        logger.error('Nenhum arquivo de áudio reconhecido em %s. O zip é do VCTK 0.92?', zip_path)
        return 1

    plan = build_plan(
        catalog, mics,
        num_speakers=args.speakers or settings.num_speakers,
        num_utterances=args.utterances or settings.num_utterances,
    )
    if not plan.speakers:
        logger.error('Nenhum locutor possui todas as trilhas pedidas (%s).', ','.join(mics))
        return 1

    write_manifest(plan, zip_path, features_root / 'vctk_manifesto.json')
    logger.info('Plano: %d locutores, %d gravações, trilhas %s.',
                len(plan.speakers), plan.total_recordings(), ','.join(mics))

    if args.dry_run:
        logger.info('Execução de ensaio: nada foi extraído nem processado.')
        return 0

    logger.info('Disco livre antes de começar: %.1f GB.', free_gigabytes(features_root))
    processed = ingest(zip_path, settings, plan, catalog, features_root, workdir,
                       args.min_free_gb, with_signatures=not args.no_signatures,
                       prefix=args.prefix)
    logger.info('Ingestão concluída: %d gravações processadas, %.1f GB livres.',
                processed, free_gigabytes(features_root))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
