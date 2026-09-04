#!/usr/bin/env python3
"""Verifica, pelos dados, que as duas trilhas de microfone estão alinhadas.

O protocolo cross-microfone afirma comparar a mesma voz dizendo a mesma frase no
mesmo instante, com apenas o transdutor mudando. Toda a interpretação dos seus
resultados repousa nessa afirmação, que o manifesto da ingestão documenta mas não
demonstra — um manifesto consistente consigo mesmo continuaria consistente se a
regra que o gerou estivesse errada.

Aqui a afirmação é testada contra o áudio. Se o índice ``(locutor, enunciado)``
designa a mesma gravação nas duas trilhas, os dois sinais são o mesmo evento
acústico captado por dois microfones, e as suas matrizes de coeficientes têm de ser
muito mais parecidas entre si do que com as de qualquer outra gravação.

O critério é de **recuperação**, e não de limiar. Para cada par, pergunta-se se a
gravação correspondente na outra trilha é a mais parecida entre ela e um conjunto de
distratores. Um limiar exigiria escolher um número defensável para "parecido o
bastante"; a recuperação não exige nenhum, e tem acaso conhecido: ``1/(K+1)``.

A verificação lê o áudio **bruto**, sem detecção de atividade vocal. Não é detalhe:
o VAD é aplicado independentemente a cada trilha, e como os dois microfones têm
níveis e ruído de fundo distintos, ele corta em pontos diferentes e dessincroniza os
sinais. O relatório mede esse efeito separadamente, porque ele tem consequência
própria para a leitura do protocolo.

Uso::

    python experiments/verify_alignment.py
    python experiments/verify_alignment.py --samples 200 --distractors 40
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import logging
import re
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from sr.config import load_settings  # noqa: E402

logger = logging.getLogger('verify_alignment')

MEMBER_PATTERN = re.compile(
    r'wav48_silence_trimmed/(?P<speaker>[ps]\d+)/'
    r'(?P=speaker)_(?P<utterance>\d+)_(?P<mic>mic\d)\.flac$'
)

MFCC_FILENAME = 'mfccs.npy'

#: Taxa de recuperação abaixo da qual o alinhamento não se considera estabelecido.
#:
#: Exige-se perfeição porque o teste é fácil quando os dados estão certos: os dois
#: sinais são o mesmo evento acústico. Qualquer falha aponta para um par mal formado,
#: e não para dificuldade intrínseca da tarefa.
REQUIRED_RETRIEVAL = 1.0


def scan(zip_path: Path) -> dict[str, dict[str, dict[str, str]]]:
    """Cataloga os membros de áudio do zip por locutor, enunciado e trilha."""
    catalog: dict[str, dict[str, dict[str, str]]] = collections.defaultdict(
        lambda: collections.defaultdict(dict))
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            match = MEMBER_PATTERN.search(name)
            if match:
                catalog[match['speaker']][match['utterance']][match['mic']] = name
    return catalog


def mfccs_from_member(archive: zipfile.ZipFile, member: str, num_mfccs: int) -> np.ndarray:
    """Decodifica um membro do zip e extrai a sua matriz de coeficientes.

    Usa os parâmetros padrão do librosa, e não os do perfil do experimento: aqui não
    se busca reproduzir a cadeia de processamento, e sim comparar dois sinais entre
    si. O que importa é que os dois lados recebam tratamento idêntico.
    """
    import librosa
    import soundfile as sf

    with archive.open(member) as source:
        audio, sampling_rate = sf.read(io.BytesIO(source.read()), dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=num_mfccs)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Correlação entre duas matrizes, após remover o perfil médio de cada coeficiente.

    A centragem por coeficiente é o que dá poder ao teste. Sem ela, a correlação fica
    dominada pelo perfil estático do espectro da fala, que é semelhante em qualquer
    gravação — e mede-se 0,90 entre dois locutores diferentes, o que não distingue
    nada. Removido esse perfil, resta a dinâmica temporal, que é específica da
    gravação.

    Args:
        a: Primeira matriz, ``(coeficientes, quadros)``.
        b: Segunda matriz.

    Returns:
        Correlação entre ``-1`` e ``1``.
    """
    frames = min(a.shape[1], b.shape[1])
    if frames < 2:
        return float('nan')

    x = a[:, :frames].astype(np.float64)
    y = b[:, :frames].astype(np.float64)
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)
    y = (y - y.mean(axis=1, keepdims=True)) / (y.std(axis=1, keepdims=True) + 1e-8)
    return float(np.corrcoef(x.ravel(), y.ravel())[0, 1])


def measure_vad_divergence(mic1: Path, mic2: Path) -> dict[str, float]:
    """Mede quanto o VAD dessincronizou as trilhas, nas features já processadas.

    As duas trilhas vêm da mesma gravação e deveriam ter a mesma duração. A detecção
    de atividade vocal é aplicada a cada uma isoladamente, com um limiar relativo ao
    pico de cada sinal; como os microfones diferem em nível e em ruído de fundo, os
    cortes caem em pontos distintos.

    Returns:
        Estatísticas da divergência relativa de duração entre as trilhas.
    """
    divergences: list[float] = []
    identical = 0

    speakers = sorted({d.name for d in mic1.iterdir() if d.name.isdigit()} &
                      {d.name for d in mic2.iterdir() if d.name.isdigit()})
    for speaker in speakers:
        for utterance_dir in sorted((mic1 / speaker).iterdir()):
            a = utterance_dir / MFCC_FILENAME
            b = mic2 / speaker / utterance_dir.name / MFCC_FILENAME
            if not (a.exists() and b.exists()):
                continue
            frames_a = int(np.load(a, mmap_mode='r').shape[1])
            frames_b = int(np.load(b, mmap_mode='r').shape[1])
            identical += frames_a == frames_b
            divergences.append(abs(frames_a - frames_b) / max(frames_a, frames_b, 1))

    if not divergences:
        return {}
    values = np.asarray(divergences)
    return {
        'pares': len(values),
        'duracao_identica': identical,
        'divergencia_mediana': float(np.median(values)),
        'divergencia_media': float(values.mean()),
        'divergencia_maxima': float(values.max()),
    }


def run_retrieval(
    zip_path: Path,
    catalog: dict[str, dict[str, dict[str, str]]],
    speakers: list[str],
    samples: int,
    distractors: int,
    num_mfccs: int,
    seed: int,
) -> dict[str, object]:
    """Executa o teste de recuperação entre as duas trilhas.

    Args:
        zip_path: Caminho do zip do corpus.
        catalog: Catálogo dos membros de áudio.
        speakers: Nomes dos locutores elegíveis.
        samples: Número de pares casados a testar.
        distractors: Tamanho do conjunto de distratores.
        num_mfccs: Número de coeficientes.
        seed: Semente do sorteio.

    Returns:
        Resultado do teste, com as correlações medidas.
    """
    rng = np.random.default_rng(seed)

    def pares_completos(speaker: str) -> list[str]:
        return sorted(u for u, tracks in catalog[speaker].items()
                      if 'mic1' in tracks and 'mic2' in tracks)

    matched: list[float] = []
    against: list[float] = []
    hits = 0
    tested = 0

    with zipfile.ZipFile(zip_path) as archive:
        # Conjunto de distratores decodificado uma única vez: sem isso, o teste
        # decodificaria o mesmo áudio dezenas de vezes.
        pool: list[tuple[tuple[str, str], np.ndarray]] = []
        while len(pool) < distractors:
            speaker = str(rng.choice(speakers))
            utterances = pares_completos(speaker)
            if not utterances:
                continue
            utterance = str(rng.choice(utterances))
            pool.append(((speaker, utterance),
                         mfccs_from_member(archive, catalog[speaker][utterance]['mic2'], num_mfccs)))
        logger.info('Conjunto de %d distratores decodificado.', len(pool))

        while tested < samples:
            speaker = str(rng.choice(speakers))
            utterances = pares_completos(speaker)
            if not utterances:
                continue
            utterance = str(rng.choice(utterances))

            a = mfccs_from_member(archive, catalog[speaker][utterance]['mic1'], num_mfccs)
            b = mfccs_from_member(archive, catalog[speaker][utterance]['mic2'], num_mfccs)

            alvo = similarity(a, b)
            rivais = [similarity(a, matriz) for chave, matriz in pool
                      if chave != (speaker, utterance)]
            if not rivais:
                continue

            matched.append(alvo)
            against.extend(rivais)
            hits += alvo > max(rivais)
            tested += 1

            if tested % 25 == 0:
                logger.info('  %d/%d pares testados.', tested, samples)

    return {
        'pares': tested,
        'distratores': len(pool),
        'acertos': hits,
        'taxa': hits / tested if tested else float('nan'),
        'acaso': 1.0 / (len(pool) + 1),
        'casados': matched,
        'distratores_valores': against,
    }


def report(retrieval: dict[str, object], vad: dict[str, float], output: Path) -> bool:
    """Grava o relatório e decide se o alinhamento está estabelecido."""
    output.mkdir(parents=True, exist_ok=True)

    casados = np.asarray(retrieval['casados'], dtype=float)
    rivais = np.asarray(retrieval['distratores_valores'], dtype=float)
    taxa = float(retrieval['taxa'])
    separado = bool(casados.min() > rivais.max())
    passou = taxa >= REQUIRED_RETRIEVAL and separado

    linhas = [
        'Verificação de alinhamento entre trilhas de microfone',
        '(áudio bruto, sem detecção de atividade vocal)',
        '',
        f'Pares testados       : {retrieval["pares"]}',
        f'Distratores por par  : {retrieval["distratores"]}',
        f'Recuperação top-1    : {retrieval["acertos"]}/{retrieval["pares"]} = {taxa * 100:.1f}%',
        f'Acaso                : {float(retrieval["acaso"]) * 100:.1f}%',
        '',
        f'Correlação casados   : média={casados.mean():.4f}  mínimo={casados.min():.4f}',
        f'Correlação rivais    : média={rivais.mean():.4f}  máximo={rivais.max():.4f}',
        f'Separação completa   : {"sim" if separado else "NÃO — as distribuições se tocam"}',
        '',
        f'Veredito: {"ALINHADO" if passou else "NÃO ESTABELECIDO"}',
    ]

    if vad:
        linhas += [
            '',
            'Efeito da detecção de atividade vocal sobre a sincronia',
            '-------------------------------------------------------',
            f'Pares com duração idêntica: {vad["duracao_identica"]}/{vad["pares"]}',
            f'Divergência de duração    : mediana={vad["divergencia_mediana"] * 100:.1f}%  '
            f'máxima={vad["divergencia_maxima"] * 100:.1f}%',
            '',
            'O VAD é aplicado a cada trilha isoladamente, com limiar relativo ao pico',
            'daquele sinal. Como os microfones diferem em nível e ruído de fundo, os',
            'cortes caem em pontos distintos e as trilhas deixam de ser quadro a quadro',
            'correspondentes. O alinhamento de rótulo permanece correto — é a mesma frase',
            'da mesma pessoa —, mas a afirmação de que "apenas o transdutor muda" deixa de',
            'ser estritamente verdadeira depois desta etapa.',
        ]

    texto = '\n'.join(linhas)
    (output / 'alinhamento.txt').write_text(texto + '\n', encoding='utf-8')
    (output / 'alinhamento.json').write_text(json.dumps({
        'recuperacao': {k: v for k, v in retrieval.items()
                        if k not in ('casados', 'distratores_valores')},
        'correlacao_casados': {'media': float(casados.mean()), 'minimo': float(casados.min())},
        'correlacao_rivais': {'media': float(rivais.mean()), 'maximo': float(rivais.max())},
        'separacao_completa': separado,
        'alinhado': passou,
        'efeito_vad': vad,
    }, indent=2, ensure_ascii=False), encoding='utf-8')

    plt.figure(figsize=(8, 5))
    plt.hist(rivais, bins=50, alpha=0.6, label=f'distratores (n={len(rivais)})', color='steelblue')
    plt.hist(casados, bins=25, alpha=0.75, label=f'par casado (n={len(casados)})', color='crimson')
    plt.xlabel('Correlação entre as matrizes de MFCC das duas trilhas')
    plt.ylabel('Pares')
    plt.title(f'Alinhamento entre microfones — recuperação top-1 = {taxa * 100:.1f}%')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output / 'alinhamento.png', dpi=150)
    plt.close()

    print(texto)
    logger.info('Relatório gravado em %s', output)
    return passou


def main() -> int:
    """Ponto de entrada.

    Returns:
        ``0`` se o alinhamento estiver estabelecido, ``2`` caso contrário.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', '-c', default='configs/vctk.env')
    parser.add_argument('--zip', dest='zip_path',
                        default='/home/lsmsqt/datasets/vctk/VCTK-Corpus-0.92.zip')
    parser.add_argument('--features-root', default=None)
    parser.add_argument('--samples', type=int, default=100, help='Pares casados a testar.')
    parser.add_argument('--distractors', type=int, default=20, help='Tamanho do conjunto rival.')
    parser.add_argument('--speakers', type=int, default=40,
                        help='Locutores dos quais amostrar.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')

    settings = load_settings(args.config)
    zip_path = Path(args.zip_path).expanduser()
    if not zip_path.is_file():
        logger.error('Zip não encontrado: %s. A verificação exige o áudio original.', zip_path)
        return 1

    root = Path(args.features_root).expanduser() if args.features_root \
        else settings.features_path.parent

    catalog = scan(zip_path)
    manifest_file = root / 'vctk_manifesto.json'
    if manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        nomes = [manifest['locutores'][k] for k in sorted(manifest['locutores'], key=int)]
        logger.info('Usando a numeração do manifesto: %d locutores.', len(nomes))
    else:
        nomes = sorted(catalog)
        logger.warning('Manifesto ausente; amostrando de todos os %d locutores.', len(nomes))

    retrieval = run_retrieval(zip_path, catalog, nomes[:args.speakers], args.samples,
                              args.distractors, settings.num_mfccs, args.seed)

    vad: dict[str, float] = {}
    mic1, mic2 = root / 'vctk_mic1', root / 'vctk_mic2'
    if mic1.is_dir() and mic2.is_dir():
        vad = measure_vad_divergence(mic1, mic2)

    passou = report(retrieval, vad, settings.models_path.parent / 'verificacao_alinhamento')
    return 0 if passou else 2


if __name__ == '__main__':
    raise SystemExit(main())
