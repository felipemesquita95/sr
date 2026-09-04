"""Testes da numeração canônica da ingestão do VCTK.

Esta é a lógica que sustenta o protocolo cross-microfone. Ele afirma comparar a
mesma voz dizendo a mesma frase no mesmo instante, com apenas o transdutor mudando.
Se o índice ``(locutor, enunciado)`` designar pessoas ou frases diferentes em cada
trilha, o protocolo passa a medir outra coisa — e, pior, continua produzindo um
número plausível, apenas mais baixo.

O modo de falha é concreto: a numeração posicional de ``sr.datasets.index`` deriva
os índices da listagem de cada trilha, separadamente. Um locutor sem ``mic2``
desloca em um todos os locutores seguintes de uma trilha em relação à outra.
"""

from __future__ import annotations

from ingest_vctk import MEMBER_PATTERN, build_plan

MICS = ('mic1', 'mic2')

#: Catálogo com as duas lacunas que quebram a numeração posicional: um locutor sem
#: a segunda trilha, e um enunciado presente em apenas uma delas.
CATALOG = {
    'p225': {'001': {'mic1': 'a', 'mic2': 'b'},
             '002': {'mic1': 'c', 'mic2': 'd'},
             '003': {'mic1': 'e', 'mic2': 'f'}},
    'p226': {'001': {'mic1': 'g', 'mic2': 'h'},
             '002': {'mic1': 'i', 'mic2': 'j'},
             '003': {'mic1': 'k'}},
    'p227': {'001': {'mic1': 'l'},
             '002': {'mic1': 'm'}},
    'p228': {'001': {'mic1': 'n', 'mic2': 'o'},
             '002': {'mic1': 'p', 'mic2': 'q'},
             '003': {'mic1': 'r', 'mic2': 's'}},
}


def plan(num_speakers: int = 10, num_utterances: int = 10):
    return build_plan(CATALOG, MICS, num_speakers, num_utterances)


def test_speaker_without_every_track_is_dropped():
    """Locutor que não tem todas as trilhas sai do plano, e não é renumerado em silêncio."""
    result = plan()

    assert result.dropped_speakers == ['p227']
    assert 'p227' not in dict(result.speakers).values()


def test_numbering_does_not_shift_between_tracks():
    """O locutor após a lacuna recebe o mesmo índice nas duas trilhas.

    É o teste central. Com numeração posicional por trilha, ``mic1`` enxergaria
    ``[p225, p226, p227, p228]`` e ``mic2`` enxergaria ``[p225, p226, p228]``: p228
    seria o locutor 4 em uma trilha e o 3 na outra, e o cross-mic compararia duas
    pessoas diferentes sem emitir aviso algum.
    """
    result = plan()

    assert dict(result.speakers) == {1: 'p225', 2: 'p226', 3: 'p228'}


def test_utterance_present_in_a_single_track_is_dropped():
    """Enunciado que só existe em uma trilha sai, senão as frases deslocam dentro do locutor."""
    result = plan()

    assert [u for _, u in result.utterances['p226']] == ['001', '002']
    assert result.dropped_utterances == {'p226': 1}


def test_utterance_indices_are_dense_and_start_at_one():
    """Os índices são densos mesmo quando há buracos na numeração original do corpus."""
    result = plan()

    assert [i for i, _ in result.utterances['p226']] == [1, 2]
    assert [i for i, _ in result.utterances['p228']] == [1, 2, 3]


def test_caps_are_respected():
    """Os limites de locutores e enunciados do perfil são aplicados após a interseção."""
    result = plan(num_speakers=2, num_utterances=2)

    assert dict(result.speakers) == {1: 'p225', 2: 'p226'}
    assert [u for _, u in result.utterances['p225']] == ['001', '002']


def test_total_counts_every_track():
    """O total conta cada trilha, porque cada uma vira um arquivo de features."""
    # p225 três, p226 dois, p228 três = oito enunciados, em duas trilhas.
    assert plan().total_recordings() == 16


def test_single_track_keeps_the_incomplete_speaker():
    """Pedindo uma trilha só, o locutor que só a tem deixa de ser excluído.

    Confirma que a exclusão vem da interseção pedida, e não de uma lista fixa.
    """
    result = build_plan(CATALOG, ('mic1',), 10, 10)

    assert result.dropped_speakers == []
    assert dict(result.speakers) == {1: 'p225', 2: 'p226', 3: 'p227', 4: 'p228'}


# ----------------------------------------------------------------------
# Reconhecimento dos membros do zip
# ----------------------------------------------------------------------

def test_pattern_reads_speaker_utterance_and_mic():
    match = MEMBER_PATTERN.search(
        'VCTK-Corpus-0.92/wav48_silence_trimmed/p225/p225_003_mic2.flac')

    assert match is not None
    assert match.group('speaker') == 'p225'
    assert match.group('utterance') == '003'
    assert match.group('mic') == 'mic2'


def test_pattern_accepts_the_s_prefixed_speaker():
    """O VCTK tem um locutor fora da convenção ``pNNN`` (``s5``)."""
    assert MEMBER_PATTERN.search('wav48_silence_trimmed/s5/s5_001_mic1.flac') is not None


def test_pattern_rejects_a_file_lodged_in_another_speakers_directory():
    """O nome do arquivo tem de concordar com o diretório que o contém.

    A retrorreferência do padrão existe para isso: um arquivo mal alojado seria
    atribuído ao locutor errado, criando um rótulo incorreto que nenhuma etapa
    posterior teria como detectar.
    """
    assert MEMBER_PATTERN.search('wav48_silence_trimmed/p225/p226_001_mic1.flac') is None


def test_pattern_ignores_non_audio_members():
    assert MEMBER_PATTERN.search('VCTK-Corpus-0.92/txt/p225/p225_001.txt') is None
    assert MEMBER_PATTERN.search('VCTK-Corpus-0.92/speaker-info.txt') is None
