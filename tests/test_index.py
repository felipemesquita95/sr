"""Testes da indexação do VCTK, com foco na numeração compartilhada entre trilhas.

O defeito que estes testes impedem de voltar é o mais caro do repositório, porque
não produz erro: numerando cada trilha isoladamente, um locutor sem ``mic2`` desloca
todos os seguintes, e o protocolo cross-microfone passa a comparar pessoas
diferentes. O experimento roda até o fim e devolve uma acurácia mais baixa, que se lê
como resultado.

No VCTK 0.92 isso não é hipotético: ``p280`` e ``p315`` não possuem ``mic2``, e estão
nas posições 54 e 84 de 110.
"""

from __future__ import annotations

import dataclasses

import pytest

from sr.config import Settings
from sr.datasets.index import build_index


@pytest.fixture
def corpus(tmp_path):
    """Corpus mínimo que reproduz a lacuna do VCTK real.

    ``p002`` não tem ``mic2``, e está no meio da ordenação — que é o que torna o
    deslocamento observável. ``p003`` tem um enunciado só em ``mic1``.
    """
    layout = {
        'p001': {'001': ('mic1', 'mic2'), '002': ('mic1', 'mic2')},
        'p002': {'001': ('mic1',), '002': ('mic1',)},
        'p003': {'001': ('mic1', 'mic2'), '002': ('mic1',), '003': ('mic1', 'mic2')},
        'p004': {'001': ('mic1', 'mic2'), '002': ('mic1', 'mic2')},
    }
    root = tmp_path / 'wav48_silence_trimmed'
    for speaker, utterances in layout.items():
        (root / speaker).mkdir(parents=True)
        for utterance, mics in utterances.items():
            for mic in mics:
                (root / speaker / f'{speaker}_{utterance}_{mic}.flac').touch()
    return root


def settings_for(root, mic: str, mics: tuple[str, ...] = ('mic1', 'mic2')) -> Settings:
    return Settings(dataset_format='vctk', vctk_root=root, vctk_mic=mic, vctk_mics=mics,
                    num_speakers=10, num_utterances=10)


def test_speaker_indices_agree_between_tracks(corpus):
    """O índice de locutor designa a mesma pessoa nas duas trilhas.

    É o teste central. Sem a interseção, ``mic1`` numeraria
    ``[p001, p002, p003, p004]`` e ``mic2`` numeraria ``[p001, p003, p004]``, de modo
    que o locutor 3 seria ``p003`` em uma trilha e ``p004`` na outra.
    """
    de_mic1 = {r.speaker: r.path.parent.name for r in build_index(settings_for(corpus, 'mic1'))}
    de_mic2 = {r.speaker: r.path.parent.name for r in build_index(settings_for(corpus, 'mic2'))}

    assert de_mic1 == de_mic2
    assert de_mic1 == {1: 'p001', 2: 'p003', 3: 'p004'}


def test_speaker_without_every_track_is_omitted(corpus):
    """``p002`` sai da numeração por não ter todas as trilhas declaradas."""
    nomes = {r.path.parent.name for r in build_index(settings_for(corpus, 'mic1'))}

    assert 'p002' not in nomes


def test_utterance_indices_agree_between_tracks(corpus):
    """O índice de enunciado designa a mesma frase nas duas trilhas.

    ``p003`` tem o enunciado ``002`` apenas em ``mic1``. Sem a interseção, o índice 2
    seria a frase ``002`` numa trilha e a ``003`` na outra — mesma pessoa, frases
    distintas, que é a versão sutil do mesmo defeito.
    """
    def mapa(mic):
        return {(r.speaker, r.utterance): r.path.name.split('_')[1]
                for r in build_index(settings_for(corpus, mic))}

    assert mapa('mic1') == mapa('mic2')


def test_partial_utterance_is_dropped(corpus):
    """O enunciado presente em uma só trilha não entra."""
    identificadores = {r.path.name.split('_')[1]
                       for r in build_index(settings_for(corpus, 'mic1'))
                       if r.path.parent.name == 'p003'}

    assert identificadores == {'001', '003'}


def test_single_track_keeps_every_speaker(corpus):
    """Declarando uma trilha só, ninguém é excluído por falta de outra."""
    indice = build_index(settings_for(corpus, 'mic1', mics=('mic1',)))

    assert {r.path.parent.name for r in indice} == {'p001', 'p002', 'p003', 'p004'}


def test_paths_point_at_the_current_track(corpus):
    """Os caminhos devolvidos são os da trilha sendo processada, não os da numeração."""
    for recording in build_index(settings_for(corpus, 'mic2')):
        assert recording.path.name.endswith('_mic2.flac')
        assert recording.path.exists()


def test_current_track_must_belong_to_the_declared_set(corpus):
    """Processar uma trilha fora do conjunto declarado é incoerente, e falha."""
    with pytest.raises(ValueError, match='não está em vctk_mics'):
        build_index(settings_for(corpus, 'mic3', mics=('mic1', 'mic2')))


def test_caps_apply_after_the_intersection(corpus):
    """Os limites do perfil contam locutores já filtrados, não os do diretório."""
    indice = build_index(dataclasses.replace(settings_for(corpus, 'mic1'), num_speakers=2))

    assert {r.path.parent.name for r in indice} == {'p001', 'p003'}
