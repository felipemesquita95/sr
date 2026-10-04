"""Garantias da inspeção: procedência, partições e controle do subprocesso."""
from __future__ import annotations

import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from sr.config import Settings, load_settings
from sr.features import FeatureAdjustmentSubsystem
from ui import dados


@pytest.fixture
def corpus_tree(tmp_path):
    dados.clear_cache()
    for track in ('vctk_mic1', 'vctk_mic2'):
        for speaker in (1, 2):
            for utterance in range(1, 7):
                sample = tmp_path / 'features' / track / str(speaker) / str(utterance)
                sample.mkdir(parents=True)
                np.save(sample / 'mfccs.npy', np.full((4, 10 + utterance), speaker * 10 + utterance, dtype=np.float32))
        (tmp_path / 'features' / track / '_resumo').mkdir()
    return tmp_path


def config(root):
    return Settings(num_speakers=2, num_utterances=6, num_folds=3, max_frames_cap=14,
                    features_path=root / 'features/vctk_mic1',
                    features_path_train=root / 'features/vctk_mic1',
                    features_path_test=root / 'features/vctk_mic2',
                    features_paths=[root / 'features/vctk_mic1', root / 'features/vctk_mic2'])


def test_inventory_excludes_summary_and_reports_missing_figures(corpus_tree):
    track = corpus_tree / 'features/vctk_mic1'
    assert list(dados.inventory(track)) == [1, 2]
    assert not dados.inventory(track)[1][6]
    assert dados.figure(track / '1/6', 'sinal_original.png') is None
    assert dados.tracks(corpus_tree / 'absent') == []


def test_complete_sample_requires_vad_figure_when_enabled(corpus_tree):
    track = corpus_tree / 'features/vctk_mic1'
    for name in dados.FIGURES:
        (track / '1/1' / name).touch()
    assert dados.inventory(track)[1][1]
    assert not dados.inventory(track, require_vad=True)[1][1]


def test_manifest_is_bidirectional_and_rejects_collisions():
    forward, reverse = dados.manifest_maps({'locutores': {'1': 'p225', '2': 'p226'}})
    assert all(reverse[name] == index for index, name in forward.items())
    with pytest.raises(ValueError, match='duplicados'):
        dados.manifest_maps({'locutores': {'1': 'p225', '2': 'p225'}})


def test_missing_signature_condition_is_not_fabricated(tmp_path):
    path = tmp_path / 'assinaturas.npz'
    np.savez(path, speech=np.ones(8))
    assert set(dados.signatures(path)) == {'speech'}
    assert dados.signatures(tmp_path / 'missing.npz') == {}


def test_diagnostic_artifacts_keep_the_value_and_their_provenance(tmp_path):
    report = tmp_path / 'models/matriz9_exemplo/matriz_transferencia.json'
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({'ajustes': [{'perda_acuracia_pp': 7.25}]}))
    loaded = dados.diagnosticos(tmp_path / 'models')
    path, value = loaded['matrizes'][0]
    assert path == report
    assert value['ajustes'][0]['perda_acuracia_pp'] == json.loads(report.read_text())['ajustes'][0]['perda_acuracia_pp']


def test_missing_diagnostic_artifact_is_reported_without_exception(tmp_path, qt_app):
    from PySide6.QtWidgets import QLabel
    from ui.evidencias import EvidencePage
    page = EvidencePage('evidencias', 'Evidências', 'Teste')
    payload = {'diagnosticos': dados.diagnosticos(tmp_path / 'models'),
               'resultados': '', 'limitacoes': ''}
    page.display(payload, None)
    labels = [item.text() for item in page.findChildren(QLabel)]
    assert any('Artefato ausente' in text for text in labels)


def test_file_cache_invalidation_and_live_progress(tmp_path):
    path = tmp_path / 'progresso.json'
    path.write_text('{"epoca": 1}')
    assert dados.read_json(path)['epoca'] == 1
    path.write_text('{"epoca": 222}')
    assert dados.read_json(path)['epoca'] == 222
    assert dados.read_json(path, live=True)['epoca'] == 222
    path.write_text('{')
    assert dados.read_json(path, live=True) == {}


@pytest.mark.parametrize('multi', [False, True])
def test_partition_is_disjoint_and_groups_microphones(corpus_tree, multi):
    split, frames = dados.inspect_split(replace(config(corpus_tree), both_mics=multi), 1)
    keys = [{(r.speaker, r.utterance) for r in refs} for refs in split.values()]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    assert len(set.union(*keys)) == 12
    assert frames == min(14, max(r.shape[1] for r in split['Treino']))
    if multi:
        assert all(len(refs) == 2 * len({(r.speaker, r.utterance) for r in refs}) for refs in split.values())


def test_cross_microphone_preserves_actual_shared_utterances(corpus_tree):
    settings = replace(config(corpus_tree), cross_mic=True, cross_mic_validation_per_speaker=1)
    split, _ = dados.inspect_split(settings, 1)
    assert [len(split[k]) for k in ('Treino', 'Validação', 'Teste')] == [10, 2, 12]
    assert all(r.path.parents[2].name == 'vctk_mic2' for r in split['Teste'])
    assert {(r.speaker, r.utterance) for r in split['Treino']} <= {
        (r.speaker, r.utterance) for r in split['Teste']}


def test_normalization_matches_training_only(corpus_tree):
    settings = config(corpus_tree)
    split, frames = dados.inspect_split(settings, 1)
    mean, std = dados.normalization(split['Treino'], frames)
    expected = np.stack([FeatureAdjustmentSubsystem.pad_or_truncate(np.load(r.path), frames)
                         for r in split['Treino']])
    np.testing.assert_allclose(mean, expected.mean(axis=(0, 2)), rtol=1e-6)
    np.testing.assert_allclose(std, expected.std(axis=(0, 2)) + 1e-8, rtol=1e-6)
    actual = FeatureAdjustmentSubsystem(settings).prepare_fold(1)
    np.testing.assert_allclose(actual.train_x, (expected - mean[None, :, None]) / std[None, :, None], atol=1e-6)


@pytest.fixture(scope='session')
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def wait_until(predicate, timeout=10000):
    deadline = time.monotonic() + timeout / 1000
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(10)
        time.sleep(.001)  # Libera também o GIL para as leituras Python dos workers.
    assert predicate(), 'A operação assíncrona não terminou.'


@pytest.fixture
def window(corpus_tree, monkeypatch, qt_app):
    from ui.app import MainWindow
    monkeypatch.setattr(dados, 'RUNS', corpus_tree)
    settings = replace(config(corpus_tree), models_path=corpus_tree / 'models/test')
    widget = MainWindow(profiles={corpus_tree / 'profile.env': settings}, persist=False)
    widget.show()
    wait_until(lambda: widget.selection is not None and widget.pages[0].loaded_key is not None)
    yield widget
    widget.close()
    wait_until(lambda: not widget.tasks.pending)
    widget.deleteLater()
    QTest.qWait(10)


def test_ui_empty_directory(tmp_path, monkeypatch, qt_app):
    from ui.app import MainWindow
    monkeypatch.setattr(dados, 'RUNS', tmp_path)
    widget = MainWindow(profiles={}, persist=False)
    assert 'Nenhuma trilha' in widget.pages[0].message.text()
    widget.close()
    widget.deleteLater()


def test_ui_navigation_and_absence_are_visible(window):
    from ui.conteudo import ROTEIRO
    window.speaker.setCurrentIndex(window.speaker.findData(2))
    window.utterance.setCurrentIndex(window.utterance.findData(6))
    for index, (stage, _, _) in enumerate(ROTEIRO):
        page = window.page_by_stage[stage]
        window.navigation.setCurrentRow(index)
        wait_until(lambda: page.loaded_key is not None and page.loaded_key[0] == window.selection.key)
        assert not page.message.isVisible(), page.message.text()
        assert window.selection.speaker == 2
        assert window.selection.utterance == 6


def test_old_background_result_cannot_replace_new_selection(window, monkeypatch):
    import threading
    import ui.app as app_module
    real_load = app_module.load_stage
    old_started, release = threading.Event(), threading.Event()

    def delayed(stage, ctx, *args):
        if stage == 'passo_sinal' and ctx.utterance == 1:
            old_started.set()
            release.wait(timeout=5)
        return real_load(stage, ctx, *args)

    monkeypatch.setattr(app_module, 'load_stage', delayed)
    window.pages[0].loaded_key = None
    window.navigate_stage('sinal')
    wait_until(old_started.is_set)
    try:
        window.select_sample(1, 2)
        page = window.page_by_stage['sinal']
        wait_until(lambda: page.loaded_key is not None and page.loaded_key[0][2] == 2)
        expected = page.loaded_key
    finally:
        release.set()
    wait_until(lambda: not window.tasks.pending)
    assert page.loaded_key == expected


@pytest.mark.parametrize('query', ['2', '002', 'p226', '2 — p226'])
def test_typed_speaker_commits_actual_selection(window, query):
    from PySide6.QtCore import Qt
    window.speaker.setItemText(window.speaker.findData(2), '2 — p226')
    field = window.speaker.lineEdit()
    field.setFocus()
    field.selectAll()
    if query.isascii():
        QTest.keyClicks(field, query)
    else:
        from PySide6.QtGui import QInputMethodEvent
        event = QInputMethodEvent()
        event.setCommitString(query)
        QApplication.sendEvent(field, event)
    QTest.keyClick(field, Qt.Key.Key_Return)
    assert window.selection.speaker == 2
    assert window.speaker.currentData() == 2
    assert window.speaker.currentText() == '2 — p226'
    field = window.utterance.lineEdit()
    field.setFocus()
    field.selectAll()
    QTest.keyClicks(field, '006')
    QTest.keyClick(field, Qt.Key.Key_Return)
    assert window.selection.utterance == 6
    assert window.utterance.currentText().startswith('006 ·')


def test_selector_commits_on_focus_out_and_rejects_unknown_id(window):
    window.speaker.setItemText(window.speaker.findData(2), '2 — p226')
    window.activateWindow()
    field = window.speaker.lineEdit()
    field.setFocus()
    QTest.qWait(30)
    field.selectAll()
    QTest.keyClicks(field, '2')
    window.navigation.setFocus()
    QTest.qWait(30)
    assert window.selection.speaker == 2
    field.setFocus()
    field.selectAll()
    QTest.keyClicks(field, '999')
    window.navigation.setFocus()
    QTest.qWait(30)
    assert window.selection.speaker == 2
    assert window.speaker.currentText() == '2 — p226'
    assert 'não identifica' in window.statusBar().currentMessage()


def test_raw_page_can_open_existing_sample_without_changing_speaker(window):
    from PySide6.QtGui import QImage, QColor
    from ui.componentes import ImagePanel
    sample = window.selection.track / '2/3'
    for name in ('sinal_original.png', 'espectro_original.png'):
        picture = QImage(80, 40, QImage.Format.Format_RGB32)
        picture.fill(QColor('green'))
        assert picture.save(str(sample / name))
    # Há figuras desta etapa, mesmo sem o conjunto completo do pipeline.
    assert not window.inventory[2][3]
    window.select_sample(2, 6)
    window.navigate_stage('sinal')
    page = window.page_by_stage['sinal']
    wait_until(lambda: page.loaded_key and page.loaded_key[0] == window.selection.key)
    assert page.figure_samples.count() == 1
    assert page.figure_samples.itemData(0) == 3
    assert page.figure_samples.currentIndex() == -1
    page.figure_samples.setCurrentIndex(0)
    wait_until(lambda: page.loaded_key and page.loaded_key[0] == window.selection.key)
    assert (window.selection.speaker, window.selection.utterance) == (2, 3)
    assert page.body.findChildren(ImagePanel)[0].source == sample / 'sinal_original.png'


def test_changing_speaker_on_signal_chooses_available_example(window):
    window.inventory[2][3] = True
    window.select_sample(1, 6)
    window.navigate_stage('sinal')
    window.speaker.setCurrentIndex(window.speaker.findData(2))
    assert (window.selection.speaker, window.selection.utterance) == (2, 3)


def test_raw_signal_without_saved_png_uses_selected_brsd_audio(window, tmp_path):
    import soundfile as sf
    from ui.conteudo import load_stage
    audio = tmp_path / 'audio'
    audio.mkdir()
    rate = 8000
    t = np.arange(rate) / rate
    for index, frequency in [(7, 220), (8, 440)]:
        sf.write(audio / f'{index}.wav', .5 * np.sin(2 * np.pi * frequency * t), rate)
    settings = replace(window.selection.settings, audio_path=audio, dataset_format='brsd')
    for utterance, frequency in [(1, 220), (2, 440)]:
        ctx = replace(window.selection, speaker=2, utterance=utterance, settings=settings)
        payload = load_stage('sinal', ctx)
        assert payload['images'] == []
        raw = payload['raw']
        assert raw['source'] == audio / f'{6 + utterance}.wav'
        assert raw['rate'] == rate and raw['samples'] == rate and raw['duration'] == 1
        assert raw['frequency'][np.argmax(raw['magnitude'])] == frequency
        assert len(raw['time']) <= 6000
        assert raw['high'].max() > .49 and raw['low'].min() < -.49
        window.page_by_stage['sinal'].display(payload, ctx)
        assert not window.page_by_stage['sinal'].message.isVisible()


def test_vctk_audio_mapping_uses_manifest_not_dense_utterance_number(tmp_path):
    settings = Settings(dataset_format='vctk', vctk_root=tmp_path)
    manifest = {'locutores': {'2': 'p226'}, 'enunciados': {'p226': {'3': '017'}}}
    path = tmp_path / 'p226/p226_017_mic2.flac'
    path.parent.mkdir()
    path.touch()
    assert dados.audio_source(settings, 2, 3, manifest, 'vctk_mic2') == path
    assert dados.audio_source(settings, 2, 3, manifest, 'vctk_mic1') is None
    assert dados.audio_source(settings, 2, 3, {}, 'vctk_mic2') is None


def wheel(widget, delta=-120):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    position = widget.rect().center()
    event = QWheelEvent(QPointF(position), QPointF(widget.mapToGlobal(position)),
                        QPoint(), QPoint(0, delta), Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)
    QTest.qWait(20)


def test_result_partition_wheel_keeps_curves_and_sections_keep_scroll(window, corpus_tree):
    for fold in (1, 2):
        directory = corpus_tree / f'models/test/cnn/particao{fold}'
        directory.mkdir(parents=True)
        (directory / 'metricas.json').write_text(json.dumps({
            'acuracia': .5, 'f1_macro': .5, 'acaso': .5,
            'num_classes': 2, 'num_amostras_teste': 4}))
        (directory / 'progresso.json').write_text(json.dumps({
            'particao': fold, 'epoca': 2, 'historico': {'loss': [2, 1], 'accuracy': [.2, .5]}}))
    page = window.page_by_stage['resultados']
    window.navigate_stage('resultados')
    page.reload.emit()
    wait_until(lambda: page.loaded_key is not None)
    page.architecture.setCurrentText('cnn')
    page.sections.setCurrentIndex(1)
    wait_until(lambda: page.result_history.isVisible())
    number, canvas = page.detail_number, page.result_history.canvas
    wheel(page.chooser)
    assert page.chooser.currentIndex() == 0
    assert page.detail_number == number
    assert page.result_history.canvas is canvas
    assert page.result_history.isVisible()
    page.areas[1].verticalScrollBar().setValue(20)
    previous = page.areas[1].verticalScrollBar().value()
    page.sections.setCurrentIndex(0)
    page.sections.setCurrentIndex(1)
    assert page.areas[1].verticalScrollBar().value() == previous
    assert page.result_history.canvas is canvas


def test_barra_de_selecao_aparece_apenas_nos_primeiros_tres_passos(window):
    from ui.conteudo import ROTEIRO
    for indice, (etapa, _, _) in enumerate(ROTEIRO):
        window.navigate_stage(etapa)
        assert window.toolbar.isVisible() == (indice < 3)
    for etapa in ('corpus', 'modelo', 'tensores', 'assinatura'):
        window.navigate_stage(etapa)
        assert window.toolbar.isVisible()


def test_roteiro_alcanca_cada_evidencia(window):
    from ui.conteudo import ROTEIRO
    assert window.navigation.count() == len(ROTEIRO) == 4
    assert [titulo for _, titulo, _ in ROTEIRO] == ["Sinal", "MFCC", "Rede", "Resultado"]
    for index, (stage, _, _) in enumerate(ROTEIRO):
        window.navigation.setCurrentRow(index)
        page = window.page_by_stage[stage]
        wait_until(lambda: page.loaded_key is not None)
        assert window.stack.currentWidget() is page
    from ui.evidencias import EvidencePage
    assert len(window.findChildren(EvidencePage)) == 1
    for etapa, secao in EvidencePage.stage_sections.items():
        window.navigate_stage(etapa)
        page = window.page_by_stage[etapa]
        assert window.stack.currentWidget() is window.page_by_stage['resultado']
        assert window.page_by_stage['resultado'].sections.currentWidget() is page
        assert page.sections.currentIndex() == page.section_names.index(secao)
        assert page.isVisible()
        assert not window.toolbar.isVisible()


def test_comparacao_le_metricas_ordena_filtra_e_preserva_procedencia(window, monkeypatch):
    from PySide6.QtCore import QItemSelectionModel, Qt
    origens = [str(dados.RUNS / f'models/ensaio_{nome}/cnn/particao1') for nome in ('a', 'b', 'c')]
    registros = [dict(experimento=f'ensaio_{nome}', arquitetura='cnn', particao='particao1',
                      acuracia=acuracia, f1_macro=acuracia, acaso=.25, num_classes=4,
                      num_amostras_teste=quantidade, diretorio=origem,
                      arquivo=f'{origem}/metricas.json')
                 for nome, acuracia, quantidade, origem in zip(('a', 'b', 'c'), (.9, .75, .6), (100, 20, 3), origens)]
    chamadas = []

    def ler(raiz):
        chamadas.append(raiz)
        return registros, []

    monkeypatch.setattr(dados, 'metrics', ler)
    window.navigate_stage('comparacao')
    pagina = window.page_by_stage['comparacao']
    wait_until(lambda: pagina.loaded_key is not None)
    assert chamadas == [dados.RUNS / 'models']
    assert pagina.tabela.rowCount() == len(registros)
    assert window.page_by_stage['resultados'].records is pagina.registros
    coluna = pagina.campos.index('num_amostras_teste')
    pagina.tabela.sortItems(coluna, Qt.SortOrder.AscendingOrder)
    assert [pagina.tabela.item(linha, coluna).data(Qt.ItemDataRole.DisplayRole)
            for linha in range(pagina.tabela.rowCount())] == [3, 20, 100]
    selecao = pagina.tabela.selectionModel()
    for linha in range(pagina.tabela.rowCount()):
        selecao.select(pagina.tabela.model().index(linha, 0),
                       QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
    assert pagina.contraste.columnCount() == len(registros) + 1
    linha_origem = pagina.campos.index('diretorio')
    linha_acuracia = pagina.campos.index('acuracia')
    for coluna in range(1, pagina.contraste.columnCount()):
        origem = pagina.contraste.item(linha_origem, coluna).text()
        registro = next(r for r in registros if r['diretorio'] == origem)
        assert pagina.contraste.item(linha_acuracia, coluna).data(Qt.ItemDataRole.DisplayRole) == registro['acuracia']
        assert pagina.contraste.item(linha_acuracia, coluna).toolTip() == origem + '/metricas.json'
    pagina.filtro.setText('ensaio_b')
    assert sum(not pagina.tabela.isRowHidden(linha) for linha in range(pagina.tabela.rowCount())) == 1
    assert pagina.contraste.item(linha_origem, 1).text() == origens[1]
    pagina.filtro.clear()
    assert pagina.contraste.columnCount() == len(registros) + 1


def test_matriz_de_transferencia_entra_na_comparacao(tmp_path):
    raiz = tmp_path / 'models'
    (raiz / 'matriz9_teste').mkdir(parents=True)
    (raiz / 'matriz9_teste/configuracao.json').write_text(json.dumps(
        {'settings': {'architectures': ['attention']}}))
    (raiz / 'matriz9_teste/matriz_transferencia.json').write_text(json.dumps({
        'acaso': .25,
        'ajustes': [{'origem': 'mic1', 'semente': 7,
                     'tamanhos': {'treino': 8, 'validacao': 2, 'teste': 4},
                     'suporte_por_locutor': [1, 1, 1, 1],
                     'celulas': {'mic1': {'accuracy': .9, 'f1': .88},
                                 'mic2': {'accuracy': .4, 'f1': .35}}}]}))
    linhas = dados.transfer_rows(raiz)
    assert [linha['particao'] for linha in linhas] == ['mic1 → mic1 · semente 7', 'mic1 → mic2 · semente 7']
    travessia = linhas[1]
    assert travessia['acuracia'] == .4 and travessia['f1_macro'] == .35
    assert travessia['arquitetura'] == 'attention' and travessia['num_classes'] == 4
    assert travessia['num_amostras_teste'] == 4 and travessia['acaso'] == .25
    # A procedência precisa apontar o arquivo real, não um metricas.json inexistente.
    assert travessia['arquivo'] == str(raiz / 'matriz9_teste/matriz_transferencia.json')


def test_comparacao_distingue_linhas_do_mesmo_diretorio(window, monkeypatch):
    from PySide6.QtCore import QItemSelectionModel, Qt
    origem = str(dados.RUNS / 'models/matriz_ensaio')
    # A matriz de transferência emite várias linhas do mesmo diretório. A tabela
    # é ordenada ao ser montada, então localizar o registro pela linha exige uma
    # identidade que sobreviva à reordenação.
    registros = [dict(experimento='matriz_ensaio', arquitetura='attention',
                      particao=f'mic1 → {destino} · semente 7', acuracia=acuracia,
                      f1_macro=acuracia, acaso=.25, num_classes=4, num_amostras_teste=4,
                      diretorio=origem, arquivo=f'{origem}/matriz_transferencia.json')
                 for destino, acuracia in (('mic2', .4), ('mic1', .9))]
    monkeypatch.setattr(dados, 'metrics', lambda raiz: (registros, []))
    window.navigate_stage('comparacao')
    pagina = window.page_by_stage['comparacao']
    wait_until(lambda: pagina.loaded_key is not None)
    coluna = pagina.campos.index('acuracia')
    pagina.tabela.sortItems(coluna, Qt.SortOrder.AscendingOrder)
    selecao = pagina.tabela.selectionModel()
    for linha in range(pagina.tabela.rowCount()):
        selecao.select(pagina.tabela.model().index(linha, 0),
                       QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
    assert pagina.contraste.columnCount() == 3
    contrastadas = [pagina.contraste.item(pagina.campos.index('particao'), c).text() for c in (1, 2)]
    assert contrastadas == ['mic1 → mic2 · semente 7', 'mic1 → mic1 · semente 7']
    acuracias = [pagina.contraste.item(coluna, c).data(Qt.ItemDataRole.DisplayRole) for c in (1, 2)]
    assert acuracias == [.4, .9]
    assert pagina.tabela.item(0, 0).toolTip() == f'{origem}/matriz_transferencia.json'


def test_resultado_nao_abre_em_experimento_sem_metricas(window):
    pagina = window.page_by_stage['resultado'].componentes['resultados']
    registros = [dict(experimento='ensaio', arquitetura='cnn', particao='particao1',
                      acuracia=.5, f1_macro=.5, acaso=.25, num_classes=4,
                      num_amostras_teste=4, diretorio='d', arquivo='d/metricas.json')]
    pagina.display({'metrics': registros, 'errors': []}, window.selection)
    # A trilha selecionada não tem modelos treinados; a página precisa cair em
    # uma combinação que exista, em vez de abrir vazia na frente da banca.
    assert pagina.experiment.currentText() == 'ensaio'
    assert pagina.architecture.currentText() == 'cnn'
    assert pagina.classes.currentText() == '4'


def test_ressalvas_saem_sem_marcacao():
    from ui.evidencias import texto_legivel
    titulo, corpo = texto_legivel('## 9. O que este documento **não** afirma\n\n'
                                  '- Que a faixa seja `identidade` vocal.\n'
                                  '- Que a **barreira** seja do dado.\n'
                                  '- Nada sobre **sessões\n  diferentes**, que é a limitação.\n\n---')
    assert titulo == '9. O que este documento não afirma'
    # A ênfase atravessa a quebra de linha no documento real, e o separador
    # horizontal não deve virar um traço solto na tela.
    assert corpo == ('• Que a faixa seja identidade vocal.\n'
                     '• Que a barreira seja do dado.\n'
                     '• Nada sobre sessões\n  diferentes, que é a limitação.')


def test_resultado_carrega_diagnosticos_uma_vez(window, monkeypatch):
    from ui.conteudo import load_stage
    chamadas = []
    original = dados.diagnosticos

    def ler(raiz):
        chamadas.append(raiz)
        return original(raiz)

    monkeypatch.setattr(dados, 'diagnosticos', ler)
    conteudo = load_stage('resultado', window.selection)
    assert chamadas == [dados.RUNS / 'models']
    assert conteudo['evidencias'] is conteudo['limites']


def test_derivadas_ausentes_e_persistidas_sem_recalculo(window):
    from PySide6.QtWidgets import QLabel
    from ui.conteudo import load_stage
    ctx = window.selection
    pagina = window.page_by_stage['mfcc']
    conteudo = load_stage('mfcc', ctx)
    assert all(matriz is None for _, matriz in conteudo['derivadas'])
    pagina.display(conteudo, ctx)
    textos = '\n'.join(item.text() for item in pagina.body.findChildren(QLabel))
    for nome in ('delta.npy', 'delta_delta.npy'):
        caminho = ctx.sample / nome
        assert str(caminho) in textos
        assert not caminho.exists()
        np.save(caminho, np.full_like(conteudo['matrices'][0], ctx.utterance))
    conteudo = load_stage('mfcc', ctx)
    for caminho, matriz in conteudo['derivadas']:
        np.testing.assert_array_equal(matriz, np.load(caminho))
    pagina.display(conteudo, ctx)
    textos = '\n'.join(item.text() for item in pagina.body.findChildren(QLabel))
    assert 'não persistiu' not in textos
    assert all(str(caminho) in textos for caminho, _ in conteudo['derivadas'])


def test_parametros_vem_do_perfil_e_manifesto_da_amostra(window):
    from PySide6.QtWidgets import QLabel
    from ui.conteudo import CAMPOS, load_stage
    ctx = replace(window.selection, settings=replace(window.selection.settings,
                  source_sampling_rate=32000, target_sampling_rate=16000, frame_size=512,
                  num_mfccs=17, batch_size=19, learning_rate=.003, early_stopping_patience=7),
                  manifest={'locutores': {'1': 'p_teste'}, 'enunciados': {'p_teste': {'1': 'frase_original'}}})
    for etapa, campos in CAMPOS.items():
        pagina = window.page_by_stage[etapa]
        pagina.display(load_stage(etapa, ctx, fold=1), ctx)
        for campo in campos:
            assert f'{campo}: {getattr(ctx.settings, campo)}' in pagina.parametros.text()
        assert str(ctx.profile) in pagina.parametros.text()
    textos = '\n'.join(item.text() for item in window.page_by_stage['corpus'].body.findChildren(QLabel))
    assert 'p_teste' in textos and 'frase_original' in textos
    assert str(dados.RUNS / 'features/vctk_manifesto.json') in textos


def test_trilha_sem_perfil_nao_recebe_configuracao_padrao(window):
    from ui.conteudo import CAMPOS, load_stage
    ctx = replace(window.selection, profile=None, settings=None)
    assert dados.track_profile(ctx.track, {}) == (None, None)
    for etapa in CAMPOS:
        pagina = window.page_by_stage[etapa]
        pagina.display(load_stage(etapa, ctx, fold=1), ctx)
        assert 'Sem perfil associado' in pagina.parametros.text()
        assert str(dados.ROOT / 'configs') in pagina.parametros.text()
    assert 'Sem perfil associado' in window.page_by_stage['tensores'].message.text()


def test_preprocessamento_sem_vad_mantem_etapas_posteriores(window):
    from ui.conteudo import load_stage
    ctx = replace(window.selection, settings=replace(window.selection.settings, enable_vad=False))
    conteudo = load_stage('preprocessamento', ctx)
    nomes = [caminho.name for _, _, caminho in conteudo['images']]
    assert 'sinal_vad.png' not in nomes
    assert nomes == ['sinal_original.png', 'espectro_filtrado.png',
                     'espectro_reamostrado.png', 'espectro_preenfase.png']
