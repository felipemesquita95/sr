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
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtTest import QTest

from sr.config import Settings, load_settings
from sr.features import FeatureAdjustmentSubsystem
from ui import dados
from ui.execucao import TrainingManager, training_command


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


def test_training_command_preserves_environment_and_handles_spaces(tmp_path, monkeypatch):
    profile = tmp_path / 'perfil com espaços.env'
    profile.write_text('NUM_SPEAKERS=2\n')
    base = {'MODELS_PATH': '/tmp/output with spaces', 'VALIDATION_SEED': '9', 'PATH': os.environ['PATH']}
    command, env = training_command(profile, 'cnn', 2, 7, base)
    assert env['MODELS_PATH'] == base['MODELS_PATH']
    assert env['VALIDATION_SEED'] == '9'
    assert env['KERAS_BACKEND'] == 'torch'
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    settings = load_settings(profile)
    assert settings.architectures == ('cnn',)
    assert (settings.max_folds, settings.epochs) == (2, 7)
    assert settings.models_path == Path(base['MODELS_PATH'])


def test_training_is_nonblocking_and_can_be_stopped(tmp_path):
    manager = TrainingManager()
    command = [sys.executable, '-u', '-c', 'import time; print("started", flush=True); time.sleep(30)']
    job = manager.start(command, os.environ.copy(), tmp_path, 'test')
    try:
        assert job.process.poll() is None
        with pytest.raises(RuntimeError, match='ativo'):
            manager.start(command, os.environ.copy(), tmp_path, 'test')
        deadline = time.monotonic() + 5
        while 'started' not in job.lines() and time.monotonic() < deadline:
            time.sleep(.01)
        assert 'started' in job.lines()
        job.stop()
        job.process.wait(timeout=10)
        assert job.stopping
    finally:
        if job.process.poll() is None:
            job.process.kill()
            job.process.wait(timeout=5)


def test_ui_training_uses_features_without_reading_missing_audio(corpus_tree, monkeypatch):
    from types import SimpleNamespace
    import librosa
    import run_experiment
    from sr.system import SpeakerRecognitionSystem
    audio = corpus_tree / 'audio_ausente'
    features = corpus_tree / 'features/vctk_mic1'
    profile = corpus_tree / 'perfil.env'
    profile.write_text(f'DATASET_FORMAT=vctk\nVCTK_ROOT={audio}\n'
                       f'FEATURES_PATH={features}\nNUM_SPEAKERS=2\n'
                       'NUM_UTTERANCES=6\nNUM_FOLDS=3\nNUM_MFCCS=4\n')
    assert not audio.exists()
    assert (features / '1/1/mfccs.npy').is_file()
    command, env = training_command(profile, 'cnn', 1, 1, {})
    monkeypatch.setattr(os, 'environ', env)
    monkeypatch.setattr(sys, 'argv', command[1:])
    monkeypatch.setattr(librosa, 'load', lambda *args, **kwargs: pytest.fail('Áudio bruto acessado'))
    trained = []

    def train(self, architecture, split, fold):
        trained.append((architecture, split, fold))
        return SimpleNamespace(accuracy=.5, f1=.5)

    monkeypatch.setattr(SpeakerRecognitionSystem, '_train_and_evaluate', train)
    monkeypatch.setattr(SpeakerRecognitionSystem, '_write_summary', lambda *args: None)
    assert run_experiment.main() == 0
    assert len(trained) == 1
    architecture, split, fold = trained[0]
    assert (architecture, fold) == ('cnn', 1)
    assert split.train_x.shape[1] == 4
    assert len(split.train_x) + len(split.validation_x) + len(split.test_x) == 12


@pytest.mark.parametrize('source', ['profile', 'environment'])
def test_training_command_respects_preprocess_only(tmp_path, monkeypatch, source):
    profile = tmp_path / 'perfil.env'
    profile.write_text('PREPROCESS_ONLY=true\n' if source == 'profile' else '')
    base = {'PREPROCESS_ONLY': 'true'} if source == 'environment' else {}
    _, env = training_command(profile, 'cnn', 1, 1, base)
    monkeypatch.setattr(os, 'environ', env)
    assert load_settings().preprocess_only


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
    from PySide6.QtWidgets import QLabel
    window.speaker.setCurrentIndex(window.speaker.findData(2))
    window.utterance.setCurrentIndex(window.utterance.findData(6))
    for index, page in enumerate(window.pages):
        window.navigation.setCurrentRow(index)
        if page.stage != 'treino':
            wait_until(lambda: page.loaded_key is not None and page.loaded_key[0] == window.selection.key)
            assert not page.message.isVisible(), page.message.text()
        assert window.selection.speaker == 2
        assert window.selection.utterance == 6
        if page.stage == 'sinal':
            assert any('figura não disponível' in item.text() for item in page.findChildren(QLabel))
        if page.stage == 'assinatura':
            assert any('Assinaturas não disponíveis' in item.text() for item in page.findChildren(QLabel))


def test_ui_requires_confirmation_before_overwriting(window, monkeypatch):
    page = window.training
    output = next(iter(page.profiles.values())).models_path
    output.mkdir(parents=True)
    (output / 'previous.json').write_text('{}')
    monkeypatch.setattr(QMessageBox, 'question', lambda *args: QMessageBox.StandardButton.No)
    monkeypatch.setattr(page.manager, 'start', lambda *args: pytest.fail('Sobrescrita sem confirmação'))
    page.start()
    assert 'preservados' in page.status.text()


def test_old_background_result_cannot_replace_new_selection(window, monkeypatch):
    import threading
    import ui.app as app_module
    real_load = app_module.load_stage
    old_started, release = threading.Event(), threading.Event()

    def delayed(stage, ctx, *args):
        if stage == 'sinal' and ctx.utterance == 1:
            old_started.set()
            release.wait(timeout=5)
        return real_load(stage, ctx, *args)

    monkeypatch.setattr(app_module, 'load_stage', delayed)
    window.navigation.setCurrentRow(1)
    wait_until(old_started.is_set)
    try:
        window.step_sample(1)
        page = window.pages[1]
        wait_until(lambda: page.loaded_key is not None and page.loaded_key[0][2] == 2)
        expected = page.loaded_key
    finally:
        release.set()
    wait_until(lambda: not window.tasks.pending)
    assert page.loaded_key == expected


def test_complete_shortcut_and_navigation_reuse_loaded_page(window, corpus_tree):
    window.inventory[2][3] = True
    window.random_complete()
    assert (window.selection.speaker, window.selection.utterance) == (2, 3)
    wait_until(lambda: window.pages[0].loaded_key and window.pages[0].loaded_key[0] == window.selection.key)
    original_body = window.pages[0].body
    window.step_page(1)
    window.step_page(-1)
    assert window.pages[0].body is original_body


def test_arrow_clicks_change_actual_signal_and_cross_speakers(window):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage, QColor
    from ui.componentes import ImagePanel
    for speaker, utterance, color in [(1, 5, 'red'), (1, 6, 'green'), (2, 1, 'blue')]:
        sample = window.selection.track / str(speaker) / str(utterance)
        for name in ('sinal_original.png', 'espectro_original.png'):
            image = QImage(80, 40, QImage.Format.Format_RGB32)
            image.fill(QColor(color))
            assert image.save(str(sample / name))
    window.select_sample(1, 5)
    window.navigation.setCurrentRow(1)
    page = window.pages[1]
    for speaker, utterance, color in [(1, 5, 'red'), (1, 6, 'green'), (2, 1, 'blue')]:
        wait_until(lambda: page.loaded_key and page.loaded_key[0] == window.selection.key)
        assert (window.selection.speaker, window.selection.utterance) == (speaker, utterance)
        panels = page.body.findChildren(ImagePanel)
        assert len(panels) == 2
        assert panels[0].source == window.selection.sample / 'sinal_original.png'
        assert panels[0].image.pixelColor(0, 0) == QColor(color)
        if color != 'blue':
            QTest.mouseClick(window.next, Qt.MouseButton.LeftButton)
            assert not page.scroll.isVisible()  # Nunca deixa a imagem anterior na nova seleção.
    QTest.mouseClick(window.previous, Qt.MouseButton.LeftButton)
    assert (window.selection.speaker, window.selection.utterance) == (1, 6)
    # Vários cliques seguidos devem terminar na seleção mais recente, mesmo sem PNG.
    for _ in range(3):
        QTest.mouseClick(window.next, Qt.MouseButton.LeftButton)
    wait_until(lambda: page.loaded_key and page.loaded_key[0] == window.selection.key)
    assert (window.selection.speaker, window.selection.utterance) == (2, 3)
    assert not page.body.findChildren(ImagePanel)
    assert window.previous.isEnabled()


def test_keyboard_arrows_after_selector_choice_and_during_search(window):
    from PySide6.QtCore import Qt
    window.activateWindow()
    window.utterance.setFocus()
    QTest.qWait(30)
    QTest.keyClick(window.utterance, Qt.Key.Key_Right)
    assert window.selection.utterance == 2
    QTest.keyClick(window.utterance, Qt.Key.Key_Left)
    assert window.selection.utterance == 1
    field = window.speaker.lineEdit()
    field.setFocus()
    field.selectAll()
    QTest.keyClicks(field, 'speaker')
    field.setCursorPosition(4)
    QTest.keyClick(field, Qt.Key.Key_Left)
    assert field.cursorPosition() == 3
    assert window.selection.utterance == 1
    QTest.keyClick(field, Qt.Key.Key_Right, Qt.KeyboardModifier.AltModifier)
    assert window.selection.utterance == 2


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
    window.next.setFocus()
    QTest.qWait(30)
    assert window.selection.speaker == 2
    field.setFocus()
    field.selectAll()
    QTest.keyClicks(field, '999')
    window.next.setFocus()
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
    window.navigation.setCurrentRow(1)
    page = window.pages[1]
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
    window.navigation.setCurrentRow(1)
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
        window.pages[1].display(payload, ctx)
        assert not window.pages[1].message.isVisible()


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


def test_wheel_does_not_change_experiment_or_numeric_controls(window):
    from PySide6.QtCore import Qt
    page = window.training
    window.navigation.setCurrentRow(9)
    page.sections.setCurrentIndex(1)
    controls = [window.track, window.speaker, window.utterance, page.architecture,
                page.profile, page.folds, page.epochs]
    for control in controls:
        control.setFocus()
        getter = control.currentIndex if hasattr(control, 'currentIndex') else control.value
        before = getter()
        wheel(control)
        assert getter() == before
    # O teclado continua sendo uma interação explícita válida.
    page.architecture.setCurrentIndex(0)
    QTest.keyClick(page.architecture, Qt.Key.Key_Down)
    assert page.architecture.currentIndex() == 1


def test_wheel_over_history_scrolls_page_and_returning_keeps_canvas(window):
    window.navigation.setCurrentRow(9)
    page = window.training
    page.history_plot.update_history({'loss': [2, 1], 'accuracy': [.2, .6]})
    area = page.areas[0]
    wait_until(lambda: area.verticalScrollBar().maximum() > 0)
    scrollbar = area.verticalScrollBar()
    scrollbar.setValue(0)
    canvas = page.history_plot.canvas
    QTest.qWait(60)
    before = canvas.grab().toImage()
    wheel(canvas)
    assert scrollbar.value() > 0
    for _ in range(3):
        scrollbar.setValue(scrollbar.maximum())
        QTest.qWait(20)
        scrollbar.setValue(0)
        QTest.qWait(20)
    assert page.history_plot.canvas is canvas
    assert canvas.grab().toImage() == before


def test_epoch_update_preserves_canvas_and_scroll_position(window, tmp_path):
    from types import SimpleNamespace
    page = window.training
    window.navigation.setCurrentRow(9)
    directory = tmp_path / 'cnn/particao1'
    directory.mkdir(parents=True)
    progress = directory / 'progresso.json'
    job = SimpleNamespace(process=SimpleNamespace(poll=lambda: None, pid=321),
                          stopping=False, output=tmp_path, architecture='cnn',
                          started=0, lines=lambda: 'época concluída\n')
    page.manager.job = job
    try:
        progress.write_text(json.dumps({'epoca': 1, 'particao': 1,
                            'historico': {'loss': [2], 'accuracy': [.2]}}))
        page.poll()
        area = page.areas[0]
        wait_until(lambda: area.verticalScrollBar().maximum() > 0)
        bar = area.verticalScrollBar()
        bar.setValue(bar.maximum() - 20)
        QTest.qWait(30)
        before, maximum, canvas = bar.value(), bar.maximum(), page.history_plot.canvas
        progress.write_text(json.dumps({'epoca': 2, 'particao': 1,
                            'historico': {'loss': [2, 1], 'accuracy': [.2, .6]}}))
        page.poll()
        QTest.qWait(50)
        assert page.history_plot.canvas is canvas
        assert bar.maximum() == maximum
        assert bar.value() == before
        assert list(page.history_plot.lines['accuracy'].get_ydata()) == [20, 60]
    finally:
        page.manager.job = None


def test_result_partition_wheel_keeps_curves_and_sections_keep_scroll(window, corpus_tree):
    for fold in (1, 2):
        directory = corpus_tree / f'models/test/cnn/particao{fold}'
        directory.mkdir(parents=True)
        (directory / 'metricas.json').write_text(json.dumps({
            'acuracia': .5, 'f1_macro': .5, 'acaso': .5,
            'num_classes': 2, 'num_amostras_teste': 4}))
        (directory / 'progresso.json').write_text(json.dumps({
            'particao': fold, 'epoca': 2, 'historico': {'loss': [2, 1], 'accuracy': [.2, .5]}}))
    page = window.pages[10]
    window.navigation.setCurrentRow(10)
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
    page.sections.setCurrentIndex(3)
    page.sections.setCurrentIndex(1)
    assert page.areas[1].verticalScrollBar().value() == previous
    assert page.result_history.canvas is canvas
