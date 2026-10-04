"""Entrada do aplicativo desktop: .venv/bin/python ui/app.py."""
import os
os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('QT_API', 'pyside6')

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for source in (ROOT, ROOT / 'src'):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtGui import QColor, QIcon, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QComboBox, QCompleter, QFrame, QHBoxLayout,
    QListWidget, QMainWindow, QStackedWidget, QVBoxLayout, QWidget)

from ui import dados
from ui.componentes import SampleSelector, button, label
from ui.conteudo import ROTEIRO, SELECTION_STAGES, STAGES, Selection, load_stage
from ui.estilo import STYLE
from ui.evidencias import EvidencePage
from ui.passos import PassoPage
from ui.tarefas import Tasks


def combo(name):
    widget = SampleSelector()
    widget.setObjectName(name)
    widget.setEditable(True)
    widget.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    widget.setMinimumWidth(130)
    widget.completer().setFilterMode(Qt.MatchFlag.MatchContains)
    widget.completer().setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
    return widget


def app_icon():
    pixmap = QPixmap(64, 64)
    pixmap.fill(QColor('#122437'))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor('#6ee3cb'), 4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    for x, height in [(12, 9), (22, 23), (32, 38), (42, 23), (52, 12)]:
        painter.drawLine(x, 32 - height // 2, x, 32 + height // 2)
    painter.end()
    return QIcon(pixmap)


class MainWindow(QMainWindow):
    def __init__(self, profiles=None, persist=True):
        super().__init__()
        self.setWindowTitle('SR Studio · Reconhecimento de locutor')
        self.setWindowIcon(app_icon())
        self.resize(1440, 940)
        self.setMinimumSize(1060, 720)
        self.setStyleSheet(STYLE)
        self.persist = persist
        self.preferences = QSettings('SR', 'Studio') if persist else None
        self.tasks = Tasks(self)
        self.profiles = dados.profiles() if profiles is None else profiles
        self.selection = None
        self.inventory = {}
        self.revision = 0
        self.inventory_revision = 0
        self.requested = {}
        self.counts = label('Lendo as trilhas…', 'brandSub')
        self.statusBar().showMessage('Abrindo artefatos locais…')
        self.build()
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(75)
        self.debounce.timeout.connect(self.refresh_page)
        self.track.currentIndexChanged.connect(self.track_changed)
        self.speaker.currentIndexChanged.connect(self.speaker_changed)
        self.utterance.currentIndexChanged.connect(self.sample_changed)
        self.navigation.currentRowChanged.connect(self.navigate)
        self.shortcuts = []
        for sequence, callback in [('Alt+Down', lambda: self.step_page(1)), ('Alt+Up', lambda: self.step_page(-1)),
                                   ('Ctrl+R', self.refresh_data), ('Ctrl+F', self.focus_speaker)]:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)
        available = dados.tracks(dados.RUNS / 'features')
        self.track.blockSignals(True)
        for path in available:
            self.track.addItem(path.name, path)
        saved_track = self.preferences.value('track', '') if self.preferences else ''
        index = next((i for i, p in enumerate(available) if p.name == saved_track),
                     next((i for i, p in enumerate(available) if p.name == 'vctk_mic1'), 0))
        self.track.setCurrentIndex(index)
        self.track.blockSignals(False)
        if self.preferences and self.preferences.contains('geometry'):
            self.restoreGeometry(self.preferences.value('geometry'))
        self.navigation.setCurrentRow(0)
        if available:
            self.track_changed()
        else:
            self.counts.setText('Nenhuma trilha disponível')
            self.pages[0].error('Nenhuma trilha com MFCCs encontrada em runs/features/. Disponibilize os artefatos para iniciar.')
            self.statusBar().showMessage('Nenhum artefato encontrado')

    def build(self):
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setCentralWidget(root)
        sidebar = QWidget()
        sidebar.setObjectName('sidebar')
        sidebar.setFixedWidth(236)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(18, 27, 18, 20)
        side.setSpacing(6)
        side.addWidget(label('SR  /  STUDIO', 'brand'))
        side.addWidget(label('RECONHECIMENTO DE LOCUTOR', 'brandSub'))
        side.addSpacing(24)
        side.addWidget(label('PROCESSAMENTO', 'navGroup'))
        self.navigation = QListWidget()
        self.navigation.setObjectName('navigation')
        self.navigation.setSpacing(1)
        self.navigation.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for i, (_, title, _) in enumerate(ROTEIRO):
            self.navigation.addItem(f'{i + 1:02d}   {title}')
            self.navigation.item(i).setToolTip(title)
        side.addWidget(self.navigation, 1)
        side.addWidget(button('Anterior · Alt + ↑', lambda: self.step_page(-1)))
        side.addWidget(button('Próximo · Alt + ↓', lambda: self.step_page(1)))
        side.addWidget(self.counts)
        side.addSpacing(8)
        side.addWidget(label('LOCAL  ·  PYTHON + QT\nArtefatos do seu computador', 'brandSub'))
        layout.addWidget(sidebar)
        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.toolbar = QFrame()
        self.toolbar.setObjectName('selector')
        top = QVBoxLayout(self.toolbar)
        top.setContentsMargins(28, 18, 28, 14)
        line = QHBoxLayout()
        line.setSpacing(14)
        self.track, self.speaker, self.utterance = combo('track'), combo('speaker'), combo('utterance')
        for selector in (self.track, self.speaker, self.utterance):
            selector.validation_failed.connect(lambda message: self.statusBar().showMessage(message, 10000))
            selector.setToolTip('Digite o número ou nome exato e pressione Enter, ou escolha na lista.')
        for name, widget, stretch in [('TRILHA', self.track, 2), ('LOCUTOR', self.speaker, 2), ('ENUNCIADO', self.utterance, 2)]:
            wrapper = QWidget()
            field = QVBoxLayout(wrapper)
            field.setContentsMargins(0, 0, 0, 0)
            field.setSpacing(5)
            field.addWidget(label(name, 'eyebrow'))
            field.addWidget(widget)
            line.addWidget(wrapper, stretch)
        top.addLayout(line)
        trail = QHBoxLayout()
        self.breadcrumb = label('Selecione uma gravação para começar.', 'muted')
        self.badge = label('ARTEFATOS LOCAIS', 'chip')
        trail.addWidget(self.breadcrumb, 1)
        trail.addWidget(self.badge)
        top.addLayout(trail)
        main_layout.addWidget(self.toolbar)
        self.stack = QStackedWidget()
        self.pages = []
        self.page_by_stage = {}
        self.passo_por_secao = {}
        for stage, title, description in STAGES:
            page = PassoPage(stage, title, description, self.profiles, self.tasks)
            page.reload.connect(lambda p=page: self.reload_page(p))
            page.select_sample.connect(self.select_sample)
            self.pages.append(page)
            self.page_by_stage[stage] = page
            for nome, componente in page.componentes.items():
                self.page_by_stage[nome] = componente
                self.passo_por_secao[nome] = page
                if hasattr(componente, 'navigate_stage'):
                    componente.navigate_stage.connect(self.navigate_stage)
                if isinstance(componente, EvidencePage):
                    for evidencia in componente.stage_sections:
                        self.page_by_stage[evidencia] = componente
                        self.passo_por_secao[evidencia] = page
            self.stack.addWidget(page)
        main_layout.addWidget(self.stack, 1)
        layout.addWidget(main, 1)

    def navigate_stage(self, stage, section=''):
        """Abre uma etapa e sua evidência, usado no roteiro da defesa."""
        componente = self.page_by_stage.get(stage)
        if componente is None:
            return
        page = self.passo_por_secao.get(stage, componente)
        if componente is not page:
            page.sections.setCurrentWidget(componente)
        if isinstance(componente, EvidencePage):
            destino = section or componente.stage_sections.get(stage)
            if destino in componente.section_names:
                componente.sections.setCurrentIndex(componente.section_names.index(destino))
        self.navigation.setCurrentRow(self.pages.index(page))
        self.update_toolbar(page)
        self.refresh_page()

    def track_changed(self):
        path = self.track.currentData()
        if path is None:
            return
        self.inventory_revision += 1
        generation = self.inventory_revision
        self.selection = None
        self.revision += 1
        self.inventory = {}
        for widget in (self.speaker, self.utterance):
            widget.setEnabled(False)
        self.profile, self.settings = dados.track_profile(path, self.profiles)
        self.breadcrumb.setText(f'{path.name} · lendo índice…')
        self.pages[self.stack.currentIndex()].loading('Lendo o índice da trilha em segundo plano…')
        settings = self.settings

        def load():
            inventory = dados.inventory(path, bool(settings and settings.enable_vad))
            manifest = dados.read_json(dados.RUNS / 'features/vctk_manifesto.json') if path.name.startswith('vctk') else {}
            names, _ = dados.manifest_maps(manifest)
            return inventory, manifest, names

        def finish(value, error):
            if generation != self.inventory_revision:
                return
            if error:
                self.pages[self.stack.currentIndex()].error(error)
                return
            self.inventory, self.manifest, self.names = value
            if not self.inventory:
                self.pages[self.stack.currentIndex()].error('Esta trilha não possui matrizes MFCC disponíveis.')
                return
            old = self.speaker.currentData()
            if old is None and self.preferences:
                old = self.preferences.value('speaker', 1, type=int)
            self.speaker.blockSignals(True)
            self.speaker.clear()
            for speaker in self.inventory:
                self.speaker.addItem(f'{speaker} — {self.names[speaker]}' if speaker in self.names else str(speaker), speaker)
            self.speaker.setCurrentIndex(max(0, self.speaker.findData(old)))
            self.speaker.blockSignals(False)
            self.speaker.setEnabled(True)
            self.utterance.setEnabled(True)
            count = sum(map(len, self.inventory.values()))
            self.counts.setText(f'{len(self.inventory)} locutores\n{count:,} gravações'.replace(',', '.'))
            self.speaker_changed()
        self.tasks.submit(load, finish)

    def speaker_changed(self):
        speaker = self.speaker.currentData()
        if speaker not in self.inventory:
            return
        old = self.utterance.currentData()
        if old is None and self.preferences:
            old = self.preferences.value('utterance', 1, type=int)
        # Ao trocar o locutor nas etapas visuais, não herdar um enunciado sem
        # sinal quando há outro exemplo disponível desse mesmo locutor.
        stage = self.pages[self.stack.currentIndex()].sections.currentWidget().stage
        if stage in ('sinal', 'preprocessamento'):
            available = [u for u, complete in self.inventory[speaker].items() if complete]
            has_audio = (stage == 'sinal' and old is not None and dados.audio_source(
                self.settings, speaker, old, self.manifest, self.track.currentData().name))
            if available and old not in available and not has_audio:
                old = available[0]
        self.utterance.blockSignals(True)
        self.utterance.clear()
        for utterance, complete in self.inventory[speaker].items():
            self.utterance.addItem(f'{utterance:03d} · {"figuras" if complete else "MFCC"}', utterance)
        self.utterance.setCurrentIndex(max(0, self.utterance.findData(old)))
        self.utterance.blockSignals(False)
        self.sample_changed()

    def sample_changed(self):
        speaker, utterance = self.speaker.currentData(), self.utterance.currentData()
        if speaker not in self.inventory or utterance not in self.inventory[speaker]:
            return
        self.selection = Selection(self.track.currentData(), speaker, utterance, self.inventory,
                                   self.manifest, self.names, self.profile, self.settings)
        self.revision += 1
        name = self.names.get(speaker, f'Locutor {speaker}')
        self.breadcrumb.setText(f'{self.track.currentText()}   /   {name}   /   enunciado {utterance:03d}')
        complete = self.inventory[speaker][utterance]
        self.badge.setText('FIGURAS DISPONÍVEIS' if complete else 'MFCC DISPONÍVEL')
        page = self.pages[self.stack.currentIndex()]
        page.loading('Carregando a gravação selecionada…')
        self.debounce.start()

    def navigate(self, index):
        if index < 0:
            return
        self.stack.setCurrentIndex(index)
        self.update_toolbar(self.pages[index])
        self.refresh_page()

    def update_toolbar(self, page):
        """Mostra a seleção apenas quando ela controla a página aberta."""
        self.toolbar.setVisible(page.stage in SELECTION_STAGES)

    def reload_page(self, page):
        page.loaded_key = None
        for componente in page.componentes.values():
            componente.loaded_key = None
        self.requested.pop(page.stage, None)
        if page is self.pages[self.stack.currentIndex()]:
            self.refresh_page()

    def refresh_page(self):
        if not self.selection:
            return
        page = self.pages[self.stack.currentIndex()]
        ctx = self.selection
        fold = page.preparar(ctx)
        parameters = (fold,)
        key = ctx.key, self.revision, parameters
        if page.loaded_key == key:
            page.ready()
            return
        pending = getattr(page, 'pending_payload', None)
        if pending and pending[0] == key:
            page.pending_payload = None
            self.show_result(page, key, pending[1], pending[2])
            return
        if self.requested.get(page.stage) == key:
            return
        self.requested[page.stage] = key
        page.loading()
        self.statusBar().showMessage('Carregando em segundo plano · a navegação continua disponível')

        def finish(data, error):
            if self.requested.get(page.stage) != key or self.selection is None or ctx.key != self.selection.key or key[1] != self.revision:
                return
            self.requested.pop(page.stage, None)
            if error:
                page.error(error)
                self.statusBar().showMessage('Não foi possível ler esta etapa')
                return
            if page is not self.pages[self.stack.currentIndex()]:
                page.pending_payload = key, data, ctx
            else:
                self.show_result(page, key, data, ctx)
        self.tasks.submit(lambda: load_stage(page.stage, ctx, ctx.settings, fold), finish)

    def show_result(self, page, key, data, ctx):
        try:
            page.display(data, ctx)
            page.loaded_key = key
            for componente in page.componentes.values():
                componente.loaded_key = key
            self.statusBar().showMessage('Pronto   ·   Alt + ↑ ↓ roteiro   ·   Ctrl + F buscar locutor')
        except Exception as error:
            page.error(str(error))

    def select_sample(self, speaker, utterance):
        self.speaker.setCurrentIndex(self.speaker.findData(speaker))
        self.utterance.setCurrentIndex(self.utterance.findData(utterance))

    def step_page(self, direction):
        self.navigation.setCurrentRow(max(0, min(len(self.pages) - 1, self.navigation.currentRow() + direction)))

    def focus_speaker(self):
        self.speaker.setFocus()
        self.speaker.lineEdit().selectAll()

    def refresh_data(self):
        dados.clear_cache()
        for page in self.pages:
            page.loaded_key = None
        self.track_changed()

    def closeEvent(self, event):
        if self.preferences:
            self.preferences.setValue('geometry', self.saveGeometry())
            self.preferences.setValue('track', self.track.currentText())
            if self.selection:
                self.preferences.setValue('speaker', self.selection.speaker)
                self.preferences.setValue('utterance', self.selection.utterance)
        self.tasks.close()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('SR Studio')
    app.setOrganizationName('SR')
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
