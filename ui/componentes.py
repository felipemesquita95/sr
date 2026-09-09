"""Widgets compartilhados para a inspeção dos artefatos.

Texto é apresentado literalmente e tabelas são de leitura. Imagens
decodificadas pelos workers só viram QPixmap aqui, na thread da janela,
onde devem ser criados e usados todos os componentes deste módulo.
"""
from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QDialog, QFrame,
    QGraphicsScene, QGraphicsView, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QPlainTextEdit, QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)


def scroll_page(widget, event):
    """Entrega a roda à página sem alterar filtros ou engolir a rolagem no canvas."""
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QScrollArea):
            viewport = parent.viewport()
            position = viewport.mapFromGlobal(event.globalPosition().toPoint())
            forwarded = QWheelEvent(QPointF(position), event.globalPosition(),
                event.pixelDelta(), event.angleDelta(), event.buttons(), event.modifiers(),
                event.phase(), event.inverted(), event.source())
            QApplication.sendEvent(viewport, forwarded)
            event.setAccepted(forwarded.isAccepted())
            return
        parent = parent.parentWidget()
    event.ignore()


class ChoiceBox(QComboBox):
    """A escolha muda por clique/teclado; rolar sobre o campo não muda o experimento."""
    def wheelEvent(self, event):
        if self.view().isVisible():
            super().wheelEvent(event)
        else:
            scroll_page(self, event)


class SampleSelector(ChoiceBox):
    """Confirma IDs digitados sem deixar o texto divergir da seleção real."""
    validation_failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.lineEdit().returnPressed.connect(self.commit_text)
        self.lineEdit().editingFinished.connect(self.commit_text)
        QApplication.instance().focusChanged.connect(self.commit_on_focus_change)

    def commit_on_focus_change(self, old, new):
        popup = self.completer().popup()
        if (old is not None and (old is self or self.isAncestorOf(old)) and new is not None
                and new is not self and not self.isAncestorOf(new)
                and new is not popup and not popup.isAncestorOf(new)
                and self.lineEdit().isModified()):
            self.commit_text()

    def commit_text(self):
        if not self.isEnabled() or not self.count():
            return
        query = self.currentText().strip().casefold()
        matches = []
        for index in range(self.count()):
            text = self.itemText(index).casefold()
            data = self.itemData(index)
            aliases = {text, text.split(' — ')[-1]}
            if isinstance(data, int):
                aliases.add(str(data))
            if query in aliases or (query.isdecimal() and isinstance(data, int) and int(query) == data):
                matches.append(index)
        if len(matches) == 1:
            self.setCurrentIndex(matches[0])
            self.setEditText(self.itemText(matches[0]))
        else:
            previous = self.itemText(self.currentIndex())
            self.setEditText(previous)
            self.validation_failed.emit(f'“{query}” não identifica uma opção disponível. Seleção mantida: {previous}.')
        self.lineEdit().setModified(False)


class NumberBox(QSpinBox):
    def wheelEvent(self, event):
        scroll_page(self, event)


class ReadOnlyTable(QTableWidget):
    def wheelEvent(self, event):
        bar = self.verticalScrollBar()
        delta = event.pixelDelta().y() or event.angleDelta().y()
        at_edge = (delta < 0 and bar.value() == bar.maximum()) or (delta > 0 and bar.value() == bar.minimum())
        if bar.maximum() == 0 or at_edge:
            scroll_page(self, event)
        else:
            super().wheelEvent(event)


class LogView(QPlainTextEdit):
    """O log rola internamente e devolve o gesto à página ao chegar às extremidades."""
    def wheelEvent(self, event):
        bar = self.verticalScrollBar()
        delta = event.pixelDelta().y() or event.angleDelta().y()
        if (delta < 0 and bar.value() == bar.maximum()) or (delta > 0 and bar.value() == bar.minimum()):
            scroll_page(self, event)
        else:
            super().wheelEvent(event)


def label(text, name='', wrap=True):
    """Apresenta textos de artefatos literalmente, sem interpretação automática de HTML.

    Args:
        text: Valor convertido para texto selecionável.
        name: Identificador usado pela folha de estilo.
        wrap: Permite quebra de linha para acomodar descrições e avisos.

    Returns:
        Rótulo Qt com formato de texto simples.
    """
    item = QLabel(str(text))
    item.setTextFormat(Qt.TextFormat.PlainText)
    item.setObjectName(name)
    item.setWordWrap(wrap)
    item.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return item


def button(text, callback=None, primary=False):
    item = QPushButton(text)
    item.setCursor(Qt.CursorShape.PointingHandCursor)
    if primary:
        item.setObjectName('primary')
    if callback:
        item.clicked.connect(callback)
    return item


def row(*widgets):
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(16)
    for item in widgets:
        layout.addWidget(item, 1)
    return widget


def card(title='', description=''):
    """Fornece um contêiner e seu layout para compor conteúdo variável nas páginas.

    Args:
        title: Cabeçalho opcional.
        description: Contexto opcional abaixo do cabeçalho.

    Returns:
        Par de frame estilizado e layout no qual o chamador acrescenta widgets.
    """
    widget = QFrame()
    widget.setObjectName('card')
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(20, 18, 20, 18)
    layout.setSpacing(12)
    if title:
        layout.addWidget(label(title, 'cardTitle'))
    if description:
        layout.addWidget(label(description, 'muted'))
    return widget, layout


def stat(title, value, detail=''):
    widget, layout = card()
    layout.addWidget(label(title.upper(), 'eyebrow'))
    layout.addWidget(label(value, 'stat'))
    if detail:
        layout.addWidget(label(detail, 'muted'))
    return widget


def table(headers, rows, height=300):
    """Mantém a ordenação numérica dos valores sem permitir edição dos resultados.

    Números são atribuídos ao papel de exibição como números, evitando que
    índices sejam ordenados lexicograficamente. Valores já formatados em texto
    continuam tendo ordenação textual.

    Args:
        headers: Títulos das colunas.
        rows: Linhas com valores textuais ou numéricos.
        height: Altura máxima do painel em pixels.

    Returns:
        Tabela de leitura com seleção por linha e ordenação habilitada.
    """
    item = ReadOnlyTable(len(rows), len(headers))
    item.setHorizontalHeaderLabels(headers)
    item.verticalHeader().hide()
    item.setAlternatingRowColors(True)
    item.setShowGrid(False)
    item.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    item.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    item.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    item.verticalHeader().setDefaultSectionSize(38)
    item.setFixedHeight(min(height, 46 + max(1, len(rows)) * 38))
    item.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    for r, values in enumerate(rows):
        for c, value in enumerate(values):
            cell = QTableWidgetItem(str(value))
            if isinstance(value, (int, float)):
                cell.setData(Qt.ItemDataRole.DisplayRole, value)
            cell.setToolTip(str(value))
            item.setItem(r, c, cell)
    item.setSortingEnabled(True)
    item.sortItems(0, Qt.SortOrder.AscendingOrder)
    return item


class ZoomView(QGraphicsView):
    """Permite examinar detalhes do PNG sem capturar a rolagem comum da página.

    O ajuste automático vale até um zoom manual. A imagem vira QPixmap aqui,
    por isso a construção deve ocorrer na thread da janela.

    Args:
        image: QImage já decodificada.
        parent: Widget pai opcional.
    """
    def __init__(self, image: QImage, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.picture = self.scene().addPixmap(QPixmap.fromImage(image))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMinimumHeight(255)
        self.setStyleSheet('background: #ffffff; border: none;')
        self.auto_fit = True

    def fit(self):
        """Retoma o ajuste automático preservando a proporção da imagem.

        Redimensionamentos seguintes voltam a ajustar a figura ao espaço disponível.
        """
        self.auto_fit = True
        self.fitInView(self.picture, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom(self, factor):
        """Preserva a escala escolhida pelo usuário nos próximos redimensionamentos.

        Args:
            factor: Multiplicador de escala; só aplica resultados entre os limites de zoom.
        """
        self.auto_fit = False
        scale = self.transform().m11() * factor
        if .03 < scale < 25:
            self.scale(factor, factor)

    def wheelEvent(self, event):
        # A roda comum continua rolando a página; Ctrl + roda é zoom explícito.
        """Reserva Ctrl + roda ao zoom e devolve a roda comum à página.

        Args:
            event: Evento de roda recebido pela viewport.
        """
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom(1.2 if event.angleDelta().y() > 0 else 1 / 1.2)
            event.accept()
        else:
            scroll_page(self, event)

    def resizeEvent(self, event):
        """Reajusta apenas imagens que ainda estejam em modo automático.

        Args:
            event: Evento de redimensionamento encaminhado também à classe base.
        """
        super().resizeEvent(event)
        if self.auto_fit:
            self.fit()

    def showEvent(self, event):
        """Adia o ajuste até o layout conhecer o tamanho visível da viewport.

        Args:
            event: Evento de exibição encaminhado também à classe base.
        """
        super().showEvent(event)
        if self.auto_fit:
            QTimer.singleShot(0, self.fit)


class ImagePanel(QFrame):
    """Reúne navegação da imagem e inspeção ampliada sem nova leitura do disco.

    Args:
        title: Legenda do artefato.
        image: QImage decodificada que será compartilhada com a ampliação.
        source: Caminho de procedência opcional, guardado pelo painel.
    """
    def __init__(self, title, image, source=None):
        super().__init__()
        self.setObjectName('card')
        self.image = image
        self.source = source
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        toolbar = QHBoxLayout()
        toolbar.addWidget(label(title, 'cardTitle'), 1)
        self.view = ZoomView(image)
        for text, action in [('−', lambda: self.view.zoom(1 / 1.3)),
                             ('+', lambda: self.view.zoom(1.3)),
                             ('Ajustar', self.view.fit), ('Ampliar', self.expand)]:
            toolbar.addWidget(button(text, action))
        layout.addLayout(toolbar)
        layout.addWidget(self.view)
        layout.addWidget(label('Ctrl + roda para zoom · arraste para mover', 'muted'))

    def expand(self):
        """Abre uma vista independente sobre a mesma imagem já carregada.

        Zoom e deslocamento no diálogo não alteram a vista embutida na página.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle('Inspeção da figura')
        dialog.resize(1150, 750)
        layout = QVBoxLayout(dialog)
        view = ZoomView(self.image)
        layout.addWidget(view)
        layout.addWidget(button('Ajustar à janela', view.fit))
        dialog.exec()
