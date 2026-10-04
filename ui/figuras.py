"""Visualização dos artefatos numéricos em canvas Qt.

As figuras usam diretamente ``FigureCanvasQTAgg`` para conviver com os
módulos do sistema que gravam com Agg. Como os gráficos são widgets,
todas as funções deste módulo devem ser chamadas na thread da janela.
"""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget

from ui.componentes import label, scroll_page


class PageCanvas(FigureCanvasQTAgg):
    def wheelEvent(self, event):
        scroll_page(self, event)


class Plot(QWidget):
    """Integra uma figura à navegação e à exportação oferecidas pelo matplotlib.

    Args:
        title: Título opcional da figura.
        height: Altura mínima do canvas em pixels.
    """
    def __init__(self, title='', height=340):
        super().__init__()
        self.figure = Figure(figsize=(10, 3.5), dpi=100, layout='constrained', facecolor='white')
        self.canvas = PageCanvas(self.figure)
        self.canvas.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setMaximumHeight(35)
        self.toolbar.setStyleSheet('QToolButton { color: #263d52; padding: 4px; }')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        if title:
            self.figure.suptitle(title, fontsize=12, color='#29485c')

    def finish(self):
        """Agenda o desenho no laço Qt após concluir a composição dos eixos.

        Returns:
            O próprio painel, pronto para inserção no layout.
        """
        for ax in self.figure.axes:
            ax.tick_params(labelsize=9, colors='#61788b')
            for spine in ax.spines.values():
                spine.set_color('#dce5ed')
        self.canvas.draw_idle()
        return self


def raw_signal(values, caption):
    plot = Plot(height=430, title=caption)
    waveform, spectrum = plot.figure.subplots(2, 1)
    waveform.fill_between(values['time'], values['low'], values['high'], color='#168f83', linewidth=.4)
    waveform.set(title='Forma de onda original', xlabel='Tempo (s)', ylabel='Amplitude',
                 xlim=(0, values['duration']))
    spectrum.plot(values['frequency'] / 1000, values['magnitude'], color='#7562b5', linewidth=.7)
    spectrum.set(title='Espectro original', xlabel='Frequência (kHz)', ylabel='Magnitude')
    for axis in (waveform, spectrum):
        axis.grid(alpha=.15)
    return plot.finish()


def mfcc(matrices, titles):
    """Usa uma escala de cor comum para tornar a comparação entre trilhas legível.

    Args:
        matrices: Matrizes não vazias de coeficientes por quadro.
        titles: Um título por matriz, na mesma ordem.

    Returns:
        Painel com os mapas de calor e barras de escala.
    """
    plot = Plot()
    axes = np.atleast_1d(plot.figure.subplots(1, len(matrices)))
    # A escala compartilhada torna a comparação entre trilhas diretamente legível.
    lower = min(float(np.min(m)) for m in matrices)
    upper = max(float(np.max(m)) for m in matrices)
    for ax, matrix, title in zip(axes, matrices, titles):
        image = ax.imshow(matrix, origin='lower', aspect='auto', cmap='viridis',
                          interpolation='nearest', vmin=lower, vmax=upper)
        ax.set(title=title, xlabel='Quadro', ylabel='Coeficiente cepstral')
        plot.figure.colorbar(image, ax=ax, shrink=.85)
    return plot.finish()


def signatures(values):
    """Expõe a fronteira entre médias e desvios nas assinaturas persistidas.

    Args:
        values: Mapa não vazio das condições disponíveis para seus vetores cepstrais.

    Returns:
        Painel com as condições presentes, sem preencher condições ausentes.
    """
    plot = Plot()
    ax = plot.figure.subplots()
    names = {'silence': 'Baixa energia', 'speech': 'Fala', 'full': 'Sinal completo'}
    for (key, value), color in zip(values.items(), ['#1a9a8b', '#6480c4', '#d79b43']):
        ax.plot(range(1, len(value) + 1), value, label=names[key], color=color, linewidth=1.7)
    middle = len(next(iter(values.values()))) / 2
    ax.axvline(middle + .5, color='#a7b6c4', linestyle='--', label='Médias | desvios')
    ax.set(xlabel='Estatística cepstral', ylabel='Valor')
    ax.legend(frameon=False)
    ax.grid(alpha=.15)
    return plot.finish()


class HistoryPlot(QFrame):
    """Mantém canvas e geometria entre épocas: atualizar o histórico não move a página."""
    def __init__(self, values=None):
        super().__init__()
        self.setObjectName('historyCard')
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 12)
        layout.setSpacing(12)
        heading = QHBoxLayout()
        heading.addWidget(label('Evolução do aprendizado', 'cardTitle'), 1)
        heading.addWidget(label('━ Treino', 'legendTrain', wrap=False))
        heading.addWidget(label('┄ Validação', 'legendValidation', wrap=False))
        heading.addSpacing(12)
        self.epochs = label('Aguardando épocas', 'chip', wrap=False)
        heading.addWidget(self.epochs)
        layout.addLayout(heading)
        self.plot = Plot(height=320)
        self.axes = self.plot.figure.subplots(1, 2)
        self.lines = {}
        for ax, key, title, ylabel in zip(self.axes, ['loss', 'accuracy'],
                                         ['Perda', 'Acurácia'], ['Entropia cruzada', 'Acertos (%)']):
            ax.set_title(title, loc='left', fontsize=13, fontweight='bold', color='#29485c', pad=16)
            ax.set(xlabel='Época', ylabel=ylabel)
            ax.set_facecolor('#f8fbfd')
            ax.grid(axis='y', alpha=.18)
            for name, caption, color, style in [(key, 'Treino', '#138f82', '-'),
                                                ('val_' + key, 'Validação', '#7e72c4', '--')]:
                self.lines[name], = ax.plot([], [], color=color, label=caption, linewidth=2.2,
                                           linestyle=style, marker='o', markersize=3)
        self.plot.finish()
        self.chart_stack = QStackedWidget()
        self.chart_stack.addWidget(self.plot)
        empty = label('Nenhuma época registrada\n\nAs curvas aparecem quando o treinamento produzir um histórico.', 'muted')
        empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.chart_stack.addWidget(empty)
        self.chart_stack.setFixedHeight(361)
        layout.addWidget(self.chart_stack)
        self.last_values = {}
        self.update_history(values or {})

    @property
    def canvas(self):
        return self.plot.canvas

    def update_history(self, values, *, reset_view=False):
        if reset_view:
            self.plot.toolbar.update()
            for ax in self.axes:
                ax.autoscale(enable=True)
        self.last_values = values
        maximum = max((len(values.get(key, [])) for key in self.lines), default=0)
        self.chart_stack.setCurrentIndex(0 if maximum else 1)
        self.epochs.setText(f'{maximum} épocas registradas' if maximum else 'Aguardando épocas')
        for key, line in self.lines.items():
            series = np.asarray(values.get(key, []), dtype=float)
            if 'accuracy' in key:
                series = series * 100
            line.set_data(np.arange(1, len(series) + 1), series)
            line.set_markevery(max(1, len(series) // 20))
        # Um zoom manual continua válido durante o acompanhamento; Restaurar
        # na barra do gráfico volta aos limites originais.
        zoomed = self.plot.toolbar._nav_stack() is not None
        if not zoomed:
            for ax in self.axes:
                ax.relim()
                ax.autoscale_view()
                ax.set_xlim(.5, max(2, maximum) + .5)
        self.canvas.draw_idle()


def history(values):
    return HistoryPlot(values)


def lengths(split, frames, cap):
    """Torna visível o efeito do comprimento comum sobre cada conjunto.

    Args:
        split: Referências de gravações agrupadas por destino.
        frames: Comprimento efetivo derivado do treino.
        cap: Teto configurado; valores não positivos o desativam.

    Returns:
        Histograma das larguras originais com marcas do comprimento e do teto.
    """
    plot = Plot(height=250)
    ax = plot.figure.subplots()
    for (name, refs), color in zip(split.items(), ['#27968e', '#d7a254', '#7481bd']):
        ax.hist([r.shape[1] for r in refs], bins=35, alpha=.5, label=name, color=color)
    ax.axvline(frames, color='#22485c', label=f'Comprimento: {frames}', linestyle='--')
    if cap > 0 and cap != frames:
        ax.axvline(cap, color='#b35f5f', label=f'Teto: {cap}', linestyle=':')
    ax.set(xlabel='Quadros antes do alinhamento', ylabel='Gravações')
    ax.legend(frameon=False)
    return plot.finish()


def comparison(rows):
    """Contextualiza a acurácia dos protocolos pelo acaso e pela variação entre partições.

    O chamador deve selecionar uma mesma arquitetura e número de classes.
    Uma única partição tem desvio não estimável, representado sem barra de erro.

    Args:
        rows: Registros com ``experiment``, ``mean``, ``std`` e ``chance`` em escala de zero a um.

    Returns:
        Painel com acurácias em porcentagem e referência do acaso do primeiro registro.
    """
    plot = Plot(height=max(280, len(rows) * 40))
    ax = plot.figure.subplots()
    y = np.arange(len(rows))
    means = [r['mean'] * 100 for r in rows]
    errors = [r['std'] * 100 if r['std'] is not None else 0 for r in rows]
    ax.barh(y, means, xerr=errors, color='#23968b', height=.58, capsize=3)
    ax.set_yticks(y, [r['experiment'] for r in rows])
    for pos, value in zip(y, means):
        ax.text(value + 1.3, pos, f'{value:.2f}%', va='center', fontsize=9, color='#3f6177')
    ax.set(xlim=(0, 113), xlabel='Acurácia (%)')
    ax.invert_yaxis()
    if rows:
        ax.axvline(rows[0]['chance'] * 100, color='#d99d47', linestyle='--', label='Acaso uniforme')
        ax.legend(frameon=False)
    ax.grid(axis='x', alpha=.12)
    return plot.finish()


def distribuicoes(series, title, ylabel='Acurácia (%)'):
    """Desenha cada distribuição persistida, sem reduzi-la a uma média.

    Args:
        series: Pares de legenda e valores em escala unitária ou percentual.
        title: Título da evidência.
        ylabel: Rótulo do eixo vertical.

    Returns:
        Painel matplotlib com pontos individuais e caixas de distribuição.
    """
    plot = Plot(title=title, height=360)
    ax = plot.figure.subplots()
    labels, values = zip(*[(name, np.asarray(value, dtype=float)) for name, value in series if len(value)])
    scale = np.asarray([item for group in values for item in group])
    factor = 100 if np.nanmax(np.abs(scale)) <= 1 else 1
    positions = np.arange(len(values))
    ax.boxplot([value * factor for value in values], positions=positions, showmeans=False)
    ax.set_xticks(positions, labels)
    for position, value in zip(positions, values):
        ax.scatter(np.full(len(value), position), value * factor, alpha=.45, color='#168f83', s=12)
    ax.set(ylabel=ylabel)
    ax.grid(axis='y', alpha=.15)
    return plot.finish()


def barras(registros, title, ylabel='Percentual'):
    """Mostra medidas heterogêneas já calculadas pelo experimento.

    Args:
        registros: Pares de legenda e valor em escala unitária ou percentual.
        title: Título da evidência.
        ylabel: Rótulo do eixo vertical.

    Returns:
        Painel matplotlib com valores rotulados a partir dos artefatos.
    """
    plot = Plot(title=title, height=340)
    ax = plot.figure.subplots()
    names, raw = zip(*registros)
    values = np.asarray(raw, dtype=float)
    factor = 100 if np.nanmax(np.abs(values)) <= 1 else 1
    ax.bar(np.arange(len(values)), values * factor, color='#23968b')
    ax.set_xticks(np.arange(len(values)), names, rotation=20, ha='right')
    ax.set(ylabel=ylabel)
    ax.grid(axis='y', alpha=.15)
    return plot.finish()
