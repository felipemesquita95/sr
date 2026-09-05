"""Visualização dos artefatos numéricos em canvas Qt.

As figuras usam diretamente ``FigureCanvasQTAgg`` para conviver com os
módulos do sistema que gravam com Agg. Como os gráficos são widgets,
todas as funções deste módulo devem ser chamadas na thread da janela.
"""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtWidgets import QVBoxLayout, QWidget


class Plot(QWidget):
    """Integra uma figura à navegação e à exportação oferecidas pelo matplotlib.

    Args:
        title: Título opcional da figura.
        height: Altura mínima do canvas em pixels.
    """
    def __init__(self, title='', height=340):
        super().__init__()
        self.figure = Figure(figsize=(10, 3.5), dpi=100, layout='constrained', facecolor='white')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(height)
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


def history(values):
    """Compara treino e validação mesmo quando só parte do histórico está disponível.

    Args:
        values: Séries de perda e acurácia, com chaves de validação prefixadas por ``val_``.

    Returns:
        Painel de curvas por época concluída.
    """
    plot = Plot(height=270)
    axes = plot.figure.subplots(1, 2)
    for ax, key, title in zip(axes, ['loss', 'accuracy'], ['Perda', 'Acurácia']):
        for name, caption, color in [(key, 'Treino', '#1a9a8b'), ('val_' + key, 'Validação', '#7d83c8')]:
            series = values.get(name, [])
            if series:
                ax.plot(range(1, len(series) + 1), series, color=color, label=caption)
        ax.set(title=title, xlabel='Época')
        ax.grid(alpha=.15)
        if ax.lines:
            ax.legend(frameon=False)
    return plot.finish()


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
