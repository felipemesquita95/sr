"""Leitura de resultados persistidos."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractItemView, QLineEdit, QVBoxLayout, QWidget

from ui import dados, figuras
from ui.componentes import ChoiceBox, ImagePanel, button, card, label, row, stat, table
from ui.conteudo import image
from ui.paginas import Page, SectionedPage

ARCHITECTURES = {
    'cnn': ('Coeficientes', 'Flatten'),
    'temporal_cnn': ('Tempo', 'Média global'),
    'temporal_cnn_stats': ('Tempo', 'Média + desvio'),
    'attention': ('Coeficientes', 'Atenção + Flatten'),
    'xvector': ('Tempo / TDNN', 'Estatísticas'),
    'xvector_attentive': ('Tempo / TDNN', 'Estatísticas com atenção'),
}


def forma(shape) -> str:
    """Descreve a forma de um tensor omitindo a dimensão do lote.

    Args:
        shape: Tupla do Keras, cuja primeira posição é o lote indefinido.

    Returns:
        Dimensões separadas por ``×``, ou um traço quando a forma é desconhecida.
    """
    dimensoes = [str(d) for d in tuple(shape)[1:] if d is not None]
    return ' × '.join(dimensoes) if dimensoes else '—'


class ModelPage(Page):
    """Mostra as redes treinadas para a trilha: camadas, formas e tamanho.

    A topologia é lida do próprio checkpoint, e não redescrita na interface,
    para que a página não possa divergir do que foi de fato treinado.

    Args:
        stage: Identificador da etapa no catálogo.
        title: Título apresentado na página.
        description: Explicação do estágio para o usuário.
    """
    def __init__(self, stage, title, description):
        super().__init__(stage, title, description)
        self.summaries = []
        self.architecture = ChoiceBox()
        self.architecture.setMinimumWidth(240)
        self.controls.addWidget(label('Arquitetura:', 'muted'))
        self.controls.addWidget(self.architecture)
        self.controls.addStretch()
        self.architecture.currentIndexChanged.connect(self.rebuild)

    def display(self, data, ctx):
        """Preenche o seletor preservando a arquitetura escolhida, quando ela persistir.

        Args:
            data: Resumos já lidos dos checkpoints em segundo plano.
            ctx: Seleção à qual esses artefatos correspondem.
        """
        self.summaries = data['modelos']
        self.context = ctx
        old = self.architecture.currentText()
        names = [resumo['arquitetura'] for resumo in self.summaries]
        self.architecture.blockSignals(True)
        self.architecture.clear()
        self.architecture.addItems(names)
        if old in names:
            self.architecture.setCurrentText(old)
        self.architecture.blockSignals(False)
        self.architecture.setEnabled(len(names) > 1)
        self.rebuild()

    def rebuild(self):
        """Redesenha a arquitetura escolhida sem repetir a leitura dos checkpoints."""
        if not hasattr(self, 'context'):
            return
        self.reset_body()
        if not self.summaries:
            self.notice('Nenhum modelo treinado foi encontrado para o perfil desta trilha. '
                        'Os resumos aparecem quando existir um arquivo modelo.keras em runs/models/.', True)
            self.content.addStretch()
            self.ready()
            return
        chosen = next((r for r in self.summaries if r['arquitetura'] == self.architecture.currentText()),
                      self.summaries[0])
        if 'erro' in chosen:
            self.notice(f'{chosen["arquitetura"]}: não foi possível ler o checkpoint. {chosen["erro"]}', True)
        else:
            self.content.addWidget(row(
                stat('Parâmetros', f'{chosen["parametros"]:,}'.replace(',', '.'), 'Pesos do modelo salvo'),
                stat('Camadas', str(len(chosen['camadas'])), chosen['arquitetura']),
                stat('Entrada → saída', f'{forma(chosen["entrada"])}  →  {forma(chosen["saida"])}',
                     'Coeficientes × quadros  →  locutores')))
            eixo, agregacao = ARCHITECTURES.get(chosen['arquitetura'], ('Não registrado', 'Não registrado'))
            widget, layout = card('Camadas, na ordem em que o sinal atravessa a rede',
                                  f'Convolução sobre: {eixo}   ·   Agregação temporal: {agregacao}')
            layout.addWidget(table(['#', 'Camada', 'Tipo', 'Saída', 'Parâmetros'],
                [[i + 1, c['nome'], c['tipo'], forma(c['saida']), f'{c["parametros"]:,}'.replace(',', '.')]
                 for i, c in enumerate(chosen['camadas'])], 360))
            self.content.addWidget(widget)
            self.content.addWidget(label(f'Procedência: {chosen["origem"]}', 'muted'))
        legiveis = [r for r in self.summaries if 'erro' not in r]
        if len(legiveis) > 1:
            widget, layout = card('As arquiteturas treinadas para esta trilha',
                                  'Mesma entrada e o mesmo número de locutores. O que muda é o tamanho '
                                  'da rede e como ela resume o tempo.')
            layout.addWidget(table(['Arquitetura', 'Parâmetros', 'Camadas', 'Convolução sobre', 'Agregação'],
                [[r['arquitetura'], f'{r["parametros"]:,}'.replace(',', '.'), len(r['camadas']),
                  *ARCHITECTURES.get(r['arquitetura'], ('Não registrado', 'Não registrado'))] for r in legiveis]))
            self.content.addWidget(widget)
            self.content.addWidget(figuras.barras([(r['arquitetura'], r['parametros'] / 1000) for r in legiveis],
                                                  'Tamanho de cada arquitetura', 'Parâmetros (milhares)'))
        self.notice('O número de parâmetros mede o tamanho da rede, não a sua capacidade de identificar '
                    'voz. Compare os resultados persistidos de cada protocolo antes de atribuir o acerto à voz.')
        self.content.addStretch()
        self.ready()


class ResultsPage(SectionedPage):
    """Compara somente resultados de uma mesma arquitetura e número de classes.

    Os perfis também aparecem quando ainda não têm métricas, tornando
    experimentos pendentes visíveis. Detalhes são carregados sob demanda
    para não decodificar todas as figuras de todos os treinos ao abrir a página.

    Args:
        stage: Identificador da etapa no catálogo.
        title: Título apresentado na página.
        description: Explicação do estágio para o usuário.
        profiles: Mapa de arquivos de perfil para configurações.
        tasks: Gerenciador criado na thread da janela para trabalhos em segundo plano.
    """
    section_names = ('Visão geral', 'Perda e acurácia', 'Matriz e erros')

    def __init__(self, stage, title, description, profiles, tasks):
        super().__init__(stage, title, description)
        self.profiles, self.tasks = profiles, tasks
        self.records = []
        self.experiment, self.architecture, self.classes = ChoiceBox(), ChoiceBox(), ChoiceBox()
        for widget in (self.experiment, self.architecture, self.classes):
            widget.setSizeAdjustPolicy(ChoiceBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            widget.setMinimumContentsLength(8)
            self.controls.addWidget(widget, 1)
            widget.currentIndexChanged.connect(self.rebuild)
        self.classes.setFixedWidth(80)
        self.classes.setToolTip('Número de classes comparáveis')
        self.controls.addWidget(button('Atualizar', self.reload.emit))
        self.chooser = ChoiceBox()
        self.chooser.setFixedWidth(170)
        self.chooser.currentIndexChanged.connect(self.choose_detail)
        self.controls.insertWidget(3, self.chooser)
        self.detail_number = 0

    def display(self, data, ctx):
        """Atualiza os filtros sem perder escolhas que ainda existam nos novos dados.

        Sinais dos seletores são suspensos durante o preenchimento para evitar
        reconstruções intermediárias. Relatórios inválidos recebem aviso próprio.

        Args:
            data: Artefatos preparados pelo carregador da etapa.
            ctx: Seleção à qual esses artefatos correspondem.
        """
        self.records = data['metrics']
        self.context = ctx
        for combo, values in [
            (self.experiment, sorted({r['experimento'] for r in self.records} | {s.models_path.name for s in self.profiles.values()})),
            (self.architecture, sorted({r['arquitetura'] for r in self.records} | set(ARCHITECTURES))),
            (self.classes, [str(x) for x in sorted({r['num_classes'] for r in self.records})])]:
            old = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(values)
            if old in values:
                combo.setCurrentText(old)
            elif combo is self.experiment and ctx.settings and ctx.settings.models_path.name in values:
                combo.setCurrentText(ctx.settings.models_path.name)
            elif combo is self.classes:
                combo.setCurrentIndex(len(values) - 1)
            combo.blockSignals(False)
        self.alinhar_selecao(ctx)
        self.rebuild()
        for caminho in data['errors']:
            self.notice(f'Métricas ilegíveis ou incompletas: {caminho}', True)

    def alinhar_selecao(self, ctx):
        """Evita abrir numa combinação sem nenhum resultado persistido.

        A trilha escolhida no topo nem sempre foi treinada: ``vctk_mic2`` tem
        features mas não tem modelos. O seletor continua listando experimentos
        pendentes, para que sua ausência fique visível, mas a página não começa
        em um deles quando existe resultado legível para mostrar.

        Args:
            ctx: Seleção corrente, usada para preferir o experimento da trilha.
        """
        if not self.records:
            return
        atual = (self.experiment.currentText(), self.architecture.currentText(), self.classes.currentText())
        if any((r['experimento'], r['arquitetura'], str(r['num_classes'])) == atual for r in self.records):
            return
        preferido = ctx.settings.models_path.name if ctx.settings else ''
        candidatos = [r for r in self.records if r['experimento'] == preferido] or self.records
        escolhido = candidatos[0]
        for combo, valor in ((self.experiment, escolhido['experimento']),
                             (self.architecture, escolhido['arquitetura']),
                             (self.classes, str(escolhido['num_classes']))):
            combo.blockSignals(True)
            combo.setCurrentText(valor)
            combo.blockSignals(False)

    def rebuild(self):
        """Agrupa partições compatíveis e explicita quando não há desvio estimável.

        Uma única partição não sustenta uma estimativa de dispersão. Cada nova
        combinação de filtros também invalida pedidos de figuras ainda em trânsito.
        """
        if not hasattr(self, 'context'):
            return
        self.reset_sections()
        self.detail_number += 1
        architecture = self.architecture.currentText()
        count = int(self.classes.currentText() or 0)
        records = [r for r in self.records if r['arquitetura'] == architecture and r['num_classes'] == count]
        selected = [r for r in records if r['experimento'] == self.experiment.currentText()]
        if selected:
            accuracy = np.mean([r['acuracia'] for r in selected])
            f1 = np.mean([r['f1_macro'] for r in selected])
            self.content.addWidget(row(stat('Acurácia média', f'{accuracy:.2%}'), stat('F1 macro médio', f'{f1:.4f}'),
                                       stat('Partições disponíveis', str(len(selected)))))
            self.content.addWidget(table(['Partição', 'Acurácia', 'F1 macro', 'Acaso', 'Vezes o acaso', 'Amostras', 'Procedência'],
                [[r['particao'], f'{r["acuracia"]:.2%}', f'{r["f1_macro"]:.4f}', f'{r["acaso"]:.2%}',
                  f'{r["acuracia"] / r["acaso"]:.1f}×', r['num_amostras_teste'], r['diretorio']] for r in selected]))
        else:
            self.notice('Ainda não há resultados para este experimento, arquitetura e número de classes.')
        if selected:
            self.content.addWidget(row(button('Abrir perda e acurácia', lambda: self.sections.setCurrentIndex(1), primary=True),
                                       button('Inspecionar os erros', lambda: self.sections.setCurrentIndex(2))))
        self.notice('Uma partição não permite estimar desvio. Confira também VAD, truncamento, '
                    'features e configuração de treino. Cross-microfone compartilha sessão e '
                    'enunciados; não demonstra voz isolada.', True)
        self.content.addStretch()
        old_directory = self.chooser.currentData()
        self.chooser.blockSignals(True)
        self.chooser.clear()
        if selected:
            for record in selected:
                self.chooser.addItem('Partição ' + record['particao'].removeprefix('particao'), record['diretorio'])
            self.chooser.setCurrentIndex(max(0, self.chooser.findData(old_directory)))
        self.chooser.blockSignals(False)
        self.chooser.setEnabled(bool(selected))
        self.use_section(1)
        self.curve_message = label('Selecione uma partição com histórico disponível.', 'notice')
        self.content.addWidget(self.curve_message)
        self.result_history = figuras.HistoryPlot()
        self.result_history.hide()
        self.content.addWidget(self.result_history)
        self.content.addStretch()
        self.use_section(2)
        self.detail_host = QWidget()
        self.detail_layout = QVBoxLayout(self.detail_host)
        self.detail_layout.setContentsMargins(0, 0, 0, 0)
        self.content.addWidget(self.detail_host)
        self.content.addStretch()
        if selected:
            self.choose_detail()
        else:
            self.detail_layout.addWidget(label('Nenhuma partição disponível para inspecionar.', 'notice'))
        self.use_section(0)
        self.ready()

    def choose_detail(self):
        directory = self.chooser.currentData()
        if directory and hasattr(self, 'detail_layout'):
            self.detail(Path(directory))

    def detail(self, directory):
        """Carrega figuras sem permitir que uma partição antiga substitua a atual.

        Um contador identifica cada pedido; o callback descarta respostas que
        cheguem depois de outra escolha ou reconstrução dos filtros.

        Args:
            directory: Diretório da partição cujos artefatos serão apresentados.
        """
        self.detail_number += 1
        number = self.detail_number
        self.curve_message.setText('Carregando histórico de ' + directory.name + '…')
        self.result_history.hide()
        while self.detail_layout.count():
            self.detail_layout.takeAt(0).widget().deleteLater()
        self.detail_layout.addWidget(label('Carregando figuras da partição…', 'notice'))

        def load():
            return (dados.read_json(directory / 'progresso.json', live=True),
                    [image(directory / name, title) for name, title in
                     [('matriz_confusao.png', 'Matriz de confusão'), ('acuracia_por_locutor.png', 'Acurácia por locutor')]])

        def finish(value, error):
            if number != self.detail_number:
                return
            while self.detail_layout.count():
                self.detail_layout.takeAt(0).widget().deleteLater()
            if error:
                self.detail_layout.addWidget(label(error, 'warning'))
                self.curve_message.setText('Não foi possível ler o histórico: ' + error)
                return
            progress, images = value
            if progress and any(progress.get('historico', {}).values()):
                self.curve_message.setText(f'{directory.parents[1].name}  /  {directory.parent.name}  /  {directory.name}')
                self.result_history.update_history(progress.get('historico', {}), reset_view=True)
                self.result_history.show()
            else:
                self.curve_message.setText(f'Histórico numérico ausente: {directory / "progresso.json"}')
            for title, value, path in images:
                if not value.isNull():
                    self.detail_layout.addWidget(ImagePanel(title, value, path))
                else:
                    self.detail_layout.addWidget(label(f'{title}: artefato indisponível em {path}.', 'notice'))
            if not (directory / 'modelo.keras').is_file():
                self.detail_layout.addWidget(label('Métricas disponíveis; o arquivo do modelo não está presente nesta partição.', 'muted'))
        self.tasks.submit(load, finish)


class ComparacaoPage(Page):
    """Permite contrastar registros persistidos sem misturar suas procedências."""

    campos = ('experimento', 'arquitetura', 'particao', 'acuracia', 'f1_macro',
              'acaso', 'num_classes', 'num_amostras_teste', 'diretorio')
    titulos = ('Experimento', 'Arquitetura', 'Partição', 'Acurácia', 'F1 macro',
               'Acaso', 'Classes', 'Amostras de teste', 'Procedência')

    def __init__(self, stage, title, description):
        super().__init__(stage, title, description)
        self.filtro = QLineEdit()
        self.filtro.setPlaceholderText('Filtrar experimento, arquitetura, partição ou diretório…')
        self.controls.addWidget(self.filtro)
        self.filtro.textChanged.connect(self.filtrar)

    def display(self, data, ctx):
        """Exibe cada linha de dados.metrics com valores numéricos ordenáveis.

        Args:
            data: Métricas e caminhos rejeitados pelo leitor.
            ctx: Contexto da seleção, sem restringir os experimentos comparáveis.
        """
        self.reset_body()
        self.registros = data['metrics']
        self.notice('Selecione linhas com Ctrl ou Shift para contrastá-las lado a lado. '
                    'Acurácia, F1 e acaso usam a escala do arquivo. '
                    'Protocolos e números de classes diferentes exigem interpretação conjunta.')
        # A tabela é montada sem ordenação para que a linha ainda corresponda ao
        # registro: a matriz de transferência gera várias linhas do mesmo
        # diretório, e só o índice as distingue depois de ordenar ou filtrar.
        self.tabela = table(self.titulos, [[registro[campo] for campo in self.campos]
                                          for registro in self.registros], ordenavel=False)
        self.tabela.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        for linha, registro in enumerate(self.registros):
            self.tabela.item(linha, 0).setData(Qt.ItemDataRole.UserRole, linha)
            for coluna in range(self.tabela.columnCount()):
                self.tabela.item(linha, coluna).setToolTip(registro['arquivo'])
        self.tabela.setSortingEnabled(True)
        self.tabela.sortItems(0, Qt.SortOrder.AscendingOrder)
        self.content.addWidget(self.tabela)
        self.contraste = table(['Campo'], [])
        self.content.addWidget(self.contraste)
        self.tabela.itemSelectionChanged.connect(self.contrastar)
        if not self.registros:
            self.notice(f'Métricas ausentes: esperado {dados.RUNS / "models"}/*/*/particao*/metricas.json '
                        f'ou {dados.RUNS / "models"}/*/matriz_transferencia.json.', True)
        for caminho in data['errors']:
            self.notice(f'Métricas ilegíveis ou incompletas: {caminho}', True)
        self.filtrar(self.filtro.text())
        self.contrastar()
        self.content.addStretch()
        self.ready()

    def filtrar(self, texto):
        """Filtra todas as colunas e retira do contraste as linhas ocultas.

        Args:
            texto: Fragmento pesquisado sem distinção de maiúsculas.
        """
        if not hasattr(self, 'tabela'):
            return
        for linha in range(self.tabela.rowCount()):
            conteudo = ' '.join(self.tabela.item(linha, coluna).text()
                                for coluna in range(self.tabela.columnCount()))
            self.tabela.setRowHidden(linha, texto.casefold() not in conteudo.casefold())
        self.contrastar()

    def contrastar(self):
        """Mantém uma coluna por registro selecionado, inclusive após ordenar."""
        linhas = sorted({indice.row() for indice in self.tabela.selectionModel().selectedRows()
                         if not self.tabela.isRowHidden(indice.row())})
        registros = [self.registros[self.tabela.item(linha, 0).data(Qt.ItemDataRole.UserRole)]
                     for linha in linhas]
        nova = table(['Campo'] + [f'{r["experimento"]} / {r["arquitetura"]} / {r["particao"]}' for r in registros],
                     [[titulo] + [r[campo] for r in registros]
                      for campo, titulo in zip(self.campos, self.titulos)], ordenavel=False)
        for coluna, registro in enumerate(registros, start=1):
            for linha in range(nova.rowCount()):
                nova.item(linha, coluna).setToolTip(registro['arquivo'])
        self.content.replaceWidget(self.contraste, nova)
        self.contraste.deleteLater()
        self.contraste = nova
