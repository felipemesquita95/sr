"""Apresentação e acompanhamento de experimentos independentes da seleção.

A seleção informa a inspeção das arquiteturas, enquanto o treino segue o
perfil do formulário. Resultados são lidos dos artefatos e filtrados por
arquitetura e número de classes para evitar comparações incompatíveis.
"""
from pathlib import Path
import shutil

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox, QVBoxLayout, QWidget

from ui import dados, figuras
from ui.componentes import ChoiceBox, NumberBox, LogView, ImagePanel, button, card, label, row, stat, table
from ui.conteudo import image, model_parameters
from ui.execucao import TrainingManager, feature_paths, training_command
from ui.paginas import SectionedPage

ARCHITECTURES = {
    'cnn': ('Coeficientes', 'Flatten'),
    'temporal_cnn': ('Tempo', 'Média global'),
    'temporal_cnn_stats': ('Tempo', 'Média + desvio'),
    'attention': ('Coeficientes', 'Atenção + Flatten'),
    'xvector': ('Tempo / TDNN', 'Estatísticas'),
    'xvector_attentive': ('Tempo / TDNN', 'Estatísticas com atenção'),
}


class TrainingPage(SectionedPage):
    """Conserva o acompanhamento do processo enquanto o usuário navega pelo pipeline.

    O timer permanece ativo mesmo com a página oculta. A leitura do log não
    pertence aos widgets; estes apenas consultam o estado do gerenciador.

    Args:
        stage: Identificador da etapa no catálogo.
        title: Título apresentado na página.
        description: Explicação do estágio para o usuário.
        profiles: Mapa de arquivos de perfil para configurações.
        tasks: Gerenciador criado na thread da janela para trabalhos em segundo plano.
    """
    section_names = ('Acompanhamento', 'Novo experimento', 'Arquiteturas')

    def __init__(self, stage, title, description, profiles, tasks):
        super().__init__(stage, title, description)
        self.profiles, self.tasks = profiles, tasks
        self.manager = TrainingManager()
        self.context = None
        self.last_progress = None
        self.last_log = ''
        self.profile = ChoiceBox()
        for path in profiles:
            self.profile.addItem(path.stem, path)
        self.architecture = ChoiceBox()
        self.architecture.addItems(list(ARCHITECTURES))
        self.folds, self.epochs = NumberBox(), NumberBox()
        self.folds.setRange(1, 5)
        self.folds.setPrefix('Partições: ')
        self.epochs.setRange(1, 100000)
        self.epochs.setValue(1000)
        self.epochs.setPrefix('Épocas: ')
        self.use_section(1)
        form, layout = card('Configurar treinamento', 'Escolha o perfil e a rede. Você pode continuar navegando durante o treino.')
        layout.addWidget(row(self.profile, self.architecture))
        layout.addWidget(row(self.folds, self.epochs))
        self.destination = label('', 'muted')
        layout.addWidget(self.destination)
        self.disk = label('', 'notice')
        layout.addWidget(self.disk)
        self.start_button = button('Iniciar treinamento', self.start, primary=True)
        self.stop_button = button('Interromper', self.stop)
        self.stop_button.setObjectName('danger')
        self.stop_button.setEnabled(False)
        layout.addWidget(row(self.start_button, self.stop_button))
        self.content.addWidget(form)
        self.content.addStretch()
        self.use_section(0)
        self.status = label('Nenhum treino iniciado.', 'notice')
        self.content.addWidget(self.status)
        actions = row(button('Configurar um experimento', lambda: self.sections.setCurrentIndex(1)),
                      button('Ver arquiteturas', lambda: self.sections.setCurrentIndex(2)))
        self.content.addWidget(actions)
        self.history_host, self.history_layout = card()
        self.history_layout.setContentsMargins(0, 0, 0, 0)
        self.history_title = label('As curvas aparecem após a primeira época concluída.', 'muted')
        self.history_title.setContentsMargins(16, 10, 16, 0)
        self.history_plot = figuras.HistoryPlot()
        self.history_layout.addWidget(self.history_title)
        self.history_layout.addWidget(self.history_plot)
        self.content.addWidget(self.history_host)
        logs, logs_layout = card('Saída do experimento', 'O log pode ser consultado sem interromper o acompanhamento.')
        self.log = LogView()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(800)
        self.log.setFixedHeight(190)
        self.log.setPlaceholderText('A saída do experimento aparecerá aqui.')
        logs_layout.addWidget(self.log)
        self.content.addWidget(logs)
        self.content.addStretch()
        self.use_section(2)
        models, layout = card('Inspecionar as arquiteturas', 'A contagem é calculada construindo cada rede na CPU, fora da thread da janela.')
        self.frames = NumberBox()
        self.frames.setRange(16, 100000)
        self.frames.setValue(300)
        self.frames.setPrefix('Quadros: ')
        self.inspect_button = button('Calcular parâmetros', self.inspect)
        layout.addWidget(row(self.frames, self.inspect_button))
        self.model_shape = label('', 'muted')
        layout.addWidget(self.model_shape)
        self.model_table = table(['Arquitetura', 'Eixo', 'Agregação', 'Parâmetros'],
                                [[name, *detail, 'Sob demanda'] for name, detail in ARCHITECTURES.items()])
        layout.addWidget(self.model_table)
        layout.addWidget(label('O eixo cepstral ordena bases da DCT. Atenção com viés posicional e Flatten '
                              'não torna a rede completa invariante à ordem dos coeficientes.', 'muted'))
        self.content.addWidget(models)
        self.content.addStretch()
        self.use_section(0)
        self.profile.currentIndexChanged.connect(self.profile_changed)
        self.profile_changed()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(700)

    def set_context(self, ctx):
        """Atualiza a referência para inspecionar redes sem mudar o treino em andamento.

        Args:
            ctx: Seleção usada para a forma de entrada ilustrativa e número de classes.
        """
        self.context = ctx
        self.model_shape.setText(f'Entrada ilustrativa: {ctx.settings.num_mfccs if ctx.settings else "?"} coeficientes '
                                f'× quadros escolhidos · {len(ctx.inventory)} classes. '
                                'O comprimento efetivo do treino é calculado na página Tensores.')

    def profile_changed(self):
        """Expõe os limites e o destino do perfil antes de iniciar o experimento.

        A consulta de espaço usa o ancestral existente do destino, que ainda pode
        não ter sido criado. Cross-microfone restringe o formulário a uma partição.
        """
        settings = self.profiles.get(self.profile.currentData())
        if not settings:
            self.start_button.setEnabled(False)
            return
        self.folds.setMaximum(1 if settings.cross_mic else settings.num_folds)
        self.epochs.setValue(settings.epochs)
        self.destination.setText(f'Destino: {settings.models_path}')
        ancestor = settings.models_path
        while not ancestor.exists():
            ancestor = ancestor.parent
        free = shutil.disk_usage(ancestor).free / 1024 ** 3
        self.disk.setText(f'{free:.2f} GiB livres · Modelos e relatórios consomem espaço a cada partição. '
                         'Resultados existentes exigirão confirmação de sobrescrita.')

    def start(self):
        """Valida as condições locais antes de entregar o treino ao gerenciador.

        A presença de artefatos no destino exige confirmação porque as partições
        escolhidas podem ser sobrescritas. Features ausentes, espaço insuficiente
        ou outro treino ativo são relatados sem iniciar um novo processo.
        """
        settings = self.profiles.get(self.profile.currentData())
        if not settings:
            return
        if self.manager.job and self.manager.job.process.poll() is None:
            self.status.setText('Já existe um treino ativo nesta janela.')
            return
        missing = [str(p) for p in feature_paths(settings) if p is None or not p.is_dir()]
        if missing:
            self.status.setText('Features indisponíveis: ' + ', '.join(missing))
            return
        output = settings.models_path
        ancestor = output
        while not ancestor.exists():
            ancestor = ancestor.parent
        if shutil.disk_usage(ancestor).free < 100 * 1024 ** 2:
            self.status.setText('Menos de 100 MiB livres no destino. Libere espaço antes de treinar.')
            return
        if output.exists() and any(output.iterdir()):
            answer = QMessageBox.question(self, 'Sobrescrever resultados?',
                f'O destino já contém artefatos:\n{output}\n\n'
                f'Arquitetura: {self.architecture.currentText()}\nPartições: 1 a {self.folds.value()}\n\n'
                'Os arquivos dessas partições e o resumo da arquitetura serão sobrescritos. '
                'Outras partições antigas permanecerão. Continuar?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                self.status.setText('Treino não iniciado. Os resultados existentes foram preservados.')
                return
        try:
            command, env = training_command(self.profile.currentData(), self.architecture.currentText(),
                                            self.folds.value(), self.epochs.value())
            job = self.manager.start(command, env, output, self.architecture.currentText())
        except (OSError, ValueError, RuntimeError) as error:
            self.status.setText(str(error))
            return
        self.last_progress, self.last_log = None, ''
        self.clear_history()
        self.status.setText(f'Treinamento iniciado · PID {job.process.pid}')
        self.sections.setCurrentIndex(0)
        self.poll()

    def stop(self):
        """Solicita interrupção sem esperar o término na thread gráfica.

        O timer continuará consultando o processo até observar sua saída.
        """
        if self.manager.job:
            self.manager.job.stop()
            self.poll()

    def clear_history(self):
        """Limpa as séries mantendo o canvas, a altura do cartão e a posição de leitura."""
        self.history_title.setText('As curvas aparecem após a primeira época concluída.')
        self.history_plot.update_history({}, reset_view=True)

    def poll(self):
        """Atualiza widgets a partir de uma cópia do log e do progresso recente.

        Executado pelo timer na thread da janela, preserva a posição de leitura
        do usuário e ignora progresso anterior ao início do processo. O JSON é
        consultado fora do cache; leituras parciais podem ser repetidas no próximo
        ciclo sem substituir o último gráfico válido.
        """
        job = self.manager.job
        if not job:
            return
        code = job.process.poll()
        active = code is None
        self.start_button.setEnabled(not active)
        self.stop_button.setEnabled(active and not job.stopping)
        state = 'Interrompendo…' if active and job.stopping else 'Em execução' if active else 'Concluído' if code == 0 else 'Interrompido' if job.stopping else f'Falhou (código {code})'
        self.status.setText(f'{state} · {job.architecture} · PID {job.process.pid} · {job.output.name}')
        log = job.lines()
        if log != self.last_log:
            scroll = self.log.verticalScrollBar()
            at_end = scroll.value() >= scroll.maximum() - 8
            position = scroll.value()
            self.log.setPlainText(log)
            scroll.setValue(scroll.maximum() if at_end else position)
            self.last_log = log
        try:
            files = [(p.stat().st_mtime_ns, p) for p in (job.output / job.architecture).glob('particao*/progresso.json')
                     if p.stat().st_mtime >= job.started]
            latest = max(files) if files else None
            if latest and latest != self.last_progress:
                progress = dados.read_json(latest[1], live=True)
                if progress:
                    self.last_progress = latest
                    self.history_title.setText(f'Partição {progress.get("particao")} · época {progress.get("epoca")} concluída')
                    self.history_plot.update_history(progress.get('historico', {}))
        except OSError:
            pass  # O arquivo pode ser substituído atomicamente durante o timer.

    def inspect(self):
        """Captura a forma escolhida antes de construir as redes em segundo plano.

        A resposta identifica a seleção do momento do cálculo, pois o usuário
        pode continuar navegando enquanto a contagem é obtida na CPU.
        """
        if not self.context:
            return
        ctx = self.context
        shape = (ctx.settings.num_mfccs if ctx.settings else dados.mfcc(ctx.sample / 'mfccs.npy').shape[0], self.frames.value())
        classes = len(ctx.inventory)
        rate = ctx.settings.learning_rate if ctx.settings else .001
        self.inspect_button.setEnabled(False)
        self.inspect_button.setText('Construindo na CPU…')

        def finish(result, error):
            self.inspect_button.setEnabled(True)
            self.inspect_button.setText('Calcular parâmetros')
            if error:
                self.model_shape.setText(error)
                return
            self.model_shape.setText(f'Contagem calculada para entrada {shape}, {classes} classes (seleção no momento do cálculo).')
            from PySide6.QtWidgets import QTableWidgetItem
            counts = dict(result)
            self.model_table.setSortingEnabled(False)
            for r in range(self.model_table.rowCount()):
                name = self.model_table.item(r, 0).text()
                self.model_table.setItem(r, 3, QTableWidgetItem(f'{counts.get(name, 0):,}'.replace(',', '.')))
            self.model_table.setSortingEnabled(True)
        self.tasks.submit(lambda: model_parameters(shape, classes, rate), finish)


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
    section_names = ('Visão geral', 'Perda e acurácia', 'Matriz e erros', 'Comparar protocolos')

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
        self.rebuild()
        if data['errors']:
            self.notice(f'{len(data["errors"])} arquivo(s) de métricas ilegível(is) ou incompleto(s).', True)

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
            self.content.addWidget(table(['Partição', 'Acurácia', 'F1 macro', 'Acaso', 'Vezes o acaso', 'Amostras'],
                [[r['particao'], f'{r["acuracia"]:.2%}', f'{r["f1_macro"]:.4f}', f'{r["acaso"]:.2%}',
                  f'{r["acuracia"] / r["acaso"]:.1f}×', r['num_amostras_teste']] for r in selected]))
        else:
            self.notice('Ainda não há resultados para este experimento, arquitetura e número de classes.')
        if selected:
            self.content.addWidget(row(button('Abrir perda e acurácia', lambda: self.sections.setCurrentIndex(1), primary=True),
                                       button('Inspecionar os erros', lambda: self.sections.setCurrentIndex(2))))
        self.notice('Uma partição não permite estimar desvio. Confira também VAD, truncamento, '
                    'features e configuração de treino. Cross-microfone compartilha sessão e '
                    'enunciados; não demonstra voz isolada.', True)
        self.content.addStretch()
        self.use_section(3)
        comparisons = []
        for exp in sorted({r['experimento'] for r in records}):
            values = [r for r in records if r['experimento'] == exp]
            accuracies = [r['acuracia'] for r in values]
            comparisons.append(dict(experiment=exp, mean=np.mean(accuracies),
                                    std=np.std(accuracies) if len(values) > 1 else None,
                                    chance=values[0]['acaso'], folds=len(values)))
        if comparisons:
            widget, layout = card('A mesma arquitetura, diferentes condições', f'{architecture} · {count} classes')
            layout.addWidget(figuras.comparison(comparisons))
            layout.addWidget(table(['Experimento', 'Acurácia', 'Desvio (p.p.)', 'Partições'],
                [[r['experiment'], f'{r["mean"]:.2%}', f'{r["std"] * 100:.2f}' if r['std'] is not None else 'Não estimável', r['folds']]
                 for r in comparisons]))
            self.content.addWidget(widget)
        if not comparisons:
            self.notice('Ainda não há resultados comparáveis para esta arquitetura e número de classes.')
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
                self.curve_message.setText('Histórico numérico não disponível para esta partição.')
            for title, value, path in images:
                if not value.isNull():
                    self.detail_layout.addWidget(ImagePanel(title, value, path))
                else:
                    self.detail_layout.addWidget(label(title + ': artefato indisponível.', 'notice'))
            if not (directory / 'modelo.keras').is_file():
                self.detail_layout.addWidget(label('Métricas disponíveis; o arquivo do modelo não está presente nesta partição.', 'muted'))
        self.tasks.submit(load, finish)
