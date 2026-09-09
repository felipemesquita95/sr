"""Apresentação persistente das etapas e da reconstrução dos tensores.

Os dados chegam prontos dos carregadores; apenas estes métodos, chamados
na thread da janela, criam widgets e canvas. Preservar a página permite
reutilizar gráficos quando a navegação volta à mesma seleção.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QHBoxLayout, QLayout, QLineEdit, QProgressBar,
    QScrollArea, QTabWidget, QVBoxLayout, QWidget)

from ui import dados, figuras
from ui.componentes import ChoiceBox, NumberBox, ImagePanel, button, card, label, row, stat, table


class Page(QWidget):
    """Mantém controles e estado de carregamento enquanto o conteúdo pode mudar.

    A janela controla ``loaded_key`` e só solicita novo desenho quando seleção
    ou parâmetros mudam. O sinal ``reload`` pede essa invalidação sem acoplar
    a página à implementação da janela.

    Args:
        stage: Identificador da etapa no catálogo.
        title: Título apresentado na página.
        description: Explicação do estágio para o usuário.
    """
    reload = Signal()
    select_sample = Signal(int, int)

    def __init__(self, stage, title, description):
        super().__init__()
        self.stage = stage
        self.loaded_key = None
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(28, 22, 28, 16)
        self.layout.setSpacing(12)
        self.layout.addWidget(label('EXPLORAR O PIPELINE', 'eyebrow'))
        self.layout.addWidget(label(title, 'pageTitle'))
        self.layout.addWidget(label(description, 'description'))
        self.controls = QHBoxLayout()
        self.layout.addLayout(self.controls)
        if stage in ('sinal', 'vad', 'filtragem', 'preenfase'):
            self.figure_speaker = None
            self.figure_samples = ChoiceBox()
            self.figure_samples.setObjectName('availableFigures')
            self.figure_samples.setPlaceholderText('Escolha um enunciado com figura')
            self.figure_samples.setMinimumWidth(240)
            self.controls.addWidget(label('Figuras salvas deste locutor:', 'muted'))
            self.controls.addWidget(self.figure_samples)
            self.controls.addStretch()
            self.figure_samples.currentIndexChanged.connect(self.open_figure_sample)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(4)
        self.progress.hide()
        self.layout.addWidget(self.progress)
        self.message = label('', 'notice')
        self.message.hide()
        self.layout.addWidget(self.message)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.verticalScrollBar().setSingleStep(32)
        self.layout.addWidget(self.scroll, 1)
        self.reset_body()

    def open_figure_sample(self, index):
        if index >= 0 and self.figure_speaker is not None:
            self.select_sample.emit(self.figure_speaker, self.figure_samples.itemData(index))

    def reset_body(self):
        """Substitui o conteúdo preservando os controles permanentes da página.

        A destruição adiada respeita o ciclo de eventos do Qt, inclusive quando
        a troca parte de um sinal de um dos widgets antigos.
        """
        old = self.scroll.takeWidget()
        if old:
            old.deleteLater()
        self.body = QWidget()
        self.content = QVBoxLayout(self.body)
        self.content.setContentsMargins(0, 4, 4, 12)
        self.content.setSpacing(16)
        self.content.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        self.scroll.setWidget(self.body)

    def loading(self, text='Carregando os artefatos…'):
        """Oculta o conteúdo anterior para não atribuí-lo à nova seleção.

        Args:
            text: Mensagem apresentada durante a tarefa em segundo plano.
        """
        if hasattr(self, 'figure_samples'):
            self.figure_samples.setEnabled(False)
        self.progress.show()
        self.message.setText(text)
        self.message.show()
        self.scroll.hide()

    def error(self, text):
        """Mantém a falha visível sem deixar gráficos antigos parecerem atuais.

        Args:
            text: Motivo da falha de leitura ou apresentação.
        """
        self.progress.hide()
        self.message.setText(text)
        self.message.show()
        self.scroll.hide()

    def ready(self):
        self.progress.hide()
        self.message.hide()
        self.scroll.show()

    def notice(self, text, warning=False):
        self.content.addWidget(label(text, 'warning' if warning else 'notice'))

    def add_images(self, images, ctx):
        """Explica a ausência de cada PNG em vez de reconstruir um estágio sem áudio.

        Args:
            images: Tuplas de título, QImage e caminho de origem.
            ctx: Seleção com inventário para indicar enunciados que têm figuras.
        """
        for title, image, path in images:
            if not image.isNull():
                self.content.addWidget(ImagePanel(f'{title} · {ctx.caption}', image, path))
            else:
                available = ', '.join(str(u) for u, yes in ctx.inventory[ctx.speaker].items() if yes)
                self.notice(f'{title}: figura não disponível para esta seleção. '
                            f'Enunciados com figuras neste locutor: {available or "nenhum"}. '
                            'Esta etapa depende da figura salva ou do áudio original; MFCCs não permitem recuperar o sinal bruto.')
        if any(image.isNull() for _, image, _ in images):
            if self.stage == 'sinal':
                self.notice('O áudio original desta gravação não foi encontrado no caminho configurado. '
                            'Escolha uma figura disponível abaixo ou explore os coeficientes na aba MFCC.', True)
            choices = [(s, u) for s, values in ctx.inventory.items() for u, yes in values.items() if yes]
            choices.sort(key=lambda pair: (pair[0] != ctx.speaker, pair))
            if choices:
                s, u = choices[0]
                self.content.addWidget(button(f'Abrir figura disponível: locutor {s}, enunciado {u:03d}',
                                              lambda: self.select_sample.emit(s, u)))

    def display(self, data, ctx):
        """Converte os dados da etapa em conteúdo da seleção já validada pela janela.

        A ausência de artefatos ou de metadados aparece em avisos. Campos do perfil
        atual são apresentados como configuração, não como medições históricas.

        Args:
            data: Artefatos preparados pelo carregador da etapa.
            ctx: Seleção à qual esses artefatos correspondem.
        """
        self.reset_body()
        stage = self.stage
        if hasattr(self, 'figure_samples'):
            self.figure_speaker = ctx.speaker
            self.figure_samples.blockSignals(True)
            self.figure_samples.clear()
            for utterance in data.get('available_figures', []):
                self.figure_samples.addItem(f'Enunciado {utterance:03d}', utterance)
            self.figure_samples.setCurrentIndex(self.figure_samples.findData(ctx.utterance))
            self.figure_samples.setEnabled(self.figure_samples.count() > 0)
            self.figure_samples.blockSignals(False)
        settings = ctx.settings
        if stage == 'corpus':
            self.overview(ctx)
        elif stage == 'sinal':
            if 'raw' in data:
                raw = data['raw']
                self.content.addWidget(row(stat('Taxa original', f'{raw["rate"] / 1000:g} kHz', 'Lida do arquivo de áudio'),
                                           stat('Duração', f'{raw["duration"]:.2f} s'),
                                           stat('Amostras', f'{raw["samples"]:,}'.replace(',', '.'))))
                self.notice(f'Prévia do áudio original: {raw["source"]}. Sem alterar features ou resultados. '
                            'O gráfico resume os extremos do sinal e os picos do espectro para manter a navegação leve.')
                self.content.addWidget(figuras.raw_signal(raw, ctx.caption))
            else:
                rate = f'{settings.source_sampling_rate / 1000:g} kHz' if settings else 'Não registrada'
                self.content.addWidget(row(stat('Taxa de leitura', rate, 'Declarada no perfil atual'),
                                           stat('Gravação', str(ctx.utterance), ctx.names.get(ctx.speaker, f'Locutor {ctx.speaker}'))))
                self.notice('Estas figuras foram salvas no processamento. A duração exata e o número de amostras '
                            'originais não foram persistidos; MFCCs não recuperam esses valores com precisão.')
        elif stage == 'vad':
            if settings and not settings.enable_vad:
                self.notice('VAD desativado no perfil desta trilha. As pausas permanecem nas features.')
            elif settings:
                self.content.addWidget(stat('Limiar relativo ao pico', f'{settings.vad_top_db} dB'))
            self.notice('Cada trilha recebe seu próprio corte de energia. Microfones com níveis e ruídos '
                        'diferentes podem produzir quadros que já não correspondem ao mesmo instante.', True)
            effect = data.get('alignment', {}).get('efeito_vad', {})
            if effect:
                self.content.addWidget(row(stat('Duração idêntica', f'{effect["duracao_identica"]} / {effect["pares"]}'),
                                           stat('Divergência mediana', f'{effect["divergencia_mediana"]:.1%}')))
        elif stage == 'filtragem' and settings:
            self.content.addWidget(row(stat('Taxa de origem', f'{settings.source_sampling_rate / 1000:g} kHz'),
                                       stat('Taxa alvo', f'{settings.target_sampling_rate / 1000:g} kHz'),
                                       stat('Nyquist alvo', f'{settings.target_sampling_rate / 2000:g} kHz')))
        elif stage == 'preenfase':
            self.content.addWidget(stat('Coeficiente de pré-ênfase', str(settings.pre_emphasis_coef) if settings else 'Não registrado'))
            self.notice('Os espectros existem como imagens. Compare as escalas ao inspecionar o ganho; '
                        'não foram salvas curvas numéricas para uma sobreposição fiel.')
        elif stage == 'mfcc':
            matrix = data['matrices'][0]
            self.content.addWidget(row(stat('Coeficientes', str(matrix.shape[0])), stat('Quadros', str(matrix.shape[1])),
                                       stat('Janela / salto', f'{settings.frame_duration_ms:g} / {settings.frame_duration_ms / 2:g} ms' if settings else 'Não registrado')))
            self.content.addWidget(figuras.mfcc(data['matrices'], data['titles']))
            self.notice('A escala de cor é compartilhada entre as trilhas. Com VAD independente, '
                        'quadros na mesma posição podem corresponder a instantes diferentes.')
        elif stage == 'assinatura':
            self.channel(data)
        elif stage == 'protocolos':
            self.protocols()
        self.add_images(data.get('images', []), ctx)
        self.content.addStretch()
        self.ready()

    def overview(self, ctx):
        """Relaciona a cobertura da trilha à numeração compartilhada dos locutores.

        As exclusões do manifesto ficam visíveis porque deslocar índices entre
        microfones invalidaria a comparação dos resultados.

        Args:
            ctx: Seleção com inventário não vazio, nomes e manifesto da trilha.
        """
        counts = [len(samples) for samples in ctx.inventory.values()]
        self.content.addWidget(row(stat('Locutores', str(len(counts)), 'Identidades conhecidas'),
                                   stat('Gravações', f'{sum(counts):,}'.replace(',', '.'), ctx.track.name),
                                   stat('Acaso uniforme', f'{100 / len(counts):.2f}%', '1 / número de locutores')))
        widget, layout = card('Uma gravação. Todas as etapas.',
                              'Escolha um locutor e um enunciado no topo. Percorra o sinal pela navegação lateral; '
                              'a seleção acompanha você em todo o aplicativo.')
        layout.addWidget(label('← →   Trocar enunciado     •     Alt + ↑ ↓   Trocar etapa     •     Ctrl + R   Atualizar', 'muted'))
        self.content.addWidget(widget)
        excluded = ctx.manifest.get('excluidos', {}).get('locutores_sem_todas_as_trilhas', [])
        if excluded:
            self.notice('Interseção das trilhas: ' + ', '.join(excluded) + ' foram excluídos. '
                        'A numeração compartilhada evita atribuir rótulos diferentes à mesma pessoa entre microfones.', True)
        widget, layout = card('Locutores da trilha', f'{min(counts)} a {max(counts)} enunciados por locutor')
        search = QLineEdit()
        search.setPlaceholderText('Buscar pelo nome no corpus ou índice…')
        listing = table(['Índice', 'Nome no corpus', 'Enunciados', 'Com figuras'],
                        [[s, ctx.names.get(s, 'Não registrado'), len(v), sum(v.values())] for s, v in ctx.inventory.items()], 280)

        def filter_rows(text):
            for r in range(listing.rowCount()):
                content = ' '.join(listing.item(r, c).text() for c in (0, 1))
                listing.setRowHidden(r, text.casefold() not in content.casefold())
        search.textChanged.connect(filter_rows)
        layout.addWidget(search)
        layout.addWidget(listing)
        self.content.addWidget(widget)

    def channel(self, data):
        """Contextualiza as assinaturas sem atribuir causalmente o acerto ao canal.

        Baixa energia pode incluir respiração ou fala fraca. A tabela de travessia
        vem do relatório persistido e não substitui condições ausentes na seleção.

        Args:
            data: Assinaturas disponíveis e relatório de travessia, quando existente.
        """
        values = data['signatures']
        if values:
            self.content.addWidget(figuras.signatures(values))
            absent = set(('silence', 'speech', 'full')) - values.keys()
            if absent:
                self.notice('Condições não persistidas nesta gravação: ' + ', '.join(sorted(absent)))
        else:
            self.notice('Assinaturas não disponíveis para esta gravação; o áudio bruto seria necessário para extraí-las.')
        self.notice('“Silêncio” é um recorte de baixa energia e pode conter respiração ou fala fraca. '
                    'Acerto acima do acaso indica pistas residuais; não mede uma parcela causal de canal ou voz.', True)
        report = data['transfer']
        if report:
            widget, layout = card('Travessia de canal · VCTK, mic1 → mic2',
                                  f'{report["num_locutores"]} locutores · acaso {report["acaso"]:.2%} · classificador linear')
            layout.addWidget(table(['Condição', 'Dentro da trilha', 'Outro microfone', 'Pares'],
                              [[k, f'{v["dentro"]:.2%}', f'{v["travessia"]:.2%}', int(v['pares'])]
                               for k, v in report['condicoes'].items()]))
            self.content.addWidget(widget)
        else:
            self.notice('Diagnóstico de travessia ainda não disponível.')

    def protocols(self):
        """Expõe o que cada comparação controla e o que permanece compartilhado.

        Os avisos evitam interpretar transferência entre transdutores como
        isolamento causal da voz, ou um controle de permutação como prova universal.
        """
        for name, route, explanation in [
            ('Intra-microfone', 'mic1 → mic1 · enunciados separados', 'Voz e pistas de sessão permanecem associadas. Mede identificação nas condições conhecidas.'),
            ('Cross-microfone', 'mic1 → mic2 · gravações simultâneas', 'Mede transferência entre transdutores. A sessão e os enunciados são compartilhados; não isola causalmente a voz.'),
            ('Multi-microfone', 'ambos → ambos · agrupamento por enunciado', 'Expõe cada locutor aos dois microfones, mas mantém a sessão compartilhada entre treino e teste.'),
            ('Permutação', 'rótulos redistribuídos entre gravações', 'Controle negativo. Desempenho próximo ao acaso é compatível com ausência de sinal aprendível, sem provar ausência de todo vazamento.')]:
            widget, layout = card(name, route)
            layout.addWidget(label(explanation))
            self.content.addWidget(widget)
        self.notice('Próximo controle proposto: separar também os enunciados no cross-microfone, '
                    'inverter o sentido da travessia e repetir com diferentes sementes de treino.')


class SectionedPage(Page):
    """Seções com rolagens independentes e cabeçalho sempre acessível."""
    section_names = ()

    def __init__(self, stage, title, description):
        super().__init__(stage, title, description)
        self.layout.removeWidget(self.scroll)
        self.scroll.deleteLater()
        self.sections = QTabWidget()
        self.sections.setObjectName('experimentSections')
        self.sections.setDocumentMode(True)
        self.areas = []
        self.section_bodies = []
        self.section_layouts = []
        for name in self.section_names:
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            area.verticalScrollBar().setSingleStep(32)
            self.sections.addTab(area, name)
            self.areas.append(area)
            self.section_bodies.append(None)
            self.section_layouts.append(None)
        self.layout.addWidget(self.sections, 1)
        self.reset_sections()

    def reset_sections(self):
        for index, area in enumerate(self.areas):
            self.scroll = area
            Page.reset_body(self)
            self.section_bodies[index] = self.body
            self.section_layouts[index] = self.content
        self.use_section(0)

    def use_section(self, index):
        """Escolhe o destino de composição sem mudar a seção que o usuário está vendo."""
        self.scroll = self.areas[index]
        self.body = self.section_bodies[index]
        self.content = self.section_layouts[index]

    def loading(self, text='Carregando os artefatos…'):
        self.progress.show()
        self.message.setText(text)
        self.message.show()
        self.sections.hide()

    def error(self, text):
        self.progress.hide()
        self.message.setText(text)
        self.message.show()
        self.sections.hide()

    def ready(self):
        self.progress.hide()
        self.message.hide()
        self.sections.show()


class TensorPage(Page):
    """Permite inspecionar uma divisão real sem iniciar um treino.

    Perfil e partição próprios permitem verificar como a seleção participa de
    outros protocolos. Estatísticas caras ficam sob solicitação explícita.

    Args:
        stage: Identificador da etapa no catálogo.
        title: Título apresentado na página.
        description: Explicação do estágio para o usuário.
        profiles: Mapa de arquivos de perfil para configurações.
        tasks: Gerenciador criado na thread da janela para trabalhos em segundo plano.
    """
    def __init__(self, stage, title, description, profiles, tasks):
        super().__init__(stage, title, description)
        self.profiles, self.tasks = profiles, tasks
        self.profile = ChoiceBox()
        for path in profiles:
            self.profile.addItem(path.stem, path)
        self.fold = NumberBox()
        self.fold.setRange(1, 5)
        self.fold.setPrefix('Partição ')
        self.controls.addWidget(self.profile, 1)
        self.controls.addWidget(self.fold)
        self.profile.currentIndexChanged.connect(self.changed)
        self.fold.valueChanged.connect(self.reload)
        self.changed()

    def changed(self):
        """Restringe a escolha de partição ao protocolo antes de pedir nova carga.

        Cross-microfone tem uma divisão única; os demais perfis usam seu número
        configurado de partições.
        """
        settings = self.profiles.get(self.profile.currentData())
        if settings:
            self.fold.setMaximum(1 if settings.cross_mic else settings.num_folds)
        self.reload.emit()

    def display(self, data, ctx):
        """Mostra os destinos reais e distingue alinhamento ilustrativo de participação.

        A seleção pode não pertencer ao perfil escolhido. Nesse caso, sua matriz
        ainda demonstra o alinhamento, mas o aviso impede confundi-la com uma
        amostra efetiva do treino.

        Args:
            data: Artefatos preparados pelo carregador da etapa.
            ctx: Seleção à qual esses artefatos correspondem.
        """
        self.reset_body()
        self.current_data = data
        split, frames, settings = data['split'], data['frames'], data['settings']
        self.content.addWidget(row(*(stat(name, str(len(refs))) for name, refs in split.items())))
        self.notice(f'Comprimento: {frames} quadros, definido pelo máximo do treino '
                    f'e limitado pelo teto {settings.max_frames_cap or "desativado"}. '
                    'Matrizes curtas repetem o conteúdo; matrizes longas são truncadas.')
        self.content.addWidget(figuras.lengths(split, frames, settings.max_frames_cap))
        self.content.addWidget(figuras.mfcc(data['matrices'], ['Antes do alinhamento', 'Após alinhamento']))
        if settings.cross_mic:
            self.notice('Cross-microfone: a mesma frase pode aparecer nas duas trilhas, '
                        'em treino/validação e teste. Sessão e enunciados continuam compartilhados.', True)
        else:
            self.content.addWidget(table(['Grupo de enunciados', 'Destino'],
                [[k, 'Teste' if k == data['fold'] else 'Treino / reserva de validação'] for k in range(1, settings.num_folds + 1)]))
        destinations = [name for name, refs in split.items() for r in refs
                        if r.path == ctx.sample / 'mfccs.npy']
        self.notice('Seleção atual: ' + (', '.join(destinations) if destinations else 'não participa deste perfil; alinhamento ilustrativo.'))
        self.notice('Perfil e features atuais: esta é uma reconstrução da divisão. '
                    'As execuções antigas não salvaram um snapshot completo de configuração.')
        self.normalize_button = button('Calcular média e desvio do treino', self.normalize)
        self.content.addWidget(self.normalize_button)
        self.norm_host = QWidget()
        self.norm_layout = QVBoxLayout(self.norm_host)
        self.content.addWidget(self.norm_host)
        self.content.addStretch()
        self.ready()

    def normalize(self):
        """Calcula estatísticas somente quando solicitadas e descarta respostas obsoletas.

        A identidade dos dados capturados permite ignorar um cálculo que termine
        depois de a página ter passado a mostrar outra divisão.
        """
        data = self.current_data
        self.normalize_button.setEnabled(False)
        self.normalize_button.setText('Calculando em segundo plano…')

        def finish(value, error):
            if self.current_data is not data:
                return
            self.normalize_button.setEnabled(True)
            self.normalize_button.setText('Recalcular estatísticas do treino')
            if error:
                self.norm_layout.addWidget(label(error, 'warning'))
            else:
                mean, std = value
                while self.norm_layout.count():
                    self.norm_layout.takeAt(0).widget().deleteLater()
                self.norm_layout.addWidget(table(['Coeficiente', 'Média', 'Desvio'],
                    [[i + 1, f'{m:.5f}', f'{s:.5f}'] for i, (m, s) in enumerate(zip(mean, std))]))
        self.tasks.submit(lambda: dados.normalization(data['split']['Treino'], data['frames']), finish)
