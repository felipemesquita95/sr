"""Composição dos passos sequenciais com seções persistentes."""
from ui import dados
from ui.componentes import label
from ui.conteudo import CAMPOS, SECOES
from ui.evidencias import DefensePage, EvidencePage
from ui.experimentos import ComparacaoPage, ModelPage, ResultsPage
from ui.paginas import Page, SectionedPage, TensorPage


class PassoPage(SectionedPage):
    """Reúne componentes existentes sem duplicar as evidências.

    Args:
        stage: Identificador do passo.
        title: Título da apresentação.
        description: Descrição do processamento.
        profiles: Perfis disponíveis.
        tasks: Executor de leituras em segundo plano.
    """

    def __init__(self, stage, title, description, profiles, tasks):
        super().__init__(stage, title, description)
        self.parametros = label('', 'notice')
        self.layout.insertWidget(self.layout.indexOf(self.sections), self.parametros)
        self.componentes = {}
        for nome, titulo in SECOES[stage]:
            classe = {'modelo': ModelPage, 'tensores': TensorPage,
                      'resultados': ResultsPage, 'comparacao': ComparacaoPage,
                      'evidencias': EvidencePage, 'limites': DefensePage}.get(nome, Page)
            argumentos = (profiles, tasks) if classe in (TensorPage, ResultsPage) else ()
            pagina = classe(nome, '', '', *argumentos)
            pagina.reload.connect(self.reload.emit)
            pagina.select_sample.connect(self.select_sample.emit)
            self.componentes[nome] = pagina
            self.sections.addTab(pagina, titulo)
        self.section_names = tuple(titulo for _, titulo in SECOES[stage])

    def reset_sections(self):
        """As seções são componentes persistentes e cuidam de sua própria rolagem."""

    def preparar(self, ctx):
        """Associa a divisão ao perfil real da trilha antes da carga.

        Args:
            ctx: Seleção atual com perfil associado ou ausência explícita.

        Returns:
            Partição escolhida, quando aplicável.
        """
        pagina = self.componentes.get('tensores')
        if pagina is None:
            return None
        pagina.fold.blockSignals(True)
        pagina.fold.setEnabled(ctx.settings is not None)
        if ctx.settings:
            pagina.fold.setMaximum(1 if ctx.settings.cross_mic else ctx.settings.num_folds)
        pagina.fold.blockSignals(False)
        return pagina.fold.value()

    def display(self, data, ctx):
        """Apresenta os parâmetros e cada artefato, isolando ausências por seção.

        Args:
            data: Conteúdo carregado de cada seção.
            ctx: Seleção capturada pela tarefa.
        """
        campos = CAMPOS.get(self.stage, ())
        self.parametros.setVisible(bool(campos))
        if campos:
            if ctx.settings is None:
                texto = (f'Sem perfil associado a {ctx.track}. Esperado: perfil em '
                         f'{dados.ROOT / "configs"} que referencie esta trilha.')
            else:
                valores = ' · '.join(f'{campo}: {getattr(ctx.settings, campo)}' for campo in campos)
                texto = (f'{ctx.caption}\n{valores}\nProcedência: {ctx.profile}. '
                         'Configuração efetiva do perfil atual; não substitui um snapshot histórico da extração ou do treino.')
            self.parametros.setText(texto)
        for nome, pagina in self.componentes.items():
            if 'erro' in data[nome]:
                pagina.error(data[nome]['erro'])
            else:
                pagina.display(data[nome], ctx)
        self.ready()

    def loading(self, text='Carregando os artefatos…'):
        """Invalida também os detalhes que ainda chegam de seleções anteriores.

        Args:
            text: Aviso durante a leitura.
        """
        for pagina in self.componentes.values():
            if isinstance(pagina, ResultsPage):
                pagina.detail_number += 1
            if isinstance(pagina, TensorPage):
                pagina.current_data = None
        self.parametros.hide()
        super().loading(text)
