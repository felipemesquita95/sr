"""Páginas de evidência para a defesa, compostas apenas de artefatos locais."""
from pathlib import Path

from PySide6.QtCore import Signal

from ui import dados, figuras
from ui.componentes import button, card, label, table
from ui.paginas import Page, SectionedPage


def percentual(value):
    """Formata uma medida do artefato sem criar valor substituto.

    Args:
        value: Medida em escala unitária ou percentual.

    Returns:
        Texto para apresentação.
    """
    return f'{value * 100:.2f}%' if abs(value) <= 1 else f'{value:.2f} pp'


def ressalva(documento, heading):
    """Extrai uma seção documental para mantê-la junto da evidência.

    Args:
        documento: Texto Markdown de origem.
        heading: Título cuja seção será apresentada.

    Returns:
        Trecho documental ou uma explicação de ausência.
    """
    start = documento.find(heading)
    if start < 0:
        return 'Ressalva não disponível: consulte docs/resultados.md e docs/limitacoes.md.'
    end = documento.find('\n## ', start + len(heading))
    return documento[start:end if end >= 0 else None].strip()


def secao_contem(documento, texto):
    """Localiza uma seção Markdown pelo texto de seu cabeçalho.

    Args:
        documento: Texto Markdown de origem.
        texto: Fragmento distintivo do cabeçalho.

    Returns:
        Seção encontrada ou vazio.
    """
    lines = documento.splitlines()
    start = next((index for index, line in enumerate(lines) if line.startswith('## ') and texto in line), None)
    if start is None:
        return ''
    end = next((index for index, line in enumerate(lines[start + 1:], start + 1) if line.startswith('## ')), len(lines))
    return '\n'.join(lines[start:end]).strip()


class DefensePage(Page):
    """Abre a defesa pelo encadeamento argumentativo do documento."""
    navigate_stage = Signal(str, str)

    def display(self, data, ctx):
        """Monta o roteiro e direciona cada passo à evidência navegável."""
        self.reset_body()
        roteiro = [
            ('Resultado convencional', 'Resultados persistidos e matriz pareada.', 'Matrizes'),
            ('O silêncio prediz', 'Diagnóstico de travessia por recorte do sinal.', 'Silêncio e travessia'),
            ('Troca de microfone', 'Matriz de transferência e protocolos auxiliares.', 'Protocolos auxiliares'),
            ('Não é só transdutor', 'Recuperação após transformação afim.', 'Sessão ou transdutor'),
            ('Não era o teto', 'Referência estática contra as sementes da CNN.', 'Matrizes'),
        ]
        self.notice('Roteiro da defesa, na ordem de docs/resultados.md. Cada botão abre a evidência; números só aparecem após a leitura dos arquivos em runs/.')
        for titulo, explicacao, secao in roteiro:
            widget, layout = card(titulo, explicacao)
            layout.addWidget(button('Abrir evidência', lambda target=secao: self.navigate_stage.emit('evidencias', target), primary=True))
            self.content.addWidget(widget)
        self.notice(ressalva(data['resultados'], '## 9. O que este documento **não** afirma'), True)
        self.ready()


class EvidencePage(SectionedPage):
    """Agrupa relatórios diagnósticos em abas para consulta durante a defesa."""
    section_names = ('Matrizes', 'Silêncio e travessia', 'Sessão ou transdutor',
                     'Estrutura dos erros', 'Controles', 'Protocolos auxiliares', 'BrSD')

    def _proveniencia(self, path, report):
        self.content.addWidget(label(f'Procedência: {path}', 'muted'))
        parent = path.parent
        metadata = []
        for name in ('configuracao.json', 'divisao.json'):
            candidate = next((p / name for p in (parent, *parent.parents) if (p / name).is_file()), None)
            if candidate:
                value = dados.read_json(candidate)
                metadata.append(f'{candidate}: {", ".join(value.keys())}')
        if metadata:
            self.notice('Como foi feito (metadados persistidos): ' + ' · '.join(metadata))
        else:
            self.notice('Como foi feito: configuração/divisão não persistidas junto deste relatório; o comando exato não pode ser inferido sem inventá-lo. Gere novamente o experimento para registrar configuracao.json e divisao.json.', True)
        if self.reproducao:
            self.notice('Comando de reprodução documentado:\n' + self.reproducao)

    def _missing(self, paths):
        for path in paths:
            self.notice(f'Artefato ausente: {path}. Gere o diagnóstico correspondente para exibir esta evidência.', True)

    def display(self, data, ctx):
        """Desenha somente séries existentes e transforma ausências em avisos."""
        self.reset_sections()
        reports = data['diagnosticos']['relatorios']
        resultados = data['resultados']
        limitacoes = data['limitacoes']
        self.reproducao = secao_contem(resultados, 'Reprodução')

        self.use_section(0)
        self.notice('Pergunta: quanto a identificação muda entre a trilha de treino e a outra trilha?')
        matrix_series = []
        for path, report in data['diagnosticos']['matrizes']:
            ajustes = report.get('ajustes', [])
            perdas = [a['perda_acuracia_pp'] for a in ajustes if 'perda_acuracia_pp' in a]
            if perdas:
                matrix_series.append((path.parent.name, perdas))
                self._proveniencia(path, report)
        if matrix_series:
            self.content.addWidget(figuras.distribuicoes(matrix_series, 'Perda pareada por ajuste', 'Perda (pp)'))
        else:
            self.notice('Nenhuma matriz de transferência legível.', True)
        self.notice(ressalva(resultados, '## 5.2 A média não descreve nenhum locutor'), True)
        if 'referencia' in reports:
            path, report = reports['referencia']
            losses = [item['perda_acuracia_pp'] for item in report.get('ajustes', []) if 'perda_acuracia_pp' in item]
            self.notice('Pergunta: a representação estática linear altera o patamar obtido pelas redes?')
            if losses:
                self.content.addWidget(figuras.distribuicoes([('referência estática', losses)], 'Perda da referência estática', 'Perda (pp)'))
            self._proveniencia(path, report)
            self.notice(ressalva(resultados, '### 2.5 Uma referência estática linear supera a CNN nesta divisão'), True)

        self.use_section(1)
        self.notice('Pergunta: o que silêncio, fala e sinal completo predizem dentro e através da trilha?')
        if 'travessia' in reports:
            path, report = reports['travessia']
            rows = [(f'{condition} · dentro', value['dentro']) for condition, value in report.get('condicoes', {}).items()]
            rows += [(f'{condition} · travessia', value['travessia']) for condition, value in report.get('condicoes', {}).items()]
            if rows:
                self.content.addWidget(figuras.barras(rows, 'Diagnóstico de travessia', 'Acurácia (%)'))
            self._proveniencia(path, report)
        else:
            self._missing([dados.RUNS / 'models/vctk_cross_mic/diagnostico_travessia/travessia_canal.json'])
        self.notice(ressalva(resultados, '## 3. Diagnóstico de canal'), True)

        self.use_section(2)
        self.notice('Pergunta: a queda ao trocar o microfone separa sessão de transdutor?')
        if 'sessao' in reports:
            path, report = reports['sessao']
            rows = [(f'{condition} · {field}', values[field]) for condition, values in report.items()
                    for field in ('mesma_trilha', 'outra_trilha', 'outra_transformada') if field in values]
            if rows:
                self.content.addWidget(figuras.barras(rows, 'Recuperação por transformação afim', 'Acurácia (%)'))
            self._proveniencia(path, report)
        else:
            self._missing([dados.RUNS / 'models/sessao_ou_transdutor/sessao_ou_transdutor.json'])
        self.notice(ressalva(resultados, '## 6. O protocolo cross-microfone não remove o confundidor de sessão'), True)

        self.use_section(3)
        self.notice('Pergunta: as confusões excedem o esperado por gênero e sotaque, e como a revocação varia por locutor?')
        if 'erros' in reports:
            path, report = reports['erros']
            rows = [(f'{origin} · {field}', value['excesso']) for origin, item in report.get('resultados', {}).items()
                    for field, value in item.get('campos', {}).items() if 'excesso' in value]
            if rows:
                self.content.addWidget(figuras.barras(rows, 'Excesso de confusão', 'Excesso (pp)'))
            recalls = [(origin, values) for origin, values in report.get('revocacao_por_locutor', {}).items()]
            if recalls:
                self.content.addWidget(figuras.distribuicoes(recalls, 'Revocação por locutor'))
            self._proveniencia(path, report)
        else:
            self._missing([dados.RUNS / 'models/vctk_transfer_matrix/estrutura_erros/estrutura_erros.json'])
        self.notice(ressalva(limitacoes, '## 2. Confundidor de sessão'), True)

        self.use_section(4)
        self.notice('Pergunta: os controles negativos e de alinhamento descartam explicações instrumentais específicas?')
        controls = [(path.parent.parent.name, report['acuracia_media']) for path, report in data['diagnosticos']['resumos']
                    if 'controle_permutacao' in str(path) and 'acuracia_media' in report]
        if controls:
            self.content.addWidget(figuras.barras(controls, 'Controles de permutação', 'Acurácia (%)'))
            for path, report in data['diagnosticos']['resumos']:
                if 'controle_permutacao' in str(path):
                    self._proveniencia(path, report)
        if 'alinhamento' in reports:
            path, report = reports['alinhamento']
            rows = [(key, value) for key, value in report.get('recuperacao', {}).items() if isinstance(value, (int, float))]
            if rows:
                self.content.addWidget(figuras.barras(rows, 'Verificação de alinhamento'))
            self._proveniencia(path, report)
        self.notice(ressalva(resultados, '## 7. Controles'), True)

        self.use_section(5)
        self.notice('Pergunta: a conclusão permanece ao inverter, separar enunciados ou remover VAD?')
        auxiliary = [(path.parent.parent.name + ' · ' + report.get('arquitetura', path.parent.name), report['acuracia_media'])
                     for path, report in data['diagnosticos']['resumos']
                     if 'vctk_cross_mic' in str(path) and 'acuracia_media' in report]
        if auxiliary:
            self.content.addWidget(figuras.barras(auxiliary, 'Protocolos cross-microfone', 'Acurácia (%)'))
            for path, report in data['diagnosticos']['resumos']:
                if 'vctk_cross_mic' in str(path):
                    self._proveniencia(path, report)
        else:
            self.notice('Protocolos auxiliares não encontrados.', True)
        self.notice(ressalva(limitacoes, '## 7. Alinhamento por índice'), True)

        self.use_section(6)
        self.notice('Pergunta: o segundo corpus permite atribuir acurácia à voz, e não ao dispositivo?')
        brsd = [(report.get('arquitetura', path.parent.name), report['acuracia_media']) for path, report in data['diagnosticos']['resumos']
                if '/brsd/' in str(path) and 'acuracia_media' in report]
        if brsd:
            self.content.addWidget(figuras.barras(brsd, 'Segundo corpus', 'Acurácia (%)'))
            for path, report in data['diagnosticos']['resumos']:
                if '/brsd/' in str(path):
                    self._proveniencia(path, report)
        else:
            self.notice('Resumos do BrSD não encontrados.', True)
        self.notice(ressalva(limitacoes, '## 1. Confundidor de canal'), True)

        for index in range(len(self.areas)):
            self.use_section(index)
            self._missing(data['diagnosticos']['ausentes'])
        self.use_section(self.sections.currentIndex())
        self.ready()
