"""Páginas de evidência para a defesa, compostas apenas de artefatos locais."""
import re
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


def texto_legivel(markdown):
    """Retira a marcação do trecho documental sem interpretá-lo como HTML.

    Os rótulos exibem artefatos em texto simples de propósito, então converter
    para HTML abriria a porta para o documento controlar a apresentação. Aqui a
    sintaxe apenas some: o leitor vê a frase, não os asteriscos.

    Args:
        markdown: Trecho extraído da documentação.

    Returns:
        Par de título e corpo já sem marcação.
    """
    # A ênfase costuma atravessar a quebra de linha no documento, então a
    # remoção acontece no texto inteiro antes de separá-lo em linhas.
    texto = re.sub(r'\*\*(.+?)\*\*', r'\1', markdown, flags=re.DOTALL)
    texto = re.sub(r'`(.+?)`', r'\1', texto, flags=re.DOTALL)
    linhas = [re.sub(r'^\s*-\s+', '• ', linha) for linha in texto.splitlines()
              if linha.strip() not in ('---', '***', '___')]
    titulo, corpo = '', linhas
    if linhas and linhas[0].lstrip().startswith('#'):
        titulo = linhas[0].lstrip('# ').strip()
        corpo = linhas[1:]
    return titulo, '\n'.join(corpo).strip('\n')


class DefensePage(Page):
    """Apresenta a delimitação final do argumento da defesa."""
    navigate_stage = Signal(str, str)

    def display(self, data, ctx):
        """Monta o roteiro e direciona cada passo à evidência navegável."""
        self.reset_body()
        titulo, corpo = texto_legivel(ressalva(data['resultados'],
                                               '## 9. O que este documento **não** afirma'))
        widget, layout = card(titulo or 'Ressalvas', '')
        layout.addWidget(label(corpo, 'notice'))
        self.content.addWidget(widget)
        self.content.addStretch()
        self.ready()


class EvidencePage(SectionedPage):
    """Agrupa relatórios diagnósticos em abas para consulta durante a defesa."""
    navigate_stage = Signal(str, str)
    section_names = ('Matrizes', 'Silêncio e travessia', 'Sessão ou transdutor',
                     'Estrutura dos erros', 'Controles', 'Protocolos auxiliares', 'BrSD')
    stage_sections = {'matrizes': 'Matrizes', 'canal': 'Silêncio e travessia',
                      'sessao': 'Sessão ou transdutor', 'erros': 'Estrutura dos erros',
                      'controles': 'Controles', 'protocolos': 'Protocolos auxiliares',
                      'brsd': 'BrSD'}

    def _proveniencia(self, path, report):
        """Registra a origem do relatório sem afogar a medida que ele sustenta.

        Nomeia os arquivos de configuração em vez de enumerar suas chaves, e
        mostra o comando de reprodução uma única vez por carga: repeti-lo a cada
        um dos seis relatórios empurrava o gráfico para fora da tela.

        Args:
            path: Caminho do relatório exibido.
            report: Conteúdo já lido, mantido para as chamadas que o inspecionam.
        """
        self.content.addWidget(label(f'Procedência: {path}', 'muted'))
        parent = path.parent
        metadata = []
        for name in ('configuracao.json', 'divisao.json'):
            candidate = next((p / name for p in (parent, *parent.parents) if (p / name).is_file()), None)
            if candidate:
                metadata.append(str(candidate))
        if metadata:
            self.content.addWidget(label('Como foi feito (metadados persistidos): ' + ' · '.join(metadata), 'muted'))
        else:
            self.notice('Como foi feito: configuração/divisão não persistidas junto deste relatório; o comando exato não pode ser inferido sem inventá-lo. Gere novamente o experimento para registrar configuracao.json e divisao.json.', True)
        if self.reproducao and not self.reproducao_exibida:
            self.notice('Comando de reprodução documentado:\n' + self.reproducao)
            self.reproducao_exibida = True

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
        self.reproducao_exibida = False

        self.use_section(0)
        self.notice('Pergunta: quanto a identificação muda entre a trilha de treino e a outra trilha?')
        matrix_series, matrix_sources = [], []
        for path, report in data['diagnosticos']['matrizes']:
            ajustes = report.get('ajustes', [])
            perdas = [a['perda_acuracia_pp'] for a in ajustes if 'perda_acuracia_pp' in a]
            if perdas:
                matrix_series.append((path.parent.name, perdas))
                matrix_sources.append((path, report))
        # Numa defesa a medida vem primeiro; a procedência fica logo abaixo dela,
        # e não empurrando o gráfico para fora da tela.
        if matrix_series:
            self.content.addWidget(figuras.distribuicoes(matrix_series, 'Perda pareada por ajuste', 'Perda (pp)'))
            for path, report in matrix_sources:
                self._proveniencia(path, report)
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
        self.content.addWidget(button('Ver assinatura de canal e MFCCs',
                                      lambda: self.navigate_stage.emit('assinatura', ''), primary=True))
        self.content.addWidget(button('Ver MFCCs da gravação selecionada',
                                      lambda: self.navigate_stage.emit('mfcc', '')))
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
        target = self.stage_sections.get(self.stage)
        if target:
            self.sections.setCurrentIndex(self.section_names.index(target))
        self.use_section(self.sections.currentIndex())
        self.ready()
