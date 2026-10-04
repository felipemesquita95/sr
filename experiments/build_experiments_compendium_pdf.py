#!/usr/bin/env python3
"""Relatório único dos experimentos corrigidos de voz a 8 kHz."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A3
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, LongTable, PageBreak,
                                Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)


ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'runs/models'
OUTPUT = ROOT / 'output/pdf/experimentos_reconhecimento_voz_8k.pdf'
FIGURES = ROOT / 'docs/figuras'
NAVY = colors.HexColor('#18344a')
TEAL = colors.HexColor('#087f82')
PALE = colors.HexColor('#e9f2f3')
GRAY = colors.HexColor('#56616d')
LINE = colors.HexColor('#d9e2e7')


def register_fonts() -> None:
    root = Path('/usr/share/fonts/truetype/dejavu')
    pdfmetrics.registerFont(TTFont('DejaVu', str(root / 'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DejaVu-Bold', str(root / 'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('DejaVu', normal='DejaVu', bold='DejaVu-Bold')


def styles() -> dict[str, ParagraphStyle]:
    return {
        'title': ParagraphStyle('title', fontName='DejaVu-Bold', fontSize=23,
                                leading=29, textColor=NAVY, spaceAfter=13),
        'subtitle': ParagraphStyle('subtitle', fontName='DejaVu', fontSize=11.5,
                                   leading=17, textColor=GRAY, spaceAfter=17),
        'h1': ParagraphStyle('h1', fontName='DejaVu-Bold', fontSize=15.5,
                            leading=20, textColor=NAVY, spaceBefore=14,
                            spaceAfter=9),
        'h2': ParagraphStyle('h2', fontName='DejaVu-Bold', fontSize=11.5,
                            leading=15, textColor=TEAL, spaceBefore=12,
                            spaceAfter=7),
        'body': ParagraphStyle('body', fontName='DejaVu', fontSize=9.7,
                              leading=15.2, textColor=NAVY, spaceAfter=7),
        'small': ParagraphStyle('small', fontName='DejaVu', fontSize=8.1,
                               leading=12.2, textColor=GRAY, spaceAfter=5),
        'table': ParagraphStyle('table', fontName='DejaVu', fontSize=8.6,
                               leading=11.2, textColor=NAVY),
        'tablehead': ParagraphStyle('tablehead', fontName='DejaVu-Bold',
                                   fontSize=8.4, leading=11.1,
                                   textColor=colors.white),
    }


def pct(value: float) -> str:
    return f'{100 * value:.2f}'.replace('.', ',') + '%'


def metric(root: str, architecture: str) -> float:
    path = MODELS / root / architecture
    values = [json.loads((path / f'particao{fold}' / 'metricas.json').read_text())
              ['acuracia'] for fold in range(1, 6)]
    return mean(values)


def cross(root: str, source: str, target: str, architecture: str) -> float:
    path = (MODELS / f'{root}_cross_pareado' /
            f'{source}_para_{target}' / architecture / 'resumo.json')
    return json.loads(path.read_text())['acuracia_outro_media']


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def table(rows: list[list[str]], st: dict, widths: list[float] | None = None,
          small: bool = False) -> LongTable:
    data = [[para(str(value), st['tablehead'] if index == 0 else st['table'])
             for value in row] for index, row in enumerate(rows)]
    result = LongTable(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
    result.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PALE]),
        ('LINEBELOW', (0, 0), (-1, 0), .65, NAVY),
        ('LINEBELOW', (0, 1), (-1, -1), .25, LINE),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6 if small else 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6 if small else 7),
    ]))
    return result


def block(st: dict, title: str, method: str, result: str) -> list:
    return [para(title, st['h2']),
            para('<b>O que faz.</b> ' + method, st['body']),
            para('<b>O que mostrou.</b> ' + result, st['body'])]


def source(st: dict, text: str) -> Paragraph:
    return para('Fonte dos números: ' + text, st['small'])


def photo(path: Path, width: float, caption: str, st: dict) -> list:
    from PIL import Image as PILImage

    with PILImage.open(path) as picture:
        height = width * picture.height / picture.width
    image = Image(str(path), width=width, height=height)
    return [image, Spacer(1, 6), para(caption, st['small'])]


def page_header(canvas, doc) -> None:
    canvas.saveState()
    width, height = A3
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(.6)
    canvas.line(52, height - 46, width - 52, height - 46)
    canvas.setFont('DejaVu', 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(52, height - 37, 'RECONHECIMENTO DE LOCUTOR · 8 kHz')
    canvas.line(52, 44, width - 52, 44)
    canvas.drawString(52, 30, 'VCTK e BrSD · resultados corrigidos')
    canvas.drawRightString(width - 52, 30, f'Página {doc.page}')
    canvas.restoreState()


def build() -> None:
    register_fonts()
    st = styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    usable = A3[0] - 104
    story = []

    # Visão geral e protocolo.
    story += [Spacer(1, 28),
              para('Experimentos de reconhecimento de locutor', st['title']),
              para('Relatório consolidado · BrSD e VCTK · áudio a 8 kHz · 24 de setembro de 2026',
                   st['subtitle']),
              para('O objetivo é distinguir o que os modelos aprendem da voz e o que '
                   'pode vir do microfone, das pausas e da sessão de gravação. '
                   'Cada porcentagem é acurácia média de cinco partições, salvo '
                   'quando a seção indica outro tipo de medida.', st['body']),
              para('1. Dados, unidade de análise e divisão', st['h1']),
              table([
                  ['Corpus/coorte', 'Locutores', 'Gravações pareadas', 'Treino / validação / teste por partição'],
                  ['BrSD', '80', '400 (um microfone)', '240 / 80 / 80'],
                  ['VCTK completo, 77 quadros', '108', '21.523 (mic1 e mic2)',
                   '12.913-12.915 / 4.304-4.305 / 4.304-4.305'],
                  ['VCTK, comparação 77 vs. 153', '108', '18.067',
                   'Mesmas gravações e mesmos papéis nas duas durações'],
                  ['VCTK, atividade/baixa de 10', '108', '17.272',
                   'Mesmas gravações e papéis em todas as condições de 10'],
                  ['VCTK, controles de 20', '108', '17.270',
                   'Mesmas gravações e papéis nos controles de 20'],
                  ['VCTK balanceado', '100', '2.500 (25 por locutor)',
                   '1.500 / 500 / 500; por locutor: 15 / 5 / 5'],
              ], st, [170, 72, 167, usable - 409]),
              Spacer(1, 11),
              para('Amostra significa <b>uma gravação</b>; quadro significa uma janela '
                   'de 32 ms avançada a cada 16 ms. Mic1 e mic2 são captações '
                   'simultâneas da mesma gravação. Em cada partição, três grupos '
                   'ficam no treino, um na validação e um no teste: a regra 60/20/20 '
                   'gira por cinco partições.', st['body']),
              para('As coortes não devem ser comparadas por acurácia bruta como se '
                   'fossem o mesmo teste. Dentro de cada tabela controlada, modelos '
                   'e condições compartilham gravações e divisão. A coorte final '
                   'tem 100 classes (acaso 1%); as outras VCTK têm 108 '
                   '(acaso 0,93%).', st['body']),
              source(st, 'docs/relatorio_8k.md; docs/comparacao_pareada_77_153.md; '
                     'docs/vctk_activity_probe_selection.json; '
                     'docs/vctk_bal100_selection.json.'),
              PageBreak()]

    story += [para('2. Uma rodada do processamento', st['h1']),
              para('Exemplo único: gravação 1. O painel mostra BrSD e as duas '
                   'captações de p225 no VCTK; as outras quatro gravações '
                   'ilustrativas do relatório anterior foram omitidas.', st['body'])]
    story += photo(FIGURES / 'relatorio_8k/gravacao_1.png', usable,
                   'Sinal original, efeito do filtro, reamostragem, pré-ênfase e '
                   'MFCC. A linha vermelha marca o trecho entregue ao modelo: '
                   '1.007 quadros no BrSD e 77 no VCTK. A matriz MFCC é '
                   'extraída do áudio completo.', st)
    story += [PageBreak(),
              para('3. Passo a passo usado nas execuções a 8 kHz', st['h1'])]
    steps = [
        ('1 · Carregar', 'Ler o arquivo na taxa nativa (48 kHz no VCTK; '
         'a maioria do BrSD também em 48 kHz). Somar canais quando houver '
         'áudio estéreo; guardar a gravação completa.'),
        ('2 · Filtrar', 'Aplicar filtro Butterworth de ordem 8, fase zero, '
         'com corte de 3,6 kHz. Ele reduz energia que causaria aliasing '
         'ao baixar a taxa.'),
        ('3 · Reamostrar', 'Converter com reamostragem polifásica para 8 kHz. '
         'O áudio passa a ter limite de Nyquist de 4 kHz.'),
        ('4 · Pré-enfatizar', 'Aplicar coeficiente 0,97 para realçar mudanças '
         'rápidas do sinal antes da extração cepstral.'),
        ('5 · Janelar e extrair', 'Janelas de 256 amostras (32 ms), salto de '
         '128 (16 ms). Em cada janela são calculados 40 MFCCs.'),
        ('6 · Selecionar quadros', 'A matriz completa é salva. Cada experimento '
         'escolhe os primeiros 77/153 quadros ou posições de atividade/baixa '
         'atividade; a seleção ocorre antes da entrada da rede.'),
        ('7 · Dividir e normalizar', 'Separar por gravação no 60/20/20. '
         'A média e o desvio da normalização vêm só do treino; validação '
         'escolhe a época e teste é reservado para avaliação.'),
    ]
    for title, body in steps:
        story += [para(title, st['h2']), para(body, st['body'])]
    story += [Spacer(1, 12),
              para('VAD nos diagnósticos posteriores', st['h2']),
              para('Para os controles de atividade, o detector de energia '
                   '`librosa.effects.split` usa top_db=30 após redução a 8 kHz. '
                   'Só entram posições com o mesmo rótulo nos dois microfones '
                   'e uma margem de segurança nas transições. Baixa atividade '
                   '<b>não é silêncio anotado</b>: pode conter fala fraca, '
                   'respiração e ruído.', st['body']),
              source(st, 'docs/reexecucao_8k.md; '
                     'experiments/prepare_vctk_activity_probe.py.'),
              PageBreak()]

    # PSD, BrSD e largura do VCTK.
    story += [para('4. Banda de 8 kHz: diagnóstico espectral', st['h1'])]
    story += block(st, 'PSD antes da redução',
                   'Medir a potência espectral em 324 pares de gravações '
                   '(três por locutor), antes do filtro e da reamostragem.',
                   'A mediana da potência abaixo de 4 kHz é 98,59% no mic1 '
                   'e 99,46% no mic2. Isso apoia a escolha da banda para '
                   'potência acústica; não prova que toda informação de '
                   'identidade esteja nela.')
    story += [table([
        ['Banda', 'Mic1', 'Mic2'],
        ['Abaixo de 3,6 kHz', '98,25%', '99,38%'],
        ['Abaixo de 4 kHz', '98,59%', '99,46%'],
        ['4 a 8 kHz', '0,86%', '0,38%'],
        ['8 a 24 kHz', '0,32%', '0,10%'],
    ], st, [250, 170, usable - 420]), Spacer(1, 8)]
    story += photo(FIGURES / 'psd_vctk_8k.png', 510,
                   'Potência espectral e energia acumulada. Medianas por gravação, '
                   'antes do filtro antialiasing.', st)
    story += [source(st, 'docs/psd_vctk_8k.md.'), PageBreak(),
              para('5. Testes de referência: BrSD e VCTK', st['h1'])]
    story += block(st, 'BrSD · 80 locutores',
                   'Reconhecer cada locutor no mesmo corpus com cinco leituras '
                   'por pessoa e 60/20/20 exato.',
                   'A CNN temporal obteve 76,75%; CNN, 72,25%; atenção, '
                   '47,50%. É referência em outro corpus, não comparação '
                   'direta com a acurácia VCTK.')
    story += [table([
        ['Rede', 'Acurácia média', 'Gravações por partição'],
        ['CNN', pct(metric('brsd', 'cnn')), '240 / 80 / 80'],
        ['CNN temporal', pct(metric('brsd', 'temporal_cnn')), '240 / 80 / 80'],
        ['Atenção', pct(metric('brsd', 'attention')), '240 / 80 / 80'],
    ], st, [190, 160, usable - 350])]
    story += block(st, 'VCTK · primeiros 77 quadros',
                   'Treinar e testar no mesmo microfone, usando os primeiros '
                   '77 quadros das 21.523 gravações pareadas.',
                   'A CNN temporal chegou a 96,04% no mic1 e 92,39% no mic2. '
                   'A acurácia inclui pistas de voz e de captação.')
    story += block(st, 'VCTK · primeiros 153 quadros',
                   'Usar 153 quadros iniciais nas 18.067 gravações longas '
                   'o suficiente, com os mesmos 108 locutores.',
                   'A CNN temporal chegou a 98,60% no mic1 e 97,08% no mic2; '
                   'o conjunto difere do experimento completo de 77.')
    rows = [['Quadros', 'Rede', 'Mesmo mic1', 'Mesmo mic2', 'mic1 → mic2', 'mic2 → mic1']]
    names = {'cnn': 'CNN', 'temporal_cnn': 'CNN temporal', 'attention': 'Atenção'}
    for width, prefix in ((77, 'vctk8k'), (153, 'vctk153')):
        for arch in names:
            rows.append([str(width), names[arch],
                         pct(metric(f'{prefix}_mic1', arch)),
                         pct(metric(f'{prefix}_mic2', arch)),
                         pct(cross(prefix, 'mic1', 'mic2', arch)),
                         pct(cross(prefix, 'mic2', 'mic1', arch))])
    story += [table(rows, st, [70, 145, 116, 116, 119, usable - 566]),
              source(st, 'docs/relatorio_8k.md; runs/models/vctk8k_*; '
                     'runs/models/vctk153_*.'), PageBreak()]

    story += [para('6. Duração controlada e troca de microfone', st['h1'])]
    story += block(st, '77 vs. 153 no mesmo conjunto',
                   'Retreinar as três redes com 77 ou 153 quadros nas mesmas '
                   '18.067 gravações e exatamente nas mesmas partições.',
                   '153 venceu nas seis combinações de rede e microfone; '
                   'a melhora vai de 2,64 a 7,02 pontos percentuais. '
                   'O produto gravações × quadros é máximo em 153.')
    rows = [['Microfone', 'Rede', '77', '153', 'Ganho de 153']]
    for mic in ('mic1', 'mic2'):
        for arch in names:
            a = metric(f'vctk77_matched153_{mic}', arch)
            b = metric(f'vctk153_{mic}', arch)
            rows.append([mic, names[arch], pct(a), pct(b),
                         f'+{(b-a)*100:.2f}'.replace('.', ',') + ' pp'])
    story += [table(rows, st, [105, 195, 120, 120, usable - 540])]
    story += block(st, 'Cross-microfone pareado · 77 e 153',
                   'Avaliar o mesmo checkpoint e as mesmas gravações de '
                   'teste no outro microfone, sem recalcular a normalização.',
                   'A CNN de 153 caiu de 97,40% para 23,68% em mic1 → mic2 '
                   'e de 93,91% para 34,96% em mic2 → mic1. A troca do '
                   'transdutor reduz muito o desempenho, sem isolar voz pura.')
    story += [para('A tabela anterior inclui as duas direções do cross para '
                   'cada arquitetura. Os cross de 77 e 153 têm coortes '
                   'diferentes; a comparação causal de duração é a tabela '
                   'pareada 77 vs. 153 acima.', st['small']),
              source(st, 'docs/comparacao_pareada_77_153.md; '
                     'runs/models/vctk8k_cross_pareado; '
                     'runs/models/vctk153_cross_pareado.'), PageBreak()]

    # Activity 10.
    story += [para('7. Atividade e baixa atividade · 108 locutores', st['h1'])]
    story += block(st, '10 quadros por condição',
                   'Examinar o áudio inteiro e formar entradas de 10 quadros '
                   'sem seleção, ativos, de baixa atividade ou mistos 5+5 '
                   'nas mesmas 17.272 gravações.',
                   'Atividade superou baixa atividade em todas as redes e '
                   'direções. A baixa atividade reconheceu pessoas no mesmo '
                   'mic, mas perdeu grande parte da acurácia no outro mic.')
    names10 = [('unfiltered', 'Sem seleção'), ('active', 'Atividade'),
               ('low', 'Baixa atividade'), ('mixed', '5+5')]
    rows = [['Condição', 'Rede', 'Mic1', 'Mic2', '1 → 2', '2 → 1']]
    for condition, label in names10:
        root = f'vctk_activity10_{condition}'
        for arch in names:
            rows.append([label, names[arch],
                         pct(metric(f'{root}_mic1', arch)),
                         pct(metric(f'{root}_mic2', arch)),
                         pct(cross(root, 'mic1', 'mic2', arch)),
                         pct(cross(root, 'mic2', 'mic1', arch))])
    story += [table(rows, st, [125, 150, 112, 112, 112, usable - 611], small=True),
              para('As posições selecionadas vêm de trechos possivelmente '
                   'separados do áudio completo. O controle 5+5 substitui '
                   'metade dos quadros de atividade mantendo a entrada com '
                   'dez posições.', st['small']),
              source(st, 'docs/resultado_atividade_baixa_vctk.md; '
                     'docs/resultado_atividade_mais_baixa_vctk.md.'), PageBreak()]

    # Activity 20 and all frames.
    story += [para('8. Controles de 20 quadros e do áudio inteiro', st['h1'])]
    story += block(st, '20 de atividade vs. 10+10',
                   'Trocar metade dos quadros ativos por baixa atividade '
                   'nas mesmas 17.270 gravações, preservando entrada de '
                   '20 quadros e cinco partições.',
                   '20 ativos venceu 10+10 nas seis combinações no mesmo '
                   'microfone e nas seis transferências. Na CNN mic1 → mic2: '
                   '30,28% contra 23,97%.')
    rows = [['Condição', 'Rede', 'Mic1', 'Mic2', '1 → 2', '2 → 1']]
    for condition, label in (('active', '20 atividade'), ('mixed', '10+10')):
        root = f'vctk_activity20_{condition}'
        for arch in names:
            rows.append([label, names[arch],
                         pct(metric(f'{root}_mic1', arch)),
                         pct(metric(f'{root}_mic2', arch)),
                         pct(cross(root, 'mic1', 'mic2', arch)),
                         pct(cross(root, 'mic2', 'mic1', arch))])
    story += [table(rows, st, [125, 150, 112, 112, 112, usable - 611])]
    story += block(st, 'Todos os quadros, classificador linear',
                   'Usar média e desvio dos 40 MFCCs de todos os quadros '
                   'de cada condição nas mesmas 17.270 gravações.',
                   'A atividade transferiu melhor que baixa atividade '
                   '(40,04% vs. 13,06% em mic1 → mic2). Este modelo estático '
                   'é comparável dentro da sua tabela, não às redes acima.')
    all_frames = [
        ['Todos, sem seleção', '209', '95,30%', '89,29%', '34,31%', '37,32%'],
        ['Todos de atividade', '88', '91,26%', '82,72%', '40,04%', '45,91%'],
        ['Todos de baixa atividade', '55', '74,56%', '61,80%', '13,06%', '22,13%'],
        ['20 atividade', '20', '87,01%', '75,98%', '35,99%', '42,72%'],
        ['10+10', '20', '85,71%', '72,84%', '26,44%', '32,47%'],
    ]
    story += [table([['Condição', 'Quadros medianos', 'Mic1', 'Mic2', '1 → 2', '2 → 1']]
                    + all_frames, st,
                    [180, 130, 98, 98, 98, usable - 604]),
              source(st, 'docs/resultado_atividade_mais_baixa_vctk.md; '
                     'docs/resultado_todos_quadros_vctk.md.'), PageBreak()]

    # Balanced cohort.
    story += [para('9. Coorte balanceada · 100 locutores', st['h1'])]
    story += block(st, 'Escolha das gravações',
                   'Reter 25 gravações por locutor que tenham ao menos '
                   '20 quadros de baixa atividade e 40 de atividade; '
                   'excluir oito locutores que não sustentavam o corte.',
                   'São 2.500 gravações com divisão exata 1.500/500/500 '
                   'em cada partição. Todas as condições e microfones '
                   'usam a mesma lista e os mesmos papéis.')
    story += photo(FIGURES / 'fronteira_locutores_atividade_vctk.png', 475,
                   'Único gráfico de disponibilidade das amostras: '
                   'quadros possíveis e gravações iguais por locutor '
                   'ao retirar poucos locutores.', st)
    story += [PageBreak(), para('10. Resultado do recorte balanceado', st['h1'])]
    story += block(st, '20 atividade vs. 20 baixa',
                   'Treinar a mesma rede com 20 posições ativas ou '
                   '20 de baixa atividade nas mesmas 2.500 gravações.',
                   'A atividade venceu em todas as seis avaliações '
                   'dentro do mic e nas seis transferências. CNN '
                   'mic1 → mic2: 31,40% contra 10,84%.')
    story += block(st, '40 quadros: completo, ativo e misto 20+20',
                   'Comparar 40 posições distribuídas no áudio, '
                   '40 ativas e a mistura de 20 ativas com 20 de baixa.',
                   'No mesmo mic, a mistura ficou próxima da atividade '
                   'pura e às vezes a superou; no cross, 40 ativos '
                   'venceu 20+20 nas seis combinações.')
    rows = [['Condição', 'Rede', 'Mic1', 'Mic2', '1 → 2', '2 → 1']]
    balanced = [('active20', '20 atividade'), ('low20', '20 baixa'),
                ('unfiltered40', '40 sem seleção'),
                ('active40', '40 atividade'), ('mixed40', '20+20')]
    for condition, label in balanced:
        root = f'vctk_bal100_{condition}'
        for arch in names:
            rows.append([label, names[arch],
                         pct(metric(f'{root}_mic1', arch)),
                         pct(metric(f'{root}_mic2', arch)),
                         pct(cross(root, 'mic1', 'mic2', arch)),
                         pct(cross(root, 'mic2', 'mic1', arch))])
    story += [table(rows, st, [130, 145, 112, 112, 112, usable - 611], small=True),
              source(st, 'docs/resultado_vctk_bal100_atividade.md; '
                     'docs/vctk_bal100_selection.json.'), PageBreak()]

    story += [para('11. Protocolos auxiliares documentados anteriormente', st['h1']),
              para('Os resultados abaixo constam em docs/resultados.md e '
                   'docs/avaliacao_atual.md, mas seus diretórios de modelos '
                   'não estão presentes na árvore atual de runs. São '
                   'registrados para não perder experimentos realizados; '
                   'não foram revalidados para este PDF e usaram coortes, '
                   'divisões ou representações diferentes das tabelas '
                   'corrigidas acima.', st['body'])]
    story += block(st, 'Matriz de transferência · seis arquiteturas',
                   'Treinar em uma trilha e testar o mesmo checkpoint '
                   'nas duas, com divisão fixa por enunciado e três '
                   'sementes de treino.',
                   'A atenção temporal foi documentada com 99,43% no '
                   'mic1, 43,12% em mic1 → mic2 e 59,57% no sentido '
                   'inverso. A direção da troca alterou bastante o '
                   'resultado; os números não são do ciclo 77/153.')
    story += block(st, 'Referência linear e diagnóstico de baixa energia',
                   'Resumir os 40 MFCCs por média e desvio; também '
                   'separar trechos de maior e menor energia.',
                   'A referência linear foi documentada com 52,87% '
                   '(1 → 2) e 58,44% (2 → 1). O diagnóstico de baixa '
                   'energia obteve 85,40% no mic de origem e 4,33% '
                   'ao atravessar, em outro recorte.')
    story += block(st, 'Calibração de sessão entre microfones',
                   'Aprender uma transformação afim sem rótulos de '
                   'identidade em 36 locutores e avaliar nos outros '
                   '72, usando assinaturas de baixa energia.',
                   'A acurácia cruzada documentada passou de 6,0% '
                   'para 29,7% após a transformação; isso indica '
                   'pista compartilhada entre capturas que não é '
                   'legível nas coordenadas originais.')
    story += block(st, 'Texto disjunto e VAD ligado/desligado',
                   'No cross histórico de uma partição, reservar '
                   'enunciados diferentes para teste ou retirar o '
                   'detector de atividade.',
                   'A CNN documentada obteve 38,69% com texto '
                   'disjunto, 38,12% no cross de referência e '
                   '38,20% sem VAD; duração e tamanho do treino '
                   'também mudaram.')
    story += block(st, 'Rótulos permutados e alinhamento',
                   'Embaralhar identidades como controle negativo '
                   'e recuperar o par de áudio correspondente '
                   'entre microfones por correlação.',
                   'A acurácia documentada caiu a 0,70% no mesmo '
                   'mic e 0,81% no cross, próxima de 0,93% de acaso; '
                   'a recuperação de pares foi 100/100.')
    story += [source(st, 'docs/resultados.md, seções 2-7; '
                     'docs/avaliacao_atual.md. Resultados históricos '
                     'separados dos artefatos reexecutados neste PDF.'),
              PageBreak(),
              para('12. O que o conjunto dos testes permite dizer', st['h1']),
              para('<b>1.</b> O sinal de atividade carrega informação de '
                   'locutor e transfere melhor entre microfones que '
                   'a baixa atividade. Isso se repetiu em entradas de '
                   '10, 20 e 40 quadros e no classificador estático.', st['body']),
              para('<b>2.</b> Baixa atividade também permite reconhecer '
                   'locutores muito acima do acaso dentro do mesmo microfone. '
                   'A queda cruzada indica forte dependência da condição '
                   'de captação; não quantifica uma fração causal de '
                   '“identidade do canal”.', st['body']),
              para('<b>3.</b> A mistura de quadros ativos e de baixa '
                   'atividade não melhora consistentemente o resultado. '
                   'Na coorte balanceada, 40 ativos venceram 20+20 em '
                   'todas as transferências, embora a acurácia interna '
                   'seja parecida em algumas redes.', st['body']),
              para('<b>4.</b> 153 quadros iniciais superam 77 quando '
                   'o conjunto de gravações é pareado, mas aumentar a '
                   'acurácia no mesmo mic não elimina a queda ao '
                   'trocar de transdutor.', st['body']),
              para('<b>5.</b> A banda até 4 kHz contém a maior parte '
                   'da potência acústica medida, mas o estudo não '
                   'testou modelos com outras bandas nas mesmas '
                   'partições. É evidência espectral, não prova '
                   'completa de suficiência para identificação.', st['body']),
              para('Limites de interpretação', st['h2']),
              para('“Baixa atividade” é um rótulo por energia, não '
                   'silêncio validado por escuta. As condições mantêm '
                   'sessão e frequentemente o mesmo enunciado ao '
                   'trocar de microfone. A inicialização das redes '
                   'não foi repetida por várias sementes no ciclo '
                   'recente; as médias são entre partições. '
                   'A coorte de 100 locutores responde a uma tarefa '
                   'diferente da coorte de 108.', st['body']),
              para('Procedência e exclusões', st['h2']),
              para('As tabelas principais usam artefatos atuais de '
                   '`runs/models/` e os relatórios citados em cada '
                   'seção. A seção histórica registra outras '
                   'execuções documentadas, sem tratá-las como '
                   'comparáveis a esta série. Números do cross '
                   'com rótulos deslocados ou da validação antiga '
                   'não foram reapresentados como resultados '
                   'válidos. O PDF contém uma única gravação '
                   'ilustrativa do processamento.', st['body'])]

    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A3,
                            leftMargin=52, rightMargin=52,
                            topMargin=62, bottomMargin=58,
                            title='Experimentos de reconhecimento de locutor a 8 kHz',
                            author='Projeto de reconhecimento de locutor')
    doc.build(story, onFirstPage=page_header, onLaterPages=page_header)
    print(OUTPUT)


if __name__ == '__main__':
    build()
