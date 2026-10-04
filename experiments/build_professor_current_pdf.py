#!/usr/bin/env python3
"""Professor-facing report for the completed Silero VCTK activity grids."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle,
                                Image)
from PIL import Image as PILImage


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    40: Path('/media/lsmsqt/HDD/sr_project/vctk40_activity_nonactivity_20260929'),
    80: Path('/media/lsmsqt/HDD/sr_project/vctk80_activity_20260929_112427'),
    100: Path('/media/lsmsqt/HDD/sr_project/vctk100_activity_20260929'),
}
XVECTOR = ROOT / 'output/vctk100_xvector_silero_20260929/results'
OUTPUT = ROOT / 'output/pdf/relatorio_experimental_reconhecimento_locutor.pdf'
ALIAS = ROOT / 'output/pdf/resumo_experimentos_voz_professor.pdf'
BRSD_OUTPUT = ROOT / 'output/pdf/relatorio_experimental_reconhecimento_locutor_com_brsd_v4.pdf'
FIGURES = ROOT / 'docs/figuras/protocolo_silero_vctk'
BRSD_FIGURES = ROOT / 'docs/figuras/protocolo_silero_brsd'
BRSD_RUNS = Path('/media/lsmsqt/HDD/sr_project')
NAVY = colors.HexColor('#17334c')
TEAL = colors.HexColor('#087d86')
PALE = colors.HexColor('#e9f3f4')
INK = colors.HexColor('#263b49')
MUTED = colors.HexColor('#58707d')
LINE = colors.HexColor('#d7e3e7')
ORANGE = colors.HexColor('#bd6935')
PAGE_W, PAGE_H = A4
WIDTH = PAGE_W - 80
DIRECTIONS = [('mic1', 'mic1'), ('mic1', 'mic2'),
              ('mic2', 'mic1'), ('mic2', 'mic2')]


def read_rows() -> tuple[dict[int, list[dict]], list[dict] | None, dict]:
    data = {}
    for frames, folder in RUNS.items():
        status = json.loads((folder / 'status.json').read_text())
        assert status['status'] == 'complete'
        with (folder / 'resultados_resumo.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == (384 if frames == 40 else 96)
        assert all(row['folds_completed'] == row['folds_expected'] == '5'
                   for row in rows)
        data[frames] = rows
    status_path = XVECTOR / 'status.json'
    xrows = None
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status['status'] == 'complete' and status['models_completed'] == 20:
            with (XVECTOR / 'resultados_resumo.csv').open(newline='') as stream:
                xrows = list(csv.DictReader(stream))
            assert len(xrows) == 8
            assert all(row['folds_completed'] == row['folds_expected'] == '5'
                       for row in xrows)
    brsd = {}
    for n_mfcc in (30, 40):
        for architecture in ('cnn', 'temporal_cnn'):
            for normalization in ('zscore', 'cmn', 'cmvn', 'rasta', 'cmvn_logmel'):
                suffix = 'brsd40' if n_mfcc == 40 else 'brsd'
                path = (BRSD_RUNS / f'isolated_silero_{normalization}_{suffix}_'
                        f'{architecture}_gpu_b128/dynamic/summary.json')
                row = json.loads(path.read_text())['directions']['audio->audio']['accuracy']
                assert len(row['fold_values']) == 5
                brsd[n_mfcc, architecture, normalization] = row
    assert len(brsd) == 20
    return data, xrows, brsd


def pct(value: str | float, digits: int = 1) -> str:
    return f'{100 * float(value):.{digits}f}'.replace('.', ',') + '%'


def pct_sd(row: dict) -> str:
    return f"{pct(row['accuracy_mean'])} ± {pct(row['accuracy_std'])}"


def matches(rows: list[dict], **wanted) -> list[dict]:
    return [row for row in rows if all(row[key] == str(value)
                                      for key, value in wanted.items())]


def best(rows: list[dict], **wanted) -> dict:
    eligible = matches(rows, **wanted)
    assert eligible, wanted
    return max(eligible, key=lambda row: float(row['accuracy_mean']))


def fonts_and_styles() -> dict:
    base = Path('/usr/share/fonts/truetype/dejavu')
    pdfmetrics.registerFont(TTFont('DV', str(base / 'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DV-B', str(base / 'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('DV', normal='DV', bold='DV-B')
    return {
        'title': ParagraphStyle('title', fontName='DV-B', fontSize=21,
                                leading=27, textColor=NAVY, spaceAfter=13),
        'deck': ParagraphStyle('deck', fontName='DV', fontSize=10.4,
                               leading=15.5, textColor=MUTED, spaceAfter=15),
        'h1': ParagraphStyle('h1', fontName='DV-B', fontSize=14,
                             leading=19, textColor=NAVY, spaceBefore=4,
                             spaceAfter=9),
        'h2': ParagraphStyle('h2', fontName='DV-B', fontSize=10.2,
                             leading=14, textColor=TEAL, spaceBefore=10,
                             spaceAfter=6),
        'body': ParagraphStyle('body', fontName='DV', fontSize=8.7,
                               leading=13.2, textColor=INK, spaceAfter=8),
        'small': ParagraphStyle('small', fontName='DV', fontSize=7.5,
                                leading=11.2, textColor=MUTED, spaceAfter=7),
        'cell': ParagraphStyle('cell', fontName='DV', fontSize=7.1,
                               leading=10, textColor=INK),
        'head': ParagraphStyle('head', fontName='DV-B', fontSize=7.1,
                               leading=10, textColor=colors.white),
        'card': ParagraphStyle('card', fontName='DV-B', fontSize=11,
                               leading=15, textColor=NAVY, alignment=TA_CENTER),
    }


def table(raw: list[list[str]], styles: dict, widths: list[float],
          *, paddings: float = 5) -> Table:
    assert abs(sum(widths) - WIDTH) < 1
    cells = [[Paragraph(str(value), styles['head' if index == 0 else 'cell'])
              for value in row] for index, row in enumerate(raw)]
    item = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
    item.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PALE]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), paddings),
        ('RIGHTPADDING', (0, 0), (-1, -1), paddings),
        ('TOPPADDING', (0, 0), (-1, -1), paddings),
        ('BOTTOMPADDING', (0, 0), (-1, -1), paddings),
        ('LINEBELOW', (0, -1), (-1, -1), .4, LINE),
    ]))
    return item


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(40, 38, PAGE_W - 40, 38)
    canvas.setFont('DV', 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(40, 25, 'VCTK e BRSD | protocolos Silero VAD 6.2.3 | 16 kHz')
    canvas.drawRightString(PAGE_W - 40, 25, str(doc.page))
    canvas.restoreState()


def build() -> None:
    data, xrows, brsd = read_rows()
    style = fonts_and_styles()
    P = lambda text, kind='body': Paragraph(text, style[kind])
    def figure(name: str, width: float = WIDTH) -> Image:
        path = FIGURES / name
        with PILImage.open(path) as source:
            height = width * source.height / source.width
        return Image(str(path), width=width, height=height)
    def brsd_figure(name: str, width: float = WIDTH) -> Image:
        path = BRSD_FIGURES / name
        with PILImage.open(path) as source:
            height = width * source.height / source.width
        return Image(str(path), width=width, height=height)
    story = []
    now = datetime.now(ZoneInfo('America/Sao_Paulo')).strftime('%d/%m/%Y')

    # 1. Scope and cohort choice, including the frame-volume table.
    story += [P('Reconhecimento de locutor', 'title'),
              P('VCTK: janelas contínuas de 40, 80 e 100 quadros. BRSD: '
                '111 quadros de fala. Protocolos e resultados separados.', 'deck'),
              P(f'Relatório de acompanhamento para o professor | {now}', 'small'),
              P('Objetivo e escopo', 'h1'),
              P('Os dois bancos avaliam identificação fechada de locutores conhecidos. '
                'No VCTK, a transferência entre mic1 e mic2 examina a dependência da '
                'captação; o BRSD dispõe de uma captação por locutor. As escolhas de '
                'quadros e os resultados pertencem a protocolos distintos.'),
              P('VCTK: coortes e volume de quadros', 'h2'),
              table([['Janela', 'Locutores', 'Leituras/locutor', 'Duração',
                      'Quadros por classe e mic'],
                     ['40 quadros', '108', '110', '0,656 s', '475.200'],
                     ['80 quadros', '108', '100', '1,296 s', '864.000'],
                     ['100 quadros', '108', '60', '1,616 s', '648.000']],
                    style, [80, 76, 112, 82, WIDTH-350]),
              Spacer(1, 9),
              P('Total descritivo = 108 locutores × leituras por locutor × quadros por '
                'leitura. Cada captação pareada conta uma vez; em 40 quadros há '
                '475.200 quadros de fala e outros 475.200 de não fala por microfone. '
                'Os quadros vizinhos se sobrepõem no tempo e não são observações '
                'independentes.', 'small'),
              P('Por que 40, 80 e 100 quadros?', 'h2'),
              P('A janela de 40 quadros preserva mais leituras elegíveis por locutor; '
                '100 quadros oferecem mais contexto em cada leitura; 80 quadros produzem '
                'o maior volume total de quadros de fala nesta seleção. Quando uma leitura '
                'comporta as três durações, as janelas menores podem ficar aninhadas '
                'no mesmo trecho contínuo.'),
              P('As coortes globais têm quantidades diferentes de leituras. Portanto, '
                'diferenças de acurácia entre durações não isolam o efeito do tempo de '
                'fala. Um controle específico exigiria a mesma coorte elegível para '
                '100 quadros e as três janelas centradas nas mesmas leituras.', 'small'),
              P('BRSD nesta análise: 80 locutores × 5 textos = 400 gravações; '
                '111 quadros contínuos de fala por gravação. A seleção exata é '
                'mostrada em seção própria.', 'small'),
              P('Como cada corpus escolhe os quadros', 'h2'),
              table([['Corpus', 'Regra de seleção executada', 'Janelas'],
                     ['VCTK atual', 'Maior trecho contínuo da classe comum aos dois '
                      'microfones; janela central.', '40/80/100; fala e, aos 40, não fala'],
                     ['BRSD 16 kHz', 'Após recorte RMS das bordas, 111 quadros '
                      'contínuos de fala Silero com maior RMS.', '111; somente fala']],
                    style, [91, 278, WIDTH-369], paddings=5),
              PageBreak()]

    # 2. Signal processing with a real paired recording.
    story += [P('1. Do áudio original ao sinal analisado', 'h1'),
              P('Exemplo real: VCTK p225_013, presente nas três coortes. O par '
                'mic1/mic2 parte dos arquivos FLAC a 48 kHz. Cada trilha é '
                'convertida para 16 kHz por decimação IIR de fase zero com '
                'antialiasing interno; não há corte anterior por RMS.', 'body'),
              figure('01_audio_cadeia.png'),
              P('Os dois primeiros painéis mostram o áudio completo do mic1. '
                'O terceiro mostra a pré-ênfase 0,97 somente no segmento comum '
                'de fala usado na extração; o detector Silero atua antes da '
                'pré-ênfase, no sinal de 16 kHz.', 'small'),
              P('Cadeia nesta etapa', 'h2'),
              table([['Entrada', 'Conversão', 'Detector'],
                     ['FLAC mono, 48 kHz, duas trilhas pareadas',
                      'decimate(q=3, n=8, IIR, zero_phase=True) → 16 kHz',
                      'Silero VAD 6.2.3 em cada trilha']],
                    style, [158, 205, WIDTH-363]),
              PageBreak()]

    story += [P('2. Efeito no espectro', 'h1'),
              P('A mesma gravação p225_013 permite conferir a conversão de '
                'taxa e a pré-ênfase. O protocolo atual usa 16 kHz, com limite '
                'de Nyquist em 8 kHz. A decimação já inclui antialiasing; '
                'não há um passa-baixas Butterworth separado a 3,6 kHz.', 'body'),
              figure('01b_espectro.png'),
              P('Acima: densidade espectral de potência (PSD) do arquivo '
                'completo antes e depois da conversão. Abaixo: o segmento '
                'contínuo selecionado, antes e depois da pré-ênfase 0,97. '
                'A pré-ênfase muda o peso relativo das frequências altas '
                'antes do cálculo log-mel.', 'small'),
              P('Essas curvas descrevem potência acústica. Não demonstram '
                'por si só quanta informação de identidade vocal foi '
                'preservada; isso é avaliado pelos testes entre microfones.',
                'small'),
              PageBreak()]

    # 3. Frame selection, including the non-speech control.
    story += [P('3. VCTK: como os quadros foram escolhidos', 'h1'),
              P('Silero VAD 6.2.3: limiar de entrada 0,50; saída 0,35; fala '
                'mínima 250 ms; silêncio mínimo 200 ms; margem de 30 ms. '
                'As regiões de fala são detectadas separadamente em mic1 e mic2. '
                'A interseção define os trechos utilizáveis nos mesmos instantes '
                'das duas captações.', 'body'),
              figure('02_vad_janelas.png'),
              P('No exemplo, os segmentos de fala começam em 0,930 s no mic1 '
                'e 0,962 s no mic2; a parte comum começa em 0,962 s. '
                'Seleciona-se o segmento comum contínuo mais longo; em empate, '
                'o mais cedo. A janela de K quadros fica no centro desse '
                'segmento. Para 40 quadros, repetimos o procedimento na classe '
                'não fala, sem misturar as classes.', 'small'),
              P('Cada quadro usa 512 amostras (32 ms), com salto de 256 '
                '(16 ms). A cobertura de K quadros é 512 + (K - 1) × 256 '
                'amostras: 0,656 s, 1,296 s e 1,616 s. '
                'As janelas são contínuas e têm os mesmos limites nos dois '
                'microfones.', 'small'),
              PageBreak()]

    # 4. Cohort selection and features.
    story += [P('4. Coortes, MFCCs e derivadas', 'h1'),
              P('Para cada duração, mantemos 108 locutores e escolhemos, '
                'sem reposição e com semente 42, o maior múltiplo de cinco '
                'de gravações elegíveis por locutor: 110 para 40 quadros, '
                '100 para 80 e 60 para 100. A coorte de 40 exige janelas '
                'tanto de fala quanto de não fala; as de 80 e 100 exigem fala.',
                'body'),
              figure('03_mel_mfcc_deltas.png'),
              P('Figura: janela de 100 quadros de p225_013 no mic1, extraída '
                'pelo código do experimento. No segmento comum aplica-se '
                'pré-ênfase 0,97, janela Hamming, 128 bandas mel, log de '
                'potência e DCT-II ortonormal. Retêm-se 20, 30 ou 40 MFCCs. '
                'Após recortar a janela, calculam-se Δ e ΔΔ com largura '
                'de nove quadros; nenhum filtro temporal cruza trechos '
                'descontínuos.', 'small'),
              table([['MFCCs estáticos', 'Δ', 'ΔΔ', 'Entrada da rede'],
                     ['20', '20', '20', '60 × K'],
                     ['30', '30', '30', '90 × K'],
                     ['40', '40', '40', '120 × K']],
                    style, [128, 91, 91, WIDTH-310]),
              Spacer(1, 7),
              P('As 128 bandas pertencem ao banco log-mel, não à quantidade de '
                'MFCCs retidos. Cada MFCC combina informações das bandas após '
                'a DCT; não corresponde a uma única frequência central. '
                'Δ e ΔΔ descrevem a variação temporal de cada coeficiente '
                'ao longo de quadros vizinhos.', 'small'),
              PageBreak()]

    # 5. Exact placement of normalization treatments.
    story += [P('5. VCTK: onde entra cada tratamento', 'h1'),
              P('As quatro condições são isoladas. A mesma gravação, a mesma '
                'janela e os mesmos folds são usados ao comparar tratamentos. '
                'CMN, CMVN e RASTA não recebem z-score global adicional.',
                'body'),
              figure('04_pontos_normalizacao.png'),
              table([['Condição', 'Ponto de aplicação', 'Estatísticas e teste'],
                     ['Z-score', 'Depois de MFCC + Δ + ΔΔ; por canal de atributo.',
                      'Média e desvio dos exemplos do treino do microfone de origem; '
                      'as mesmas estatísticas vão à validação e aos dois testes.'],
                     ['CMN', 'Depois da janela e das derivadas; apenas MFCCs estáticos.',
                      'Subtrai a média temporal de cada coeficiente na própria '
                      'janela. Δ e ΔΔ ficam inalterados.'],
                     ['CMVN', 'Depois da janela e das derivadas; MFCCs estáticos.',
                      'Média e desvio da própria janela; Δ e ΔΔ são divididos '
                      'pelo mesmo desvio dos estáticos.'],
                     ['RASTA', 'Nas 128 trajetórias log-mel do segmento contínuo, '
                      'antes da DCT e do recorte final.',
                      'Filtro temporal [0,2; 0,1; 0; -0,1; -0,2] / [1; -0,94]; '
                      'depois DCT, janela e derivadas.']],
                    style, [72, 205, WIDTH-277], paddings=4),
              Spacer(1, 8),
              P('Nos tratamentos por janela, a transformação da gravação de '
                'teste usa somente seus próprios quadros, sem rótulo. O z-score '
                'usa somente o conjunto de treino da origem para estimar '
                'parâmetros; a troca de microfone não reajusta esses valores.',
                'small'),
              PageBreak()]

    # 6. Actual feature example, not a schematic.
    story += [P('6. Efeito visual na mesma gravação', 'h1'),
              P('Os cinco painéis usam os mesmos 100 quadros de p225_013, mic1. '
                'O z-score usa estatísticas salvas pelo treino do fold 1 no '
                'mic1; CMN e CMVN usam essa própria janela; RASTA vem de uma '
                'extração alternativa do mesmo segmento.', 'body'),
              figure('05_normalizacoes_exemplo.png'),
              P('Cada painel ajusta sua escala de cor para mostrar a estrutura; '
                'a intensidade das cores não deve ser comparada diretamente '
                'entre painéis. As cinco entradas foram conferidas contra '
                'os tensores usados nos experimentos.', 'small'),
              PageBreak()]

    story += [P('VCTK: interpretação dos tratamentos', 'h1'),
              P('As quatro condições usam a mesma seleção de gravações e quadros; '
                'as diferenças abaixo descrevem o processamento de atributos. '
                'CMN, CMVN e RASTA são comparações exploratórias motivadas '
                'pela queda de acurácia entre microfones.', 'body'),
              P('Z-score', 'h2'),
              P('A média e o desvio padrão de cada canal de atributo vêm apenas '
                'do treino no microfone de origem. O teste cruzado usa esses '
                'mesmos parâmetros; não se recalculam estatísticas no microfone '
                'de destino.'),
              P('CMN e CMVN', 'h2'),
              P('A CMN remove a média temporal de cada MFCC estático na própria '
                'janela, o que pode reduzir componentes relativamente constantes '
                'do canal. A CMVN também ajusta a escala; neste protocolo, Δ e '
                'ΔΔ são divididos pelo mesmo desvio dos estáticos. Normalizar '
                'mais não implicou necessariamente melhor transferência.'),
              P('RASTA', 'h2'),
              P('O filtro age nas trajetórias temporais das 128 bandas log-mel '
                'antes da DCT. Ele foi testado para atenuar componentes lentos '
                'ou relativamente estacionários. Um ganho de acurácia não '
                'identifica, sozinho, qual componente físico foi removido.'),
              P('Continuidade temporal', 'h2'),
              P('As derivadas são calculadas depois de recortar cada janela '
                'contínua. Concatenar trechos separados criaria mudanças '
                'artificiais nas emendas, que não ocorreram no sinal original.',
                'small'),
              PageBreak()]

    # 3. Evaluation design.
    story += [P('Desenho da avaliação', 'h1'),
              P('A unidade de amostragem é a leitura pareada. As duas captações e as '
                'classes da mesma leitura ficam no mesmo fold. Cada locutor participa '
                'dos cinco folds; em cada rodada, 60% das leituras são treino, 20% '
                'validação e 20% teste. Semente 42.'),
              table([['Quadros', 'Treino por locutor', 'Validação', 'Teste'],
                     ['40', '66', '22', '22'], ['80', '60', '20', '20'],
                     ['100', '36', '12', '12']],
                    style, [100, 170, 123, WIDTH - 393]),
              Spacer(1, 10),
              P('Quatro direções de microfone', 'h2'),
              P('Treina-se e valida-se em mic1 ou mic2. O mesmo modelo é testado '
                'nas leituras reservadas do microfone de origem e nas captações '
                'pareadas do outro microfone: 1→1, 1→2, 2→1 e 2→2. '
                'O microfone de teste não altera o checkpoint nem as estatísticas '
                'de z-score.'),
              P('Grade executada', 'h2'),
              table([['Condição', 'Modelos', 'Avaliações', 'Combinações resumidas'],
                     ['40 quadros: fala e não fala', '480', '1.920', '384'],
                     ['80 quadros: fala', '240', '480', '96'],
                     ['100 quadros: fala', '240', '480', '96'],
                     ['X-vector, 100 quadros', '20', '40', '8']],
                    style, [215, 70, 90, WIDTH - 375]),
              Spacer(1, 10),
              P('Métricas e leitura correta', 'h2'),
              P('Reportamos acurácia, precisão, revocação e F1 macro, como média '
                'e desvio padrão entre cinco folds. O EER é apenas um diagnóstico '
                'um-contra-resto com softmax de classes fechadas; não equivale à '
                'verificação aberta por embeddings. Os folds compartilham locutores '
                'e não são réplicas independentes.'),
              P('As três durações usam coortes diferentes. Uma diferença entre '
                '40, 80 e 100 quadros mistura duração e seleção de gravações; '
                'não deve ser apresentada como efeito causal isolado da duração.',
                'small'),
              P('Os mesmos 108 locutores participam dos cinco folds: cada '
                'rodada treina novos pesos para identificar leituras reservadas '
                'de pessoas já conhecidas. A média e o desvio entre folds '
                'reduzem a dependência de uma única divisão, mas não medem '
                'generalização a locutores inéditos.', 'small'),
              PageBreak()]

    # Fixed-condition class comparison before the architecture grid.
    story += [P('Fala e não fala: o controle de 40 quadros', 'h1'),
              P('A figura mantém fixos a CNN temporal, o z-score e 40 MFCCs. '
                'Muda apenas a classe do trecho de treino e de teste. '
                'Cada célula é a acurácia média dos mesmos cinco folds.', 'body'),
              figure('06_atividade_resultados.png'),
              P('Fala → fala mantém acurácia alta no mesmo microfone. '
                'Não fala → não fala também permite reconhecimento no mesmo '
                'microfone, porém cai fortemente na transferência. '
                'Trocar a classe entre treino e teste leva as acurácias '
                'para perto do acaso de 1/108 = 0,93%.', 'body'),
              P('“Não fala” é o complemento dos intervalos marcados pelo '
                'Silero nesta gravação; pode incluir respiração, fala fraca '
                'e ruído. O resultado não demonstra que silêncio puro '
                'identifica o locutor.', 'small'),
              P('A coorte de 40 quadros tem 11.880 leituras pareadas. '
                'Cada uma fornece uma janela contínua de fala e outra de '
                'não fala nos dois microfones, com papéis de treino, '
                'validação e teste pareados.', 'small'),
              PageBreak()]

    story += [P('Panorama dos resultados em fala', 'h1'),
              P('Melhor acurácia média observada em cada duração, direção e '
                'quantidade de MFCCs. Os valores resumem CNN, CNN temporal '
                'e, em 100 quadros, x-vector.', 'body')]
    summary = [['Quadros', 'Treino → teste', '20 MFCCs', '30 MFCCs', '40 MFCCs']]
    for frames, rows in data.items():
        pool = rows + (xrows if frames == 100 and xrows else [])
        for source, target in DIRECTIONS:
            winners = [best(pool, train_class='activity', test_class='activity',
                            train_mic=source, test_mic=target, n_mfcc=n)
                       for n in (20, 30, 40)]
            summary.append([str(frames), f'{source} → {target}',
                            *[pct(row['accuracy_mean']) for row in winners]])
    story += [table(summary, style, [67, 135, 104, 104, WIDTH - 410], paddings=4),
              Spacer(1, 10),
              P('Cada célula é o maior valor descritivo entre arquiteturas e '
                'normalizações dentro daquele recorte; não representa um único '
                'modelo comum a todas as células. As tabelas seguintes mostram '
                'média ± desvio padrão e permitem comparar condições fixas.',
                'small'),
              P('A transferência para o outro microfone reduz a acurácia em '
                'relação ao teste no microfone de origem. Como as coortes de '
                '40, 80 e 100 quadros diferem, este panorama não atribui a '
                'mudança exclusivamente à duração da janela.', 'small'),
              PageBreak()]

    # 4-5. Detailed comparable tables.
    for frames in (40, 80, 100):
        rows = data[frames]
        story += [P(f'Resultados | {frames} quadros de fala', 'h1'),
                  P('Acurácia média ± desvio padrão em cinco folds; 40 MFCCs + Δ + ΔΔ '
                    '(120 atributos). Todas as condições desta tabela usam a mesma '
                    'coorte, janela e divisão.', 'small')]
        raw = [['Rede', 'Normalização', '1→1', '1→2', '2→1', '2→2']]
        for arch, label in [('cnn', 'CNN'), ('temporal_cnn', 'CNN temporal')]:
            for norm, norm_label in [('zscore', 'Z-score'), ('cmn', 'CMN'),
                                     ('cmvn', 'CMVN'), ('rasta', 'RASTA')]:
                values = []
                for source, target in DIRECTIONS:
                    row = matches(rows, n_mfcc=40, train_class='activity',
                                  test_class='activity', architecture=arch,
                                  normalization=norm, train_mic=source,
                                  test_mic=target)
                    assert len(row) == 1
                    values.append(pct_sd(row[0]))
                raw.append([label, norm_label, *values])
        story += [table(raw, style, [78, 90, 87, 87, 87, WIDTH - 429], paddings=5),
                  Spacer(1, 12)]
        if frames == 100:
            story += [P('X-vector adicional', 'h2')]
            if xrows:
                xraw = [['Normalização', '1→1', '1→2', '2→1', '2→2']]
                for norm, label in [('zscore', 'Z-score'), ('cmn', 'CMN')]:
                    values = []
                    for source, target in DIRECTIONS:
                        row = matches(xrows, normalization=norm,
                                      train_mic=source, test_mic=target)
                        assert len(row) == 1
                        values.append(pct_sd(row[0]))
                    xraw.append([label, *values])
                story += [table(xraw, style, [100, 103, 103, 103, WIDTH - 409]),
                          Spacer(1, 8),
                          P('20 modelos concluídos; mesma coorte e mesmos folds '
                            'das CNNs de 100 quadros.', 'small'),
                          Spacer(1, 10),
                          figure('07_normalizacoes_cross_100.png'),
                          P('Comparação cruzada fixa em 100 quadros e 40 MFCCs. '
                            'Z-score favorece a x-vector no mesmo microfone; '
                            'CMN eleva as médias cruzadas. Os valores exatos '
                            'e desvios aparecem nas tabelas.', 'small')]
            else:
                current = json.loads((XVECTOR / 'status.json').read_text())
                story += [P(f"Treino em andamento: {current['models_completed']}/20 modelos "
                            'concluídos nesta versão do relatório. Os resultados '
                            'da x-vector serão inseridos quando os cinco folds '
                            'de cada direção estiverem completos.', 'small')]
        story.append(PageBreak())

    # BRSD belongs to a distinct, previously completed 16 kHz experiment.
    story += [P('BRSD: seleção real dos 111 quadros', 'h1'),
              P('O BRSD contém 400 WAVs de 80 locutores, com cinco textos por '
                'pessoa. Os arquivos são convertidos para mono pela média dos '
                'canais. Antes da detecção de fala, o experimento remove apenas '
                'as bordas por RMS relativo de -30 dB, com margens de 100 ms '
                'e 250 ms e fade de 8 ms.', 'body'),
              P('A taxa final é 16 kHz: arquivos nativos de 48 kHz passam por '
                'decimação IIR com antialiasing; os arquivos 106 a 110, de '
                '44,1 kHz, usam resample_poly. Depois vêm pré-ênfase 0,97, '
                'Hamming de 32 ms, 128 bandas mel, log-mel e DCT-II.', 'small'),
              brsd_figure('01_selecao_111_quadros.png'),
              P('Exemplo auditado: BRSD 1.wav. A faixa verde representa os '
                'intervalos Silero após o recorte de bordas; a faixa laranja '
                'é a janela efetivamente usada, quadros 250 a 360 (4,000 a '
                '5,792 s após o recorte). Os manifestos de 30 e 40 MFCCs '
                'registram os mesmos limites nas 400 gravações.', 'small'),
              P('Regra de escolha', 'h2'),
              P('O Silero 6.2.3 atua a 16 kHz, com limiar 0,50, fala mínima '
                '250 ms, silêncio mínimo 100 ms e margem 30 ms. Um quadro '
                'é de fala quando seu centro pertence a um intervalo Silero. '
                'Entre os trechos contínuos que comportam 111 quadros, '
                'seleciona-se a janela de maior soma de RMS normalizado. '
                'A ausência de uma janela elegível interromperia a extração; '
                'as 400 gravações foram processadas.'),
              P('Cada quadro cobre 32 ms, com salto de 16 ms; 111 quadros '
                'cobrem 1,792 s. As derivadas são calculadas dentro da janela '
                'contínua. O BRSD desta rodada tem apenas fala: não foi '
                'treinada uma classe separada de não fala.', 'small'),
              PageBreak()]

    braw = [['Rede', 'Tratamento', '30 MFCCs', '40 MFCCs']]
    for arch, label in [('cnn', 'CNN'), ('temporal_cnn', 'CNN temporal')]:
        for norm, norm_label in [('zscore', 'Z-score'), ('cmn', 'CMN'),
                                 ('cmvn', 'CMVN'), ('rasta', 'RASTA'),
                                 ('cmvn_logmel', 'CMVN log-mel')]:
            braw.append([label, norm_label, *[
                f"{pct(brsd[n, arch, norm]['mean'], 2)} ± "
                f"{pct(brsd[n, arch, norm]['std'], 2)}" for n in (30, 40)]])
    story += [P('BRSD: resultados com 30 e 40 MFCCs', 'h1'),
              P('Acurácia média ± desvio padrão em cinco folds. As duas '
                'representações usam as mesmas 400 gravações, janelas Silero '
                'e divisões; 30/40 MFCCs + Δ + ΔΔ correspondem a 90/120 '
                'atributos por quadro.', 'body'),
              table(braw, style, [115, 145, 128, WIDTH-388], paddings=5),
              Spacer(1, 10),
              P('Divisão e tratamentos', 'h2'),
              P('Cada fold reserva um dos cinco textos por locutor para teste '
                '(80 arquivos), escolhe outro para validação com semente 42 '
                '(80) e usa os três restantes para treino (240). Z-score usa '
                'apenas estatísticas do treino. CMN, CMVN, RASTA e CMVN '
                'log-mel são condições isoladas; a última normaliza as '
                '128 bandas log-mel usando quadros Silero antes da DCT.'),
              P('A maior média é 58,75% na CNN temporal com z-score, tanto '
                'com 30 como com 40 MFCCs. O acaso uniforme é 1,25%. '
                'Como cada locutor do BRSD está associado a um aparelho, '
                'esses resultados não medem transferência entre dispositivos. '
                'As acurácias do BRSD e do VCTK não são diretamente '
                'comparáveis: mudam classes, textos, captação e seleção.',
                'small'),
              PageBreak()]

    # Final page.
    story += [P('Conclusões e limites', 'h1'),
              P('O melhor resultado no mesmo microfone é alto, enquanto a troca '
                'de microfone reduz a acurácia em todas as durações e quantidades '
                'de MFCCs resumidas. Isso revela dependência da cadeia de captação '
                'sob este protocolo.'),
              P('A tabela de 40 quadros permite examinar fala e não fala mantendo '
                'coorte, largura, rede e normalização constantes. A classe não fala '
                'não é equivalente a silêncio puro.'),
              P('O conjunto VCTK compartilha locutores entre folds e contém '
                'gravações relacionadas; este estudo mede identificação fechada '
                'na coorte selecionada. Generalização a novas sessões, aparelhos '
                'ou locutores exige outro protocolo.'),
              P('No BRSD, os 111 quadros foram escolhidos por RMS dentro da '
                'fala Silero, em 400 gravações de 80 locutores. As condições '
                'de 30 e 40 MFCCs compartilham exatamente as janelas. '
                'O dispositivo permanece associado ao locutor, limitando a '
                'interpretação das acurácias.'),
              P('Leitura dos resultados', 'h2'),
              P('As comparações diretas entre redes e normalizações usam a mesma '
                'duração, direção e classe. Cada valor detalhado é a média ± '
                'desvio padrão dos cinco folds. No VCTK, cada direção de teste conserva '
                'o checkpoint e, quando aplicável, o z-score do microfone '
                'de origem.'),
              P('Rastreabilidade', 'h2'),
              P('Protocolos e seleções: output/vctk40_activity_grid_plan_20260929, '
                'output/vctk80_activity_grid_plan_20260929_112427 e '
                'output/vctk100_activity_grid_plan_20260929. '
                'BRSD: brsd_silero_isolated_features e '
                'brsd40_silero_isolated_features no HDD. '
                'Resultados VCTK: resultados_resumo.csv das três rodadas; '
                'x-vector em output/vctk100_xvector_silero_20260929/results. '
                'Resultados BRSD: dynamic/summary.json das 20 condições. '
                'Código: experiments/vctk40_activity_grid.py, '
                'experiments/vctk80_activity_grid.py, '
                'experiments/vctk100_activity_grid.py e '
                'experiments/build_brsd_silero_conditions.py.', 'small')]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=40,
                            rightMargin=40, topMargin=49, bottomMargin=50,
                            title='Reconhecimento de locutor - protocolo Silero VCTK',
                            author='Projeto SR')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    shutil.copy2(OUTPUT, ALIAS)
    shutil.copy2(OUTPUT, BRSD_OUTPUT)
    print(OUTPUT)
    print(ALIAS)
    print(BRSD_OUTPUT)
    print('X-vector:', 'complete' if xrows else 'in progress')


if __name__ == '__main__':
    build()
