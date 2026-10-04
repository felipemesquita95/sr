#!/usr/bin/env python3
"""Create the complete VCTK 40/80/100-frame results booklet."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path('/home/lsmsqt/Documents/sr')
RUNS = [
    (40, Path('/media/lsmsqt/HDD/sr_project/vctk40_activity_nonactivity_20260929')),
    (80, Path('/media/lsmsqt/HDD/sr_project/vctk80_activity_20260929_112427')),
    (100, Path('/media/lsmsqt/HDD/sr_project/vctk100_activity_20260929')),
]
XVECTOR = ROOT / 'output/vctk100_xvector_silero_20260929/results'
OUTPUT = ROOT / 'output/pdf/resultados_vctk_40_80_100_quadros.pdf'
PAGE_W, PAGE_H = landscape(A4)
REPORT_DATE = datetime.now(ZoneInfo('America/Sao_Paulo')).strftime('%d/%m/%Y')
BLUE = colors.HexColor('#17324d')
TEAL = colors.HexColor('#087e8b')
PALE = colors.HexColor('#eaf2f4')
INK = colors.HexColor('#233446')
MUTED = colors.HexColor('#597080')
LINE = colors.HexColor('#d8e2e8')


def read_runs():
    result = {}
    for frames, folder in RUNS:
        status = json.loads((folder / 'status.json').read_text())
        assert status['status'] == 'complete'
        rows = list(csv.DictReader((folder / 'resultados_resumo.csv').open(newline='')))
        assert len(rows) == (384 if frames == 40 else 96)
        assert all(r['folds_completed'] == '5' and r['folds_expected'] == '5' for r in rows)
        result[frames] = (folder, status, rows)
    xstatus = json.loads((XVECTOR / 'status.json').read_text())
    assert xstatus['status'] == 'complete' and xstatus['models_completed'] == 20
    xrows = list(csv.DictReader((XVECTOR / 'resultados_resumo.csv').open(newline='')))
    assert len(xrows) == 8
    assert all(r['folds_completed'] == '5' and r['folds_expected'] == '5' for r in xrows)
    assert all(r['architecture'] == 'xvector' and r['n_mfcc'] == '40'
               and r['train_class'] == r['test_class'] == 'activity' for r in xrows)
    folder, status, rows = result[100]
    result[100] = (folder, dict(status, models_completed=status['models_completed'] + 20),
                   rows + xrows)
    return result


def label_class(value):
    return {'activity': 'fala', 'non_activity': 'não fala'}[value]


def label_arch(value):
    return {'cnn': 'CNN', 'temporal_cnn': 'CNN temporal',
            'xvector': 'X-vector'}[value]


def pct(value, digits=1):
    return f'{float(value) * 100:.{digits}f}'.replace('.', ',')


def footer(c, page, total):
    c.setStrokeColor(LINE)
    c.line(38, 39, PAGE_W - 38, 39)
    c.setFillColor(MUTED)
    c.setFont('DVS', 7.5)
    c.drawString(39, 25, f'VCTK | Resultados completos | {REPORT_DATE}')
    c.drawRightString(PAGE_W - 39, 25, f'{page} / {total}')


def heading(c, eyebrow, title, subtitle):
    c.setFillColor(TEAL)
    c.setFont('DVS-Bold', 9)
    c.drawString(39, PAGE_H - 36, eyebrow.upper())
    c.setFillColor(BLUE)
    c.setFont('DVS-Bold', 18)
    c.drawString(39, PAGE_H - 65, title)
    c.setFillColor(MUTED)
    c.setFont('DVS', 8.5)
    c.drawString(39, PAGE_H - 84, subtitle)


def summary_page(c, data, total):
    heading(c, 'Relatório de resultados', 'VCTK: 40, 80 e 100 quadros',
            'CNN e CNN temporal nas três durações; X-vector em 100 quadros | cinco folds por combinação')
    x0 = 39
    card_y = PAGE_H - 146
    card_w = 238
    for index, (frames, (_, status, rows)) in enumerate(data.items()):
        x = x0 + index * (card_w + 24)
        c.setFillColor(PALE)
        c.roundRect(x, card_y, card_w, 45, 7, fill=1, stroke=0)
        c.setFillColor(BLUE)
        c.setFont('DVS-Bold', 14)
        c.drawString(x + 13, card_y + 23, f'{frames} quadros')
        c.setFont('DVS', 8.4)
        c.drawString(x + 13, card_y + 10, f"{status['models_completed']} modelos  |  {len(rows)} combinações")

    c.setFillColor(BLUE)
    c.setFont('DVS-Bold', 10)
    c.drawString(39, PAGE_H - 178, 'Melhor acurácia por direção de microfone - fala → fala')
    headers = ['Quadros', 'Treino → teste', '20 MFCCs', '30 MFCCs', '40 MFCCs', 'Melhor rede']
    xs = [39, 116, 276, 387, 498, 609]
    y = PAGE_H - 202
    c.setFillColor(BLUE)
    c.roundRect(39, y - 5, PAGE_W - 78, 22, 4, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont('DVS-Bold', 8.2)
    for x, header in zip(xs, headers):
        c.drawString(x + 7, y + 2, header)
    y -= 19
    for frames, (_, _, rows) in data.items():
        for source, target in [('mic1', 'mic1'), ('mic1', 'mic2'), ('mic2', 'mic1'), ('mic2', 'mic2')]:
            found = []
            winners = []
            for n in ('20', '30', '40'):
                eligible = [r for r in rows if r['train_class'] == 'activity'
                            and r['test_class'] == 'activity' and r['train_mic'] == source
                            and r['test_mic'] == target and r['n_mfcc'] == n]
                winner = max(eligible, key=lambda r: float(r['accuracy_mean']))
                found.append(pct(winner['accuracy_mean']))
                winners.append(winner['architecture'])
            if y % 2:
                pass
            c.setFillColor(PALE if (source, target) in [('mic1', 'mic1'), ('mic2', 'mic1')] else colors.white)
            c.rect(39, y - 7, PAGE_W - 78, 18, fill=1, stroke=0)
            values = [str(frames), f'{source} → {target}', *[v + '%' for v in found],
                      label_arch(winners[-1])]
            c.setFillColor(INK)
            c.setFont('DVS', 8.7)
            for x, value in zip(xs, values):
                c.drawString(x + 7, y, value)
            y -= 18

    c.setFillColor(INK)
    c.setFont('DVS-Bold', 8.4)
    c.drawString(39, 103, 'Leitura dos resultados')
    c.setFont('DVS', 7.7)
    c.drawString(39, 89, 'Cada célula acima escolhe a melhor rede e normalização. As páginas seguintes mostram todas as combinações.')
    c.drawString(39, 77, '20, 30 e 40 MFCCs equivalem a 60, 90 e 120 características por quadro, com Δ e ΔΔ.')
    c.drawString(39, 65, 'As coortes diferem entre durações; a diferença entre linhas não mede isoladamente o efeito do tempo de áudio.')
    footer(c, 1, total)
    c.showPage()


def table_page(c, frames, folder, rows, scenario, page, total):
    train_class, test_class, source, target = scenario
    subset = [r for r in rows if r['architecture'] != 'xvector'
              and r['train_class'] == train_class
              and r['test_class'] == test_class and r['train_mic'] == source
              and r['test_mic'] == target]
    assert len(subset) == 24
    order_arch = {'cnn': 0, 'temporal_cnn': 1}
    order_norm = {'zscore': 0, 'cmn': 1, 'cmvn': 2, 'rasta': 3}
    subset.sort(key=lambda r: (order_arch[r['architecture']], int(r['n_mfcc']), order_norm[r['normalization']]))
    title = f'{frames} quadros | {label_class(train_class)} → {label_class(test_class)}'
    subtitle = f'Treino {source} → teste {target}  |  108 locutores  |  média de cinco folds  |  valores em %'
    heading(c, f'Grade detalhada {frames} quadros', title, subtitle)

    left, right = 39, PAGE_W - 39
    widths = [45, 55, 100, 82, 125, 91, 91, 91, 84]
    xs = [left]
    for w in widths[:-1]:
        xs.append(xs[-1] + w)
    headers = ['MFCCs', 'Atributos', 'Rede', 'Norm.', 'Acurácia ± DP', 'Precisão', 'Revocação', 'F1 macro', 'EER*']
    y_top = PAGE_H - 111
    c.setFillColor(BLUE)
    c.roundRect(left, y_top - 1, right - left, 22, 4, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont('DVS-Bold', 8)
    for x, h in zip(xs, headers):
        c.drawString(x + 5, y_top + 7, h)

    y = y_top - 15
    row_h = 17
    for i, r in enumerate(subset):
        best = max((x for x in subset if x['architecture'] == r['architecture']
                    and x['n_mfcc'] == r['n_mfcc']), key=lambda x: float(x['accuracy_mean']))
        if i % 2 == 0:
            c.setFillColor(PALE)
            c.rect(left, y - 6, right - left, row_h, fill=1, stroke=0)
        values = [r['n_mfcc'], str(3 * int(r['n_mfcc'])), label_arch(r['architecture']),
                  r['normalization'].upper() if r['normalization'] != 'zscore' else 'Z-score',
                  f"{pct(r['accuracy_mean'])} ± {pct(r['accuracy_std'])}",
                  pct(r['precision_macro_mean']), pct(r['recall_macro_mean']),
                  pct(r['f1_macro_mean']), pct(r['one_vs_rest_eer_mean'])]
        c.setFillColor(TEAL if r is best else INK)
        c.setFont('DVS-Bold' if r is best else 'DVS', 8)
        for x, value in zip(xs, values):
            c.drawString(x + 5, y, value)
        y -= row_h

    c.setStrokeColor(LINE)
    c.line(left, y + 7, right, y + 7)
    c.setFillColor(MUTED)
    c.setFont('DVS', 7.2)
    c.drawString(left, 58, 'Azul: maior acurácia entre normalizações da mesma rede e quantidade de MFCCs.')
    c.drawString(left, 48, '* EER: diagnóstico um-contra-resto com softmax fechado; não é verificação aberta de locutores.')
    footer(c, page, total)
    c.showPage()


def xvector_page(c, rows, page, total):
    subset = [r for r in rows if r['architecture'] == 'xvector']
    assert len(subset) == 8
    subset.sort(key=lambda r: (r['train_mic'], r['test_mic'],
                               {'zscore': 0, 'cmn': 1}[r['normalization']]))
    heading(c, 'Grade detalhada 100 quadros', 'X-vector | fala → fala',
            '40 MFCCs + Δ + ΔΔ | 108 locutores | média de cinco folds | valores em %')
    left, right = 39, PAGE_W - 39
    widths = [110, 90, 120, 100, 100, 100, 80]
    xs = [left]
    for w in widths[:-1]:
        xs.append(xs[-1] + w)
    headers = ['Treino → teste', 'Norm.', 'Acurácia ± DP',
               'Precisão', 'Revocação', 'F1 macro', 'EER*']
    y_top = PAGE_H - 130
    c.setFillColor(BLUE)
    c.roundRect(left, y_top - 1, right - left, 25, 4, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont('DVS-Bold', 8)
    for x, header in zip(xs, headers):
        c.drawString(x + 6, y_top + 8, header)
    y = y_top - 29
    for index, row in enumerate(subset):
        if index % 2 == 0:
            c.setFillColor(PALE)
            c.rect(left, y - 8, right - left, 27, fill=1, stroke=0)
        values = [f"{row['train_mic']} → {row['test_mic']}",
                  'Z-score' if row['normalization'] == 'zscore' else 'CMN',
                  f"{pct(row['accuracy_mean'])} ± {pct(row['accuracy_std'])}",
                  pct(row['precision_macro_mean']),
                  pct(row['recall_macro_mean']), pct(row['f1_macro_mean']),
                  pct(row['one_vs_rest_eer_mean'])]
        c.setFillColor(INK)
        c.setFont('DVS', 8.5)
        for x, value in zip(xs, values):
            c.drawString(x + 6, y, value)
        y -= 32
    c.setFillColor(MUTED)
    c.setFont('DVS', 8)
    c.drawString(left, 145, 'Mesma coorte e mesmos folds dos resultados CNN e CNN temporal de 100 quadros.')
    c.drawString(left, 128, 'Z-score usa estatísticas somente do treino de origem; CMN subtrai a média dos MFCCs estáticos por janela.')
    c.drawString(left, 111, 'Os desvios padrão descrevem os cinco folds, que compartilham locutores.')
    c.drawString(left, 89, '* EER: diagnóstico um-contra-resto com softmax fechado; não é verificação aberta de locutores.')
    footer(c, page, total)
    c.showPage()


def main():
    pdfmetrics.registerFont(TTFont('DVS', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    pdfmetrics.registerFont(TTFont('DVS-Bold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    data = read_runs()
    scenarios = []
    for frames, (_, _, rows) in data.items():
        classes = [('activity', 'activity')]
        if frames == 40:
            classes = [('activity', 'activity'), ('activity', 'non_activity'),
                       ('non_activity', 'activity'), ('non_activity', 'non_activity')]
        for train_class, test_class in classes:
            for source, target in [('mic1', 'mic1'), ('mic1', 'mic2'), ('mic2', 'mic1'), ('mic2', 'mic2')]:
                scenarios.append((frames, (train_class, test_class, source, target)))
    total = 2 + len(scenarios)
    assert total == 26
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle('Resultados completos VCTK - 40, 80 e 100 quadros')
    c.setAuthor('Projeto SR')
    summary_page(c, data, total)
    for page, (frames, scenario) in enumerate(scenarios, start=2):
        folder, _, rows = data[frames]
        table_page(c, frames, folder, rows, scenario, page, total)
    xvector_page(c, data[100][2], total, total)
    c.save()
    print(OUTPUT)
    print(f'{total} pages | {sum(len(rows) for _, _, rows in data.values())} complete result combinations')


if __name__ == '__main__':
    main()
