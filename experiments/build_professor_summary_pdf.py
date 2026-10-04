#!/usr/bin/env python3
"""Relatório experimental visual de reconhecimento de locutor."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / 'tmp/pdfs/resumo_professor'
OUT = ROOT / 'output/pdf/relatorio_experimental_reconhecimento_locutor.pdf'
NAVY = '#18344a'
TEAL = '#087f82'
ORANGE = '#d97735'


def chosen_record():
    mapping = json.loads((ROOT / 'docs/vctk_bal100_selection.json').read_text())['mapeamento']
    pick = random.Random(42).choice(mapping)
    with (ROOT / 'docs/vctk_activity8k_candidates.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['locutor'] == pick['locutor_original'] and row['enunciado'] == pick['enunciado_original']:
                return pick, row
    raise RuntimeError('Gravação sorteada ausente do manifesto de atividade')


def even(values, size):
    import numpy as np
    return np.asarray(values)[np.linspace(0, len(values) - 1, size, dtype=np.int64)]


def figures():
    import numpy as np
    import soundfile as sf
    import librosa
    from scipy.signal import butter, sosfreqz, welch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sr.preprocessing import signal

    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180})
    pick, row = chosen_record()
    path = (Path('/media/lsmsqt/HDD/datasets/vctk/VCTK-Corpus-0.92/wav48_silence_trimmed') /
            pick['arquivo_original'].split('_')[0] /
            f"{pick['arquivo_original']}_mic1.flac")
    raw, rate = sf.read(path, dtype='float32')
    if raw.ndim == 2:
        raw = raw.mean(axis=1)
    filtered = signal.antialias_filter(raw, rate, 8000)
    narrow = signal.resample(filtered, rate, 8000)
    emphasized = signal.pre_emphasis(narrow, 0.97)
    mfcc = np.load(ROOT / 'runs/features/vctk8k_mic1' /
                   str(pick['locutor_original']) / str(pick['enunciado_original']) / 'mfccs.npy')
    duration = len(raw) / rate

    fig, axes = plt.subplots(4, 1, figsize=(10, 6.5), constrained_layout=True)
    for ax, values, sr, title, color in [
        (axes[0], raw, rate, f"Entrada VCTK silence_trimmed: {rate/1000:g} kHz", NAVY),
        (axes[1], narrow, 8000, 'Filtro 3,6 kHz + reamostragem: 8 kHz', TEAL),
        (axes[2], emphasized, 8000, 'Pré-ênfase: coeficiente 0,97', ORANGE),
    ]:
        step = max(1, len(values) // 4500)
        times = np.arange(0, len(values), step) / sr
        ax.plot(times, values[::step], linewidth=.55, color=color)
        ax.set_xlim(0, duration)
        ax.set_ylabel('Amplitude')
        ax.set_title(title, loc='left', fontweight='bold')
    axes[0].axvspan(0, 77 * .016, color=TEAL, alpha=.14, zorder=0)
    axes[0].axvspan(77 * .016, 153 * .016, color=ORANGE, alpha=.13, zorder=0)
    axes[0].axvline(77 * .016, color=TEAL, ls='--', lw=1)
    axes[0].axvline(153 * .016, color=ORANGE, ls='--', lw=1)
    axes[0].text(77 * .016, .92, '77 quadros', transform=axes[0].get_xaxis_transform(),
                 ha='right', va='top', color=TEAL, fontsize=7, fontweight='bold')
    axes[0].text(153 * .016, .92, '153 quadros', transform=axes[0].get_xaxis_transform(),
                 ha='right', va='top', color=ORANGE, fontsize=7, fontweight='bold')
    axes[3].imshow(mfcc, origin='lower', aspect='auto', cmap='magma',
                   extent=(0, mfcc.shape[1] * .016, 0, 40))
    axes[3].axvspan(0, 77 * .016, color=TEAL, alpha=.20, zorder=2)
    axes[3].axvspan(77 * .016, 153 * .016, color=ORANGE, alpha=.18, zorder=2)
    axes[3].axvline(77 * .016, color=TEAL, ls='--', lw=1, zorder=3)
    axes[3].axvline(153 * .016, color=ORANGE, ls='--', lw=1, zorder=3)
    axes[3].set_xlim(0, duration)
    axes[3].set_ylabel('MFCC')
    axes[3].set_xlabel('Tempo (s)')
    axes[3].set_title('40 MFCCs por quadro de 32 ms, salto de 16 ms', loc='left', fontweight='bold')
    fig.savefig(FIG / 'processamento.png', bbox_inches='tight')
    plt.close(fig)

    # PSD do mesmo arquivo em cada estágio e resposta teórica dos dois filtros.
    fr, pr = welch(raw, fs=rate, nperseg=3072)
    ff, pf = welch(filtered, fs=rate, nperseg=3072)
    fn, pn = welch(narrow, fs=8000, nperseg=512)
    fe, pe = welch(emphasized, fs=8000, nperseg=512)
    db = lambda power, reference: 10 * np.log10(np.maximum(power, 1e-18) / reference)
    ref_raw = max(pr.max(), 1e-18)
    ref_narrow = max(pn.max(), 1e-18)
    sos = butter(8, 3600, btype='low', fs=rate, output='sos')
    wf, hf = sosfreqz(sos, worN=2048, fs=rate)
    wp = np.linspace(0, 4000, 1000)
    hp = np.abs(1 - .97 * np.exp(-2j * np.pi * wp / 8000))
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.0), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(fr / 1000, db(pr, ref_raw), color=NAVY, lw=1.2,
            label='Entrada silence_trimmed, 48 kHz')
    ax.plot(ff / 1000, db(pf, ref_raw), color=TEAL, lw=1.2, label='Após Butterworth')
    ax.axvline(3.6, color=ORANGE, ls='--', lw=1)
    ax.axvline(4.0, color='#777', ls=':', lw=1)
    ax.set(xlim=(0, 8), ylim=(-105, 7), xlabel='Frequência (kHz)',
           ylabel='PSD relativa (dB)', title='Arquivo de entrada e sinal filtrado')
    ax.legend(frameon=False, fontsize=8)
    ax = axes[0, 1]
    ax.plot(wf / 1000, 40 * np.log10(np.maximum(np.abs(hf), 1e-8)),
            color=TEAL, lw=1.6)
    ax.axvline(3.6, color=ORANGE, ls='--', lw=1, label='corte 3,6 kHz')
    ax.axvline(4.0, color='#777', ls=':', lw=1, label='Nyquist 4 kHz')
    ax.set(xlim=(0, 8), ylim=(-100, 5), xlabel='Frequência (kHz)',
           ylabel='Ganho (dB)', title='Resposta do Butterworth (duas passagens)')
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 0]
    ax.plot(fn / 1000, db(pn, ref_narrow), color=TEAL, lw=1.2, label='8 kHz')
    ax.plot(fe / 1000, db(pe, ref_narrow), color=ORANGE, lw=1.2,
            label='Após pré-ênfase')
    ax.set(xlim=(0, 4), ylim=(-85, 10), xlabel='Frequência (kHz)',
           ylabel='PSD relativa (dB)', title='Áudio em 8 kHz e pré-ênfase')
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    ax.plot(wp / 1000, 20 * np.log10(np.maximum(hp, 1e-8)), color=ORANGE, lw=1.6)
    ax.set(xlim=(0, 4), ylim=(-35, 8), xlabel='Frequência (kHz)',
           ylabel='Ganho (dB)', title='Resposta da pré-ênfase (coeficiente 0,97)')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.title.set_fontweight('bold')
    fig.savefig(FIG / 'frequencias.png', bbox_inches='tight')
    plt.close(fig)

    active20 = even(row['active'], 20)
    low20 = even(row['low'], 20)
    active40 = even(row['active'], 40)
    all40 = even(list(range(row['quadros'])), 40)
    fig, axes = plt.subplots(4, 1, figsize=(10, 6.6), sharex=True, constrained_layout=True,
                             gridspec_kw={'height_ratios': [2.1, 1.1, .65, .65]})
    stride = max(1, len(narrow) // 5500)
    axes[0].plot(np.arange(0, len(narrow), stride) / 8000, narrow[::stride],
                 color=NAVY, lw=.55)
    for frame in active20:
        axes[0].axvspan(max(0, frame * .016 - .016), frame * .016 + .016,
                        color=TEAL, alpha=.11, lw=0)
    for frame in low20:
        axes[0].axvspan(max(0, frame * .016 - .016), frame * .016 + .016,
                        color=ORANGE, alpha=.10, lw=0)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f"Gravação real {pick['arquivo_original']} - mic 1, {row['quadros']} quadros MFCC",
                      loc='left', fontweight='bold')
    rms = librosa.feature.rms(y=narrow, frame_length=256, hop_length=128)[0]
    rms_db = 20 * np.log10(np.maximum(rms, 1e-10) / np.max(rms))
    axes[1].plot(np.arange(len(rms_db)) * .016, rms_db, color=NAVY, lw=1.1)
    axes[1].axhline(-30, color=ORANGE, ls='--', lw=1.3, label='limiar −30 dB')
    axes[1].scatter(active20 * .016, rms_db[active20], s=13, color=TEAL,
                    zorder=3, label='20 de atividade')
    axes[1].scatter(low20 * .016, rms_db[low20], s=13, color=ORANGE,
                    zorder=3, label='20 de baixa atividade')
    axes[1].set_ylabel('RMS rel. (dB)')
    axes[1].set_ylim(min(-65, np.percentile(rms_db, 1) - 3), 5)
    axes[1].legend(loc='upper right', frameon=False, fontsize=7, ncol=3)
    axes[2].vlines(np.asarray(row['active']) * .016, 0, 1, color='#acd8cd', lw=1.5)
    axes[2].vlines(active20 * .016, 0, 1, color=TEAL, lw=2.6)
    axes[2].set_ylabel('Atividade')
    axes[3].vlines(np.asarray(row['low']) * .016, 0, 1, color='#f5cfb8', lw=1.5)
    axes[3].vlines(low20 * .016, 0, 1, color=ORANGE, lw=2.6)
    axes[3].set_ylabel('Baixa')
    for ax in axes[2:]:
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.grid(axis='x', alpha=.2)
    axes[3].set_xlim(0, duration)
    axes[3].set_xlabel('Tempo (s); cada barra escura marca o centro de um quadro selecionado')
    fig.savefig(FIG / 'selecao_quadros.png', bbox_inches='tight')
    plt.close(fig)

    padded = np.pad(narrow, (128, 128))
    taper = np.hanning(256)
    def average_frame_power(indices):
        spectra = [np.abs(np.fft.rfft(padded[int(i) * 128:int(i) * 128 + 256] * taper)) ** 2
                   for i in indices]
        return np.mean(spectra, axis=0)
    active_power = average_frame_power(active20)
    low_power = average_frame_power(low20)
    frequency = np.fft.rfftfreq(256, 1 / 8000) / 1000
    reference = max(active_power.max(), 1e-18)
    fig, ax = plt.subplots(figsize=(10, 2.7), constrained_layout=True)
    ax.plot(frequency, db(active_power, reference), color=TEAL, lw=1.6,
            label='20 quadros de atividade')
    ax.plot(frequency, db(low_power, reference), color=ORANGE, lw=1.6,
            label='20 quadros de baixa atividade')
    ax.set(xlim=(0, 4), xlabel='Frequência (kHz)',
           ylabel='Potência relativa (dB)',
           title='Espectro médio dos quadros realmente selecionados')
    ax.title.set_fontweight('bold')
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG / 'espectro_quadros_selecionados.png', bbox_inches='tight')
    plt.close(fig)

    # A mesma gravação fornece as cinco entradas; guardamos os índices para auditoria.
    detail = {'gravacao': pick['arquivo_original'], 'arquivo_de_entrada': str(path),
              'novo_locutor': pick['novo_locutor'],
              'novo_enunciado': pick['novo_enunciado'], 'duracao_s': duration,
              'quadros_totais': row['quadros'], 'atividade_elegivel': len(row['active']),
              'baixa_elegivel': len(row['low']), 'active20': active20.tolist(),
              'low20': low20.tolist(), 'active40': active40.tolist(),
              'unfiltered40': all40.tolist(),
              'mixed40': sorted(active20.tolist() + low20.tolist()),
              'activity_first77': sum(i < 77 for i in row['active']),
              'low_first77': sum(i < 77 for i in row['low']),
              'activity_first153': sum(i < 153 for i in row['active']),
              'low_first153': sum(i < 153 for i in row['low'])}
    (FIG / 'amostra.json').write_text(json.dumps(detail, ensure_ascii=False, indent=2))

    labels = ['CNN', 'Temporal', 'Atenção']
    x = np.arange(3)
    series = [
        ('20 quadros: mic 1 → mic 2', [31.40, 26.72, 30.04], [10.84, 5.92, 7.48], 'Atividade', 'Baixa atividade'),
        ('20 quadros: mic 2 → mic 1', [40.92, 27.56, 38.16], [10.76, 6.60, 7.24], 'Atividade', 'Baixa atividade'),
        ('40 quadros: mic 1 → mic 2', [34.92, 29.60, 32.44], [25.28, 17.08, 20.92], '40 atividade', '20 + 20'),
        ('40 quadros: mic 2 → mic 1', [39.96, 27.28, 36.60], [33.76, 19.56, 29.16], '40 atividade', '20 + 20'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for ax, (title, a, b, la, lb) in zip(axes.flat, series):
        ax.bar(x - .18, a, .36, color=TEAL, label=la)
        ax.bar(x + .18, b, .36, color=ORANGE, label=lb)
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 48)
        ax.set_ylabel('Acurácia (%)')
        ax.set_title(title, loc='left', fontweight='bold')
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(FIG / 'resultados_cross.png', bbox_inches='tight')
    plt.close(fig)

    baseline = [
        ('Mic 1 → Mic 1', [93.09, 96.04, 95.06], [97.40, 98.60, 98.25]),
        ('Mic 2 → Mic 2', [86.21, 92.39, 88.76], [93.91, 97.08, 95.25]),
        ('Mic 1 → Mic 2', [19.29, 10.60, 15.87], [23.68, 13.98, 19.75]),
        ('Mic 2 → Mic 1', [26.63, 16.65, 22.00], [34.96, 22.23, 29.02]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 4.8), constrained_layout=True)
    for ax, (title, short, long) in zip(axes.flat, baseline):
        ax.bar(x - .18, short, .36, color='#8297a5', label='77 quadros')
        ax.bar(x + .18, long, .36, color=TEAL, label='153 quadros')
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 105 if 'Mic 1 → Mic 1' in title or 'Mic 2 → Mic 2' in title else 45)
        ax.set_ylabel('Acurácia (%)')
        ax.set_title(title, loc='left', fontweight='bold')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG / 'resultados_baseline.png', bbox_inches='tight')
    plt.close(fig)


def pdf():
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                    SimpleDocTemplate, Spacer, Table, TableStyle)
    from PIL import Image as PILImage

    pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    pdfmetrics.registerFont(TTFont('DejaVu-Bold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
    pdfmetrics.registerFontFamily('DejaVu', normal='DejaVu', bold='DejaVu-Bold')
    navy = colors.HexColor(NAVY)
    teal = colors.HexColor(TEAL)
    pale = colors.HexColor('#edf5f5')
    gray = colors.HexColor('#56616d')
    st = {
        'title': ParagraphStyle('title', fontName='DejaVu-Bold', fontSize=20, leading=25,
                                textColor=navy, spaceAfter=10),
        'h1': ParagraphStyle('h1', fontName='DejaVu-Bold', fontSize=13, leading=17,
                             textColor=navy, spaceBefore=10, spaceAfter=7),
        'h2': ParagraphStyle('h2', fontName='DejaVu-Bold', fontSize=10.3, leading=14,
                             textColor=teal, spaceBefore=8, spaceAfter=5),
        'body': ParagraphStyle('body', fontName='DejaVu', fontSize=9, leading=13.3,
                               textColor=navy, spaceAfter=6),
        'small': ParagraphStyle('small', fontName='DejaVu', fontSize=7.2, leading=10.3,
                                textColor=gray, spaceAfter=5),
        'th': ParagraphStyle('th', fontName='DejaVu-Bold', fontSize=7.8, leading=10.1,
                             textColor=colors.white),
        'td': ParagraphStyle('td', fontName='DejaVu', fontSize=7.8, leading=10.1,
                             textColor=navy),
    }
    def P(s, kind='body'):
        return Paragraph(s, st[kind])
    def T(rows, widths):
        data = [[P(str(v), 'th' if i == 0 else 'td') for v in row]
                for i, row in enumerate(rows)]
        obj = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
        obj.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), navy),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, pale]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        return obj
    def photo(name, width):
        path = FIG / name
        with PILImage.open(path) as pic:
            height = width * pic.height / pic.width
        return Image(str(path), width=width, height=height)
    def footer(canvas, doc):
        canvas.saveState()
        w, h = A4
        canvas.setStrokeColor(colors.HexColor('#d9e2e7'))
        canvas.line(40, h - 35, w - 40, h - 35)
        canvas.line(40, 36, w - 40, 36)
        canvas.setFont('DejaVu', 7)
        canvas.setFillColor(gray)
        canvas.drawString(40, h - 29, 'RELATÓRIO EXPERIMENTAL - RECONHECIMENTO DE LOCUTOR')
        canvas.drawString(40, 24, 'VCTK e BrSD - 8 kHz')
        canvas.drawRightString(w - 40, 24, f'{doc.page}')
        canvas.restoreState()
    def page(story):
        story.append(PageBreak())

    psd = json.loads((ROOT / 'docs/psd_vctk_8k.json').read_text())
    count1 = sum(r['abaixo_4k'] >= .99 for r in psd['records']['mic1'])
    count2 = sum(r['abaixo_4k'] >= .99 for r in psd['records']['mic2'])
    sample = json.loads((FIG / 'amostra.json').read_text())
    story = []

    # Page 1: scope and audio path.
    story += [Spacer(1, 14), P('Experimentos de reconhecimento de locutor', 'title'),
              P('Protocolo, processamento do áudio e resultados em BrSD e VCTK.', 'body'),
              P('Dados e objetivo', 'h1'),
              P('<b>VCTK:</b> 108 locutores e 21.523 leituras, cada uma gravada nos microfones 1 e 2. '
                '<b>BrSD:</b> 80 locutores e 400 gravações, em uma única captação. '
                'A tarefa é identificar o locutor de uma gravação inédita. Uma amostra é uma gravação; '
                'um quadro é uma pequena janela temporal do áudio.'),
              P('No VCTK, o arquivo efetivamente lido vem da pasta '
                '<b>wav48_silence_trimmed</b>, versão já aparada do corpus. '
                'O nosso processamento não aplica outro corte de silêncio nos testes de 77 e 153 quadros.'),
              P('Processamento do áudio - exemplo real', 'h1'),
              P('Lemos o áudio completo, aplicamos um passa-baixas Butterworth de 8ª ordem em duas passagens '
                '(corte de 3,6 kHz), reamostramos para 8 kHz por método polifásico e aplicamos '
                'pré-ênfase de 0,97. Extraímos 40 MFCCs por quadro de 256 amostras (32 ms), '
                'com salto de 128 (16 ms) e janela Hann. A janela reduz vazamento espectral na análise por FFT.'),
              photo('processamento.png', 500),
              P(f'Arquivo de entrada exibido: <b>wav48_silence_trimmed/p245/{sample["gravacao"]}_mic1.flac</b>. '
                'A faixa verde marca os <b>primeiros 77 quadros</b> desde o começo desse arquivo; '
                'a faixa laranja completa os <b>primeiros 153</b>. As marcas temporais '
                'são aproximadas porque os quadros MFCC são centrados a cada 16 ms.', 'small'),
              P(f'Neste exemplo, entre os primeiros 77 quadros, o detector posterior marcou '
                f'{sample["activity_first77"]} de atividade comum e {sample["low_first77"]} '
                'de baixa atividade comum segura. Entre os primeiros 153, marcou '
                f'{sample["activity_first153"]} e {sample["low_first153"]}, respectivamente. '
                'Essas marcas são apenas diagnósticas: não foram usadas para escolher '
                'os quadros dos experimentos de 77 e 153.', 'small')]
    page(story)

    # Page 2: frequency behavior and population PSD statistic.
    story += [P('Análise em frequência do exemplo', 'h1'),
              P('Os gráficos da esquerda mostram a densidade espectral de potência (PSD) '
                'da mesma gravação em etapas sucessivas. Os da direita mostram o ganho '
                'teórico dos filtros. O Butterworth reduz a região acima de 3,6 kHz; '
                'a pré-ênfase aumenta a participação relativa das frequências mais altas '
                'na banda de 0-4 kHz.'),
              photo('frequencias.png', 500),
              P('As curvas de PSD são relativas ao pico do sinal de referência em cada '
                'comparação; os dB do ganho pertencem aos filtros. O corte de 3,6 kHz '
                'é anterior à reamostragem; a linha de 4 kHz indica o limite de Nyquist '
                'dos sinais já convertidos para 8 kHz.', 'small'),
              P('Potência na banda: amostra de PSD do VCTK', 'h1'),
              P(f'Na análise de PSD, foram sorteadas <b>324 leituras pareadas</b> (3 de cada um dos 108 '
                f'locutores), ou 324 sinais por microfone. Em <b>{count1}/324</b> sinais do mic 1 e '
                f'<b>{count2}/324</b> do mic 2, pelo menos <b>99% da potência</b> estava abaixo de '
                '4 kHz. As medianas foram 98,59% e 99,46%, respectivamente. A banda de 0-4 kHz '
                'é a que pode ser representada após a conversão para 8 kHz.'),
              P('Essa medida diz onde está a potência acústica; não demonstra que toda a informação '
                'que identifica o locutor está nessa banda.', 'small')]
    page(story)

    # Page 3: data split, normalization and architecture.
    story += [P('Divisão e normalização', 'h1'),
              P('Em cada locutor, as gravações são distribuídas em cinco grupos. Cada rodada usa '
                '<b>três grupos para treino, um para validação e um para teste</b> (60/20/20). '
                'As duas captações da mesma leitura recebem o mesmo papel. Média e desvio padrão '
                'de cada MFCC são calculados somente no treino; aplicamos o mesmo z-score à '
                'validação e ao teste. No teste cruzado, a normalização continua sendo a da origem.'),
              P('Redes usadas em ambos os bancos', 'h1'),
              T([['Rede', 'Sequência de camadas', 'Parâmetros no VCTK de 77 quadros'],
                 ['CNN cepstral', 'Conv1D: 32 kernels de 4 MFCCs, stride 1; Flatten; Dense 256 (tanh); softmax', '341.004'],
                 ['CNN temporal', 'Conv1D: 64 kernels de 5 quadros; MaxPool 2; Conv1D: 128 kernels de 5; MaxPool 2; média global; Dense 256; softmax', '114.732'],
                 ['Atenção', 'Projeção 128; 2 blocos com 4 cabeças; Flatten; Dense 256; softmax', '1.618.796']],
                [95, 310, 110]),
              Spacer(1, 8),
              P('<b>CNN temporal, passo a passo (77 quadros):</b> 77×40 → 77×64 → max pooling 38×64 '
                '→ 38×128 → max pooling 19×128 → média global 128 → Dense 256 → locutores. '
                'As convoluções usam ReLU, kernel temporal 5, stride 1 e preenchimento same; '
                'os poolings usam janela 2 e stride 2. A Dense usa tanh e L2=0,04; '
                'segue dropout de 30%. A CNN cepstral usa tanh, L2=0,04 na Dense e dropout 0.'),
              P('No BrSD (1.007 quadros), os tamanhos são: CNN 452.848, temporal 107.536 e atenção '
                '1.730.640 parâmetros. A saída tem 80 classes no BrSD, 108 no VCTK completo '
                'e 100 no controle balanceado.', 'small')]
    page(story)

    # Next page: baseline and cross.
    story += [P('Quatro testes entre microfones', 'h1'),
              P('Em cada rodada, treinamos e validamos uma rede no mic 1 e a testamos primeiro '
                'na partição reservada do mic 1 (1→1), depois nas mesmas leituras captadas pelo '
                'mic 2 (1→2). Repetimos com treino/validação no mic 2 para 2→2 e 2→1. '
                'O teste cruzado usa o mesmo checkpoint; nenhuma leitura de teste aparece no treino.'),
              P('"Primeiros" significa os quadros iniciais do arquivo <b>silence_trimmed</b> '
                'após a extração dos MFCCs. O corte não pula quadros classificados '
                'como baixa atividade que ainda permaneçam no início do arquivo.', 'small'),
              P('VCTK: primeiros 77 quadros (21.523 leituras pareadas)', 'h2'),
              T([['Rede', '1→1', '2→2', '1→2', '2→1'],
                 ['CNN', '93,09%', '86,21%', '19,29%', '26,63%'],
                 ['Temporal', '96,04%', '92,39%', '10,60%', '16,65%'],
                 ['Atenção', '95,06%', '88,76%', '15,87%', '22,00%']],
                [135, 93, 93, 93, 91]),
              P('VCTK: primeiros 153 quadros (18.067 leituras pareadas)', 'h2'),
              T([['Rede', '1→1', '2→2', '1→2', '2→1'],
                 ['CNN', '97,40%', '93,91%', '23,68%', '34,96%'],
                 ['Temporal', '98,60%', '97,08%', '13,98%', '22,23%'],
                 ['Atenção', '98,25%', '95,25%', '19,75%', '29,02%']],
                [135, 93, 93, 93, 91]),
              P('Os testes 1→1 e 2→2 mostram reconhecimento alto mantendo o microfone; '
                'a grande queda em 1→2 e 2→1 mostra dependência da captação. '
                'O teste cruzado mantém a leitura e a sessão, portanto não isola a voz pura.'),
              P('BrSD: teste na única captação disponível', 'h2'),
              T([['CNN', 'CNN temporal', 'Atenção'], ['72,25%', '76,75%', '47,50%']],
                [172, 171, 172]),
              P('Para atribuir o ganho ao comprimento, também refizemos 77 quadros nas mesmas '
                '18.067 gravações e partições de 153. No mesmo microfone, 153 melhorou as seis '
                'combinações de rede e microfone (ganhos de 2,64 a 7,02 pontos percentuais). '
                'O corte de 153 maximiza gravações × quadros com uma largura fixa nesta seleção.'),
              photo('resultados_baseline.png', 500),
              P('Valores: médias de cinco rodadas. 77 completo e 153 têm coortes diferentes; '
                'a comparação de duração usa a coorte pareada de 18.067.', 'small')]
    page(story)

    # Next page: activity method and real sample.
    story += [P('Seleção de atividade e baixa atividade', 'h1'),
              P('Após converter o áudio completo para 8 kHz, um detector de energia '
                '(librosa.effects.split, top_db=30, quadro 256, salto 128) marca atividade. '
                'Só aceitamos quadros em que os dois microfones concordam e afastamos '
                'as fronteiras em um quadro. O limiar é −30 dB em relação ao pico de energia '
                'da própria gravação, não um valor absoluto. Baixa atividade é uma medida de energia: '
                'pode incluir fala fraca, respiração e ruído.'),
              P('Para manter igual número de gravações por locutor e divisão 60/20/20, '
                'selecionamos 100 locutores com 25 gravações pareadas cada. Cada gravação '
                'tem pelo menos 20 quadros seguros de baixa atividade e 40 de atividade. '
                'Em cada rodada, cada locutor fornece 15 gravações de treino, 5 de validação '
                'e 5 de teste. Os quadros escolhidos cobrem o áudio inteiro e podem não ser contíguos. '
                'As condições têm 20 ativos, 20 de baixa atividade, 40 sem seleção, 40 ativos ou 20+20.'),
              photo('selecao_quadros.png', 465),
              P(f'Gravação escolhida de forma reprodutível (semente 42): <b>{sample["gravacao"]}</b>, '
                f'{sample["duracao_s"]:.2f} s, {sample["quadros_totais"]} quadros MFCC; '
                f'{sample["atividade_elegivel"]} elegíveis de atividade e '
                f'{sample["baixa_elegivel"]} de baixa atividade. As barras escuras mostram '
                'os 20 quadros de cada tipo usados na condição 20+20. Cada quadro cobre '
                '32 ms, com centros espaçados em 16 ms.', 'small'),
              P('A curva de RMS em dB mostra os níveis ao longo do tempo; a linha tracejada '
                'marca −30 dB. Os pontos verdes e laranja indicam os quadros efetivamente '
                'escolhidos. A seleção final exige concordância entre os dois microfones.', 'small'),
              photo('espectro_quadros_selecionados.png', 465),
              P('Espectro médio dos 20 quadros marcados em cada condição, antes da pré-ênfase. '
                'A curva de baixa atividade tem menor potência no conjunto. O critério '
                'de −30 dB é aplicado à energia RMS de cada quadro no tempo (gráfico acima), '
                'não a cada frequência deste espectro.', 'small'),
              P('Exemplos da seleção: <b>quadro 61 de atividade</b> (0,960-0,992 s) e '
                '<b>quadro 5 de baixa atividade</b> (0,064-0,096 s). '
                'Os outros 38 quadros estão marcados no gráfico. As redes foram treinadas '
                'por condição e microfone, mantendo gravações e partições.', 'small')]
    page(story)

    # Page 5: activity 20 results.
    story += [P('Resultados: 20 quadros de cada tipo', 'h1'),
              P('Comparação justa em largura: cada entrada tem 20 quadros. '
                'São 100 locutores, 2.500 leituras pareadas e cinco rodadas.'),
              T([['Teste', 'Rede', '20 atividade', '20 baixa'],
                 ['1→1', 'CNN', '91,64%', '75,40%'], ['1→1', 'Temporal', '83,24%', '68,08%'],
                 ['1→1', 'Atenção', '89,04%', '68,96%'],
                 ['2→2', 'CNN', '82,32%', '58,08%'], ['2→2', 'Temporal', '75,44%', '54,12%'],
                 ['2→2', 'Atenção', '74,96%', '53,20%'],
                 ['1→2', 'CNN', '31,40%', '10,84%'], ['1→2', 'Temporal', '26,72%', '5,92%'],
                 ['1→2', 'Atenção', '30,04%', '7,48%'],
                 ['2→1', 'CNN', '40,92%', '10,76%'], ['2→1', 'Temporal', '27,56%', '6,60%'],
                 ['2→1', 'Atenção', '38,16%', '7,24%']],
                [80, 150, 143, 142]),
              Spacer(1, 9),
              P('<b>Leitura:</b> atividade venceu baixa atividade em todas as redes, '
                'tanto dentro do mesmo microfone quanto no teste cruzado. '
                'Baixa atividade ainda supera o acaso de 1% dentro do mesmo microfone, '
                'mas isso não prova que silêncio puro identifica pessoas.', 'body')]
    page(story)

    # Page 6: activity 40 results and cross plot.
    story += [P('Resultados: 40 quadros e mistura 20+20', 'h1'),
              P('As três entradas têm o mesmo tamanho: 40 posições. Na mistura, '
                'metade dos quadros ativos é substituída por baixa atividade.'),
              T([['Teste', 'Rede', '40 sem seleção', '40 atividade', '20+20'],
                 ['1→1', 'CNN', '93,68%', '93,60%', '93,96%'],
                 ['1→1', 'Temporal', '86,64%', '88,08%', '87,48%'],
                 ['1→1', 'Atenção', '92,04%', '89,84%', '91,60%'],
                 ['2→2', 'CNN', '85,36%', '85,60%', '84,32%'],
                 ['2→2', 'Temporal', '78,84%', '83,32%', '81,28%'],
                 ['2→2', 'Atenção', '79,00%', '77,76%', '80,60%'],
                 ['1→2', 'CNN', '28,08%', '34,92%', '25,28%'],
                 ['1→2', 'Temporal', '17,08%', '29,60%', '17,08%'],
                 ['1→2', 'Atenção', '20,64%', '32,44%', '20,92%'],
                 ['2→1', 'CNN', '35,64%', '39,96%', '33,76%'],
                 ['2→1', 'Temporal', '20,80%', '27,28%', '19,56%'],
                 ['2→1', 'Atenção', '28,44%', '36,60%', '29,16%']],
                [65, 105, 120, 120, 105]),
              Spacer(1, 6),
              photo('resultados_cross.png', 500),
              P('Nas transferências entre microfones, 40 quadros ativos superaram '
                '20+20 em todas as três redes e nos dois sentidos. Dentro do mesmo '
                'microfone, a mistura não teve ganho consistente. As condições '
                'compartilham locutores, gravações e partições.', 'small')]
    page(story)
    story += [P('Conclusões e limites', 'h1'),
              P('As redes identificam locutores com alta acurácia no mesmo microfone. '
                'A troca do microfone reduz muito o desempenho: há dependência de '
                'características da captação. Os quadros classificados como atividade carregam mais informação '
                'transferível do que os quadros de baixa atividade sob este detector. '
                'Os resultados não autorizam chamar baixa atividade de silêncio puro '
                'nem afirmar que o teste cruzado mede somente identidade vocal.'),
              P('Resultados centrais', 'h2'),
              T([['Pergunta', 'Exemplo do resultado'],
                 ['A banda de 0-4 kHz concentra 99% da potência?',
                  '122/324 sinais no mic 1; 221/324 no mic 2 (amostra de PSD).'],
                 ['A troca de mic afeta o reconhecimento?',
                  'CNN com 153 quadros: 97,40% em 1→1; 23,68% em 1→2.'],
                 ['Atividade carrega mais que baixa atividade?',
                  'CNN com 20 quadros, 1→2: 31,40% contra 10,84%.'],
                 ['A mistura melhora a transferência?',
                  'CNN com 40 quadros, 1→2: atividade 34,92%; mistura 25,28%.']],
                [242, 273]),
              Spacer(1, 10),
              P('Os resultados de 100, 108 e 80 classes pertencem a protocolos '
                'diferentes. Comparações causais são feitas dentro da mesma coorte, '
                'com as mesmas gravações, partições e número de quadros.', 'body'),
              P('Fontes e rastreabilidade', 'h1'),
              P('Código: src/sr/preprocessing/signal.py; src/sr/features/adjustment.py; '
                'src/sr/models/convolutional.py; src/sr/models/attention.py; '
                'experiments/prepare_vctk_activity_probe.py; '
                'experiments/prepare_vctk_balanced100.py.', 'small'),
              P('Resultados: docs/relatorio_8k.md; docs/comparacao_pareada_77_153.md; '
                'docs/resultado_vctk_bal100_atividade.md; docs/psd_vctk_8k.json; '
                'docs/vctk_bal100_selection.json.', 'small'),
              P(f'Gravação ilustrativa: {sample["gravacao"]}, mic 1. '
                'Seleção reprodutível: sorteio com semente 42 sobre o mapeamento '
                'das 2.500 gravações; índices de quadros extraídos pelo algoritmo '
                'do experimento.', 'small')]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=40, rightMargin=40,
                            topMargin=48, bottomMargin=49, title='Relatório experimental de reconhecimento de locutor',
                            author='Projeto de reconhecimento de locutor')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('figures', 'pdf'):
        raise SystemExit('Uso: build_professor_summary_pdf.py figures|pdf')
    if sys.argv[1] == 'figures':
        figures()
    else:
        pdf()
