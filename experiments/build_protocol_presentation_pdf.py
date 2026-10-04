#!/usr/bin/env python3
"""Update the professor report with the current 16-kHz matched protocol."""
from __future__ import annotations
import json
import shutil
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STORAGE = Path('/media/lsmsqt/HDD/sr_project')
FIG = ROOT/'tmp/pdfs/protocolo16k'
OUT = ROOT/'output/pdf/relatorio_experimental_reconhecimento_locutor.pdf'
CONDITIONS = ('zscore', 'cmn', 'cmvn', 'rasta', 'cmvn_logmel')
LABELS = dict(zscore='Z-score', cmn='CMN', cmvn='CMVN MFCC', rasta='RASTA', cmvn_logmel='CMVN log-mel')
ARCH = dict(cnn='CNN', temporal_cnn='CNN temporal')
DIRS = ('mic1->mic1', 'mic1->mic2', 'mic2->mic1', 'mic2->mic2')
NAVY, TEAL, ORANGE = '#18344a', '#087f82', '#d97735'


def snapshot():
    data = dict(at=datetime.now(ZoneInfo('America/Sao_Paulo')).strftime('%d/%m/%Y às %H:%M'), results={})
    forty = json.loads((ROOT/'output/normalization40_status.json').read_text())
    data['forty_status'], data['forty_stage'] = forty['status'], forty['current_stage']
    stages40 = {s['name']: s for s in forty['stages']}
    for n in (30, 40):
        for dataset in ('vctk', 'brsd'):
            for architecture in ARCH:
                for condition in CONDITIONS:
                    if n == 40:
                        stage = stages40[f'{dataset}/{architecture}/{condition}']
                        if stage.get('status') != 'complete':
                            continue
                        result = Path(stage['result_root'])
                    elif dataset == 'brsd':
                        result = STORAGE/f'isolated_silero_{condition}_brsd_{architecture}_gpu_b128'
                    elif condition == 'zscore':
                        result = STORAGE/f'baseline_vctk16_{architecture}_gpu_b128'
                    elif condition == 'cmvn_logmel':
                        result = STORAGE/f'isolated_logmel_cmvn_vctk16_{architecture}_gpu_b128'
                    else:
                        result = STORAGE/f'isolated_{condition}_vctk16_{architecture}_gpu_b128'
                    channels = ('audio',) if dataset == 'brsd' else ('mic1', 'mic2')
                    for fold in range(1, 6):
                        for channel in channels:
                            p = result/'dynamic'/f'fold{fold}'/channel/'completed.json'
                            completed = json.loads(p.read_text())
                            assert completed.get('n_mfcc', 30) == n
                            assert completed.get('global_zscore', True) == (condition == 'zscore')
                    value = json.loads((result/'dynamic/summary.json').read_text())
                    assert all(len(v['accuracy']['fold_values']) == 5 for v in value['directions'].values())
                    data['results'][f'{n}/{dataset}/{architecture}/{condition}'] = value['directions']
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG/'snapshot.json').write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    return data


def figures(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import soundfile as sf
    from scipy.signal import decimate
    from process_vctk_pair_corrected import soft_crop
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                         'axes.spines.top':False, 'axes.spines.right':False, 'savefig.dpi':180})
    manifest = json.loads((ROOT/'output/vctk16_corrected_features/p225/241/manifest.json').read_text())
    audio, rate = sf.read(manifest['inputs']['mic1'], dtype='float64')
    start, end = manifest['trim_start_sample'], manifest['trim_end_sample']
    reduced = decimate(soft_crop(audio, start, end, rate, 8), q=3, n=8, ftype='iir', zero_phase=True)
    first, last = manifest['selected_window_start_frame'], manifest['selected_window_end_frame']
    window_begin, window_end = first*.016, (last-1)*.016+.032
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.0), constrained_layout=True)
    step = max(1, len(audio)//4500)
    axes[0].plot(np.arange(0,len(audio),step)/rate, audio[::step], color=NAVY, lw=.6)
    axes[0].axvspan(start/rate, end/rate, color=TEAL, alpha=.18, label='Recorte suave retido')
    axes[0].set_title('Entrada real: p225_241_mic1, 48 kHz', loc='left', fontweight='bold')
    axes[0].legend(loc='upper right', frameon=False, fontsize=8)
    axes[0].set_ylabel('Amplitude')
    step = max(1, len(reduced)//4500)
    axes[1].plot(np.arange(0,len(reduced),step)/16000, reduced[::step], color=TEAL, lw=.6)
    axes[1].axvspan(window_begin,window_end,color=ORANGE,alpha=.22,label='111 quadros contínuos')
    axes[1].set_title('Após recorte e decimação para 16 kHz',loc='left',fontweight='bold')
    axes[1].set_xlabel('Tempo (s)'); axes[1].set_ylabel('Amplitude')
    axes[1].legend(loc='upper right',frameon=False,fontsize=8)
    fig.savefig(FIG/'audio_atual.png');plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,3.0),constrained_layout=True)
    paths = (ROOT/'output/vctk16_corrected_features/p225/241/mic1.npz',
             STORAGE/'vctk40_isolated_features/baseline/p225/241/mic1.npz')
    for ax,p,n in zip(axes,paths,(30,40)):
        with np.load(p) as features: static = features['mfcc']
        ax.imshow(static,origin='lower',aspect='auto',cmap='magma',extent=(0,111,0,n))
        ax.set_title(f'{n} MFCCs estáticos: mesma janela',loc='left',fontweight='bold')
        ax.set_xlabel('Quadro');ax.set_ylabel('Coeficiente')
    fig.savefig(FIG/'mfcc30_40.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.3),constrained_layout=True)
    x=np.arange(5)
    for ax,direction in zip(axes,('mic1->mic2','mic2->mic1')):
        vals=[100*data['results'][f'30/vctk/temporal_cnn/{c}'][direction]['accuracy']['mean'] for c in CONDITIONS]
        errs=[100*data['results'][f'30/vctk/temporal_cnn/{c}'][direction]['accuracy']['std'] for c in CONDITIONS]
        ax.bar(x,vals,yerr=errs,color=[NAVY,TEAL,'#72b0b1',ORANGE,'#bdb6ce'],capsize=3)
        ax.set_xticks(x,['Z-score','CMN','CMVN','RASTA','Log-mel'],rotation=22)
        ax.set_ylim(0,100);ax.set_ylabel('Acurácia (%)');ax.set_title(direction.replace('->',' → '),loc='left',fontweight='bold')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(FIG/'vctk30_cross.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.5),constrained_layout=True)
    for architecture,offset,color in (('cnn',-.18,NAVY),('temporal_cnn',.18,TEAL)):
        vals=[100*data['results'][f'30/brsd/{architecture}/{c}']['audio->audio']['accuracy']['mean'] for c in CONDITIONS]
        errs=[100*data['results'][f'30/brsd/{architecture}/{c}']['audio->audio']['accuracy']['std'] for c in CONDITIONS]
        ax.bar(x+offset,vals,.36,yerr=errs,label=ARCH[architecture],color=color,capsize=3)
    ax.axhline(1.25,color=ORANGE,ls='--',lw=1,label='Acaso: 1,25%')
    ax.set_xticks(x,[LABELS[c] for c in CONDITIONS]);ax.set_ylim(0,75);ax.set_ylabel('Acurácia (%)')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=8)
    fig.savefig(FIG/'brsd30.png');plt.close(fig)
    return manifest


def main():
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from PIL import Image as PILImage
    data = snapshot(); manifest=figures(data)
    archive = OUT.parent/'arquivo_protocolo_anterior/relatorio_experimental_2026-09-24.pdf'
    if OUT.exists() and not archive.exists():
        archive.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OUT,archive)
    for name,file in [('DV','DejaVuSans.ttf'),('DV-B','DejaVuSans-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name,'/usr/share/fonts/truetype/dejavu/'+file))
    pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-B',italic='DV',boldItalic='DV-B')
    styles={
        'title':ParagraphStyle('title',fontName='DV-B',fontSize=22,leading=28,textColor=colors.HexColor(NAVY),spaceAfter=14),
        'h':ParagraphStyle('h',fontName='DV-B',fontSize=15,leading=20,textColor=colors.HexColor(NAVY),spaceAfter=10),
        'sub':ParagraphStyle('sub',fontName='DV-B',fontSize=10.5,leading=14,textColor=colors.HexColor(TEAL),spaceBefore=10,spaceAfter=5),
        'body':ParagraphStyle('body',fontName='DV',fontSize=9.1,leading=13.5,textColor=colors.HexColor(NAVY),spaceAfter=8),
        'small':ParagraphStyle('small',fontName='DV',fontSize=7.8,leading=11,textColor=colors.HexColor('#56616d'),spaceAfter=7),
        'cell':ParagraphStyle('cell',fontName='DV',fontSize=7.7,leading=10.5,textColor=colors.HexColor(NAVY)),
        'head':ParagraphStyle('head',fontName='DV-B',fontSize=7.7,leading=10.5,textColor=colors.white),
    }
    story=[];md=[]
    def p(text,kind='body'):
        story.append(Paragraph(text,styles[kind]));md.append(text.replace('<b>','**').replace('</b>','**')+'\n')
    def title(text):
        if story:story.append(PageBreak())
        p(text,'h');md[-1]='# '+md[-1]
    def table(rows,widths):
        obj=Table([[Paragraph(escape(str(v)),styles['head' if i==0 else 'cell']) for v in row] for i,row in enumerate(rows)],colWidths=widths,repeatRows=1)
        obj.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(NAVY)),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#edf5f5')]),
            ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),
            ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        story.extend([obj,Spacer(1,8)])
        md.append('| '+' | '.join(map(str,rows[0]))+' |\n')
        md.append('| '+' | '.join(['---']*len(rows[0]))+' |\n')
        md.extend(['| '+' | '.join(map(str,row))+' |\n' for row in rows[1:]]);md.append('\n')
    def figure(name,width=510):
        with PILImage.open(FIG/name) as im:height=width*im.height/im.width
        story.extend([Image(str(FIG/name),width=width,height=height),Spacer(1,7)])
    def accuracy(n,dataset,architecture,condition,direction):
        key=f'{n}/{dataset}/{architecture}/{condition}'
        if key not in data['results']:return 'Pendente'
        v=data['results'][key][direction]['accuracy']
        return f"{100*v['mean']:.2f} ± {100*v['std']:.2f}%".replace('.',',')
    def footer(canvas,doc):
        canvas.saveState();w,h=A4;canvas.setStrokeColor(colors.HexColor('#d9e2e7'))
        canvas.line(40,h-35,w-40,h-35);canvas.line(40,36,w-40,36)
        canvas.setFillColor(colors.HexColor('#56616d'));canvas.setFont('DV',7)
        canvas.drawString(40,h-29,'RELATÓRIO EXPERIMENTAL - PROTOCOLO ATUALIZADO')
        canvas.drawString(40,24,'VCTK e BRSD | 16 kHz | 30 e 40 MFCCs')
        canvas.drawRightString(w-40,24,str(doc.page));canvas.restoreState()

    p('Reconhecimento de locutor','title')
    p('Protocolo pareado de 16 kHz e normalizações separadas','sub')
    p(f'Atualização de {data["at"]}. Resultados de 30 MFCCs concluídos; a rodada de 40 permanece em execução nesta fotografia dos dados.','small')
    p('<b>Objetivo:</b> identificar o locutor em uma gravação reservada para teste e medir quanto a classificação muda quando o microfone de treino difere do microfone de teste.')
    table([['Corpus','Coorte desta rodada','Captação'],['VCTK','108 locutores × 120 leituras = 12.960 pares','Dois microfones por leitura'],['BRSD','80 locutores × 5 textos = 400 arquivos','Um aparelho por locutor']],[70,272,173])
    p('Processamento do áudio - exemplo real','sub');figure('audio_atual.png')
    p(f'VCTK p225_241_mic1: recorte de {manifest["trim_start_seconds"]:.3f} a {manifest["trim_end_seconds"]:.3f} s do arquivo de entrada. A janela de 111 quadros é comum ao par mic1/mic2, contínua e escolhida por RMS. O tempo do segundo painel começa no recorte.','small')
    p('Esta versão substitui o relato de 8 kHz, Hann, filtro Butterworth separado e janelas de 77/153 quadros. Esses experimentos anteriores ficam arquivados; seus números não entram nas tabelas atuais.','small')

    title('1. Cadeia de processamento vigente')
    p('A frequência de saída é 16 kHz nos dois corpora. O pré-processamento e a janela são mantidos entre condições; a intervenção muda apenas a normalização ou a representação indicada.')
    table([['Etapa','Configuração atual'],
        ['Leitura','VCTK: FLAC mono nativo de 48 kHz. BRSD: WAV; arquivos multicanais convertidos em mono pela média dos canais.'],
        ['Recorte de bordas','RMS com limiar relativo de -30 dB, cinco quadros consecutivos; preservar pausas internas. VCTK: união da atividade dos dois microfones.'],
        ['Margens e suavização','100 ms no início, 250 ms no fim; fade de 8 ms nas bordas recortadas.'],
        ['48 → 16 kHz','decimate(q=3, n=8, ftype=iir, zero_phase=True). Antialiasing Chebyshev I interno; nenhum Butterworth separado.'],
        ['Exceção BRSD','Arquivos 106-110 são nativos de 44,1 kHz: resample_poly diretamente para 16 kHz, com seu filtro antialiasing interno.'],
        ['Pré-ênfase','Coeficiente 0,97 depois da redução da taxa.'],
        ['Análise espectral','Hamming de 32 ms (512 amostras); salto de 16 ms (256 amostras); center=False; 128 bandas mel.'],
        ['Representação','Log-mel e DCT-II ortonormal; reter 30 ou 40 MFCCs. Calcular Δ e ΔΔ apenas na janela selecionada, largura de nove quadros.'],
        ['Janela VCTK','111 quadros contínuos: maximizar a soma de RMS normalizado dos dois canais; mesmos índices nas duas trilhas e em todas as condições.'],
        ['Janela BRSD','111 quadros contínuos dentro de uma região de fala Silero; escolher por RMS. Mesma janela e máscara em todas as condições.']],[105,410])
    p('Silero 6.2.3: limiar 0,5; fala mínima de 250 ms; silêncio mínimo de 100 ms; margem de fala de 30 ms. Um quadro conta como fala quando seu centro pertence ao intervalo detectado. No BRSD, ausência de um trecho contínuo de 111 quadros interrompe a extração para inspeção.','small')

    title('2. De 30 para 40 coeficientes')
    p('Uma gravação é uma amostra. Um quadro é uma janela temporal sobreposta de 32 ms. A janela final contém 111 quadros; sua cobertura no áudio é de 1,792 s (110 saltos de 16 ms mais a última janela de 32 ms).')
    figure('mfcc30_40.png')
    table([['Representação','Estáticos','Δ','ΔΔ','Entrada por gravação'],['30 MFCCs','30','30','30','90 × 111'],['40 MFCCs','40','40','40','120 × 111']],[120,63,63,63,206])
    p('Os dez coeficientes adicionais são reextraídos do mesmo áudio; não são obtidos por preenchimento. A coorte, os recortes, a janela, as partições e as sementes permanecem iguais. Os primeiros 30 coeficientes foram conferidos contra a referência em amostras dos dois corpora.')
    p('Redes desta comparação','sub')
    p('<b>CNN:</b> convoluções sobre o eixo das características na entrada original. <b>CNN temporal:</b> permuta a entrada para aplicar as convoluções ao longo dos quadros. A normalização pode afetar cada arquitetura de forma diferente; a comparação mantém a mesma rede entre 30 e 40.')
    p('A rodada ativa contém somente CNN e CNN temporal. X-vector, attention e combinações de normalizações não fazem parte desta rodada.','small')

    title('3. Cinco tratamentos separados')
    p('Z-score é a referência. CMN, CMVN, RASTA e CMVN log-mel são testados individualmente, sem acrescentar o z-score global e sem combinar os tratamentos entre si.')
    table([['Condição','Domínio e estatísticas','Transformação'],
        ['Z-score','90/120 características; média e desvio estimados apenas nas gravações do treino de origem.','(x - média_treino) / (desvio_treino + 1e-8). Mesmas estatísticas no treino, validação e ambos os testes.'],
        ['CMN','MFCCs estáticos; média temporal da própria janela de 111 quadros.','Subtrair a média de cada coeficiente. Δ e ΔΔ mantidos.'],
        ['CMVN MFCC','MFCCs estáticos; média e desvio da própria janela de 111 quadros.','Centrar e dividir pelo desvio (piso 1e-8). Escalar Δ e ΔΔ pelo mesmo desvio estático.'],
        ['RASTA','Trajetórias das 128 bandas log-mel do áudio recortado completo, antes da DCT.','Filtragem temporal causal; depois DCT, janela fixa e deltas. Sem CMN/CMVN extra.'],
        ['CMVN log-mel','Cada banda log-mel; estatísticas nos quadros de fala Silero do próprio áudio recortado, antes da DCT.','Centrar e escalar as bandas; depois DCT, janela fixa e deltas. Sem CMN extra nem RASTA.']],[85,195,235])
    p('RASTA implementado','sub')
    p('Numerador [0,2; 0,1; 0; -0,1; -0,2]; denominador [1; -0,94]. Inicialização estacionária a partir do primeiro quadro; sem compensação de atraso. Os coeficientes canônicos são mantidos com salto de 16 ms, portanto a resposta em Hz difere da implementação habitual com salto de 10 ms.','small')
    p('No CMVN log-mel VCTK, se Silero não encontrar fala, as estatísticas usam todos os quadros do recorte e o fallback é avisado. Isso não muda a janela de avaliação. No BRSD, a seleção exige fala contínua suficiente e interrompe quando ela falta.','small')
    p('As tabelas anteriores de CMN/RASTA acompanhados de z-score descrevem outra condição experimental. Elas são preservadas no histórico, mas não são misturadas aos resultados isolados abaixo.','small')

    title('4. Como os testes são separados do treino')
    table([['Aspecto','VCTK','BRSD'],['Classes','108 locutores','80 locutores'],['Cinco folds','Divisão por leitura: 60% treino, 20% validação, 20% teste.','Leave-one-text-out: um texto reservado para teste em cada fold.'],['Por fold','7.776 pares de treino; 2.592 de validação; 2.592 de teste.','240 arquivos de treino; 80 de validação; 80 de teste.'],['Validação','24 leituras por locutor, distintas das 72 de treino e 24 de teste.','Uma leitura dos quatro textos restantes sorteada por locutor; semente 42.'],['Direções','Treinar no mic1 ou mic2; testar o mesmo checkpoint em mic1 e mic2.','Uma direção audio → audio.'],['Treino','GPU obrigatória; lote 128; máximo 150 épocas, paciência 15.','GPU obrigatória; lote 128; máximo 1.000 épocas, paciência 30.']],[90,217,208])
    p('Escolha do modelo e retomada','sub')
    p('O checkpoint é escolhido pela acurácia da validação de origem. O teste não decide a época. Em uma interrupção, o treino retoma o melhor checkpoint salvo; resultados concluídos e protocolos compatíveis são reaproveitados.')
    p('Estatísticas e comparabilidade','sub')
    p('A média e o desvio das tabelas são calculados entre cinco folds. Os folds compartilham locutores e conjuntos de treino; o desvio não é um intervalo de confiança. Diferenças pequenas de médias não demonstram significância estatística.')
    p('Limites da interpretação','sub')
    p('No VCTK, a transferência compara capturas pareadas da mesma leitura e sessão. Uma melhora cruzada não prova isolamento da identidade vocal dos efeitos de sessão. No BRSD há um aparelho por locutor, portanto voz e dispositivo continuam confundidos.')
    p('A identificação é fechada: os locutores das classes aparecem no treino e no teste, em gravações distintas. Esta rodada não testa identificação de locutores nunca vistos nem autenticação em ambiente aberto.','small')

    title('5. VCTK: normalizações isoladas com 30 MFCCs')
    p('Rodada concluída. Acurácia média ± desvio nos cinco folds. As quatro colunas representam treino → teste no microfone indicado.','small')
    rows=[['Rede / condição','1 → 1','1 → 2','2 → 1','2 → 2']]
    for architecture in ARCH:
        for condition in CONDITIONS:
            rows.append([ARCH[architecture]+' / '+LABELS[condition]]+[accuracy(30,'vctk',architecture,condition,d) for d in DIRS])
    table(rows,[143,93,93,93,93]);figure('vctk30_cross.png')
    p('Gráfico: CNN temporal, média e desvio entre folds. CMN teve as maiores médias cruzadas nesta rodada (83,00% e 79,62%). CMVN log-mel alcançou 76,77% e 72,17%. A CNN cepstral apresentou queda forte nas intervenções; isso não sustenta uma conclusão universal de que a normalização elimina a informação de locutor.','small')

    title('6. BRSD: normalizações isoladas com 30 MFCCs')
    p('Rodada concluída: 80 locutores, 400 arquivos, Silero e janela contínua de 111 quadros. Acurácia média ± desvio nos cinco folds leave-one-text-out.','small')
    rows=[['Condição','CNN','CNN temporal']]
    for c in CONDITIONS:rows.append([LABELS[c],accuracy(30,'brsd','cnn',c,'audio->audio'),accuracy(30,'brsd','temporal_cnn',c,'audio->audio')])
    table(rows,[155,180,180]);figure('brsd30.png')
    p('Leitura do resultado','sub')
    p('Z-score teve a maior média nas duas redes: 47,00% na CNN e 58,75% na CNN temporal. As quatro intervenções reduziram a acurácia neste corpus e protocolo. A CNN com CMVN log-mel ficou em 1,25%, o nível médio do acaso para 80 classes.')
    p('A redução pode envolver perda de pistas vocais, de canal, diferenças de escala ou dificuldade de otimização. Os testes medem o efeito da intervenção; não identificam sozinhos a causa da queda. O resultado do BRSD também não contradiz automaticamente os ganhos de transferência observados no VCTK.','small')

    title('7. Rodada de 40 MFCCs e próximos resultados')
    p(f'Fotografia de {data["at"]}: status <b>{escape(data["forty_status"])}</b>; etapa <b>{escape(str(data["forty_stage"]))}</b>. Só são exibidas médias quando os cinco folds e seus testes estão completos.','small')
    rows=[['Rede / condição','1 → 1','1 → 2','2 → 1','2 → 2']]
    for a in ARCH:
        for c in CONDITIONS:
            if f'40/vctk/{a}/{c}' in data['results']:
                rows.append([ARCH[a]+' / '+LABELS[c]]+[accuracy(40,'vctk',a,c,d) for d in DIRS])
    table(rows,[143,93,93,93,93])
    p('Comparação direta disponível: z-score, 30 → 40','sub')
    rows=[['Rede / direção','30 MFCCs','40 MFCCs','Diferença da média']]
    for a in ARCH:
        for d in DIRS:
            thirty=data['results'][f'30/vctk/{a}/zscore'][d]['accuracy']['mean']
            forty_value=data['results'].get(f'40/vctk/{a}/zscore',{}).get(d)
            gain=f"{100*(forty_value['accuracy']['mean']-thirty):+.2f} pp".replace('.',',') if forty_value else 'Pendente'
            rows.append([ARCH[a]+' / '+d.replace('->',' → '),accuracy(30,'vctk',a,'zscore',d),accuracy(40,'vctk',a,'zscore',d),gain])
    table(rows,[143,132,132,108])
    p('As oito médias VCTK com z-score melhoraram ao passar para 40 coeficientes. Esta é uma descrição das médias, sem teste de significância. As demais condições e o BRSD com 40 ainda devem ser lidos na tabela atualizada quando concluírem.','small')
    p('Execução e rastreabilidade','sub')
    p('As duas CNNs e as cinco condições são executadas sequencialmente, com uma única fila de GPU. O monitor local verifica a cada 30 minutos e retoma interrupções recuperáveis pelos checkpoints. O relatório é uma fotografia; os arquivos de status e resultados continuam sendo atualizados pelo treino.','small')
    p('Fontes locais: manifestos em output/vctk16_corrected_features; tabelas output/vctk16_isolated_suite_comparison.md, output/normalization_remaining_comparison.md e output/normalization40_comparison.md; resumos summary.json e completed.json em cada diretório de resultado no HDD. Gerador: experiments/build_protocol_presentation_pdf.py.','small')
    p('Referência metodológica RASTA: Hermansky e Morgan, “RASTA processing of speech”, IEEE Transactions on Speech and Audio Processing, 1994. Implementação e parâmetros atuais registrados nos manifestos; os números apresentados vêm dos artefatos locais, não da literatura.','small')

    OUT.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=54,bottomMargin=50,
                         title='Reconhecimento de locutor - protocolo atual de 16 kHz',author='Projeto de reconhecimento de locutor')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (ROOT/'docs/apresentacao_protocolo_16k.md').write_text('\n'.join(md),encoding='utf-8')
    print(OUT)


if __name__ == '__main__':
    main()
