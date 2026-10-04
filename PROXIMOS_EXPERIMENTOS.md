# Fila atual — VCTK 16 kHz, normalizações separadas

Atualização do usuário em 28/09/2026: retirar X-vector agora; treinar somente as duas redes rápidas da comparação atual, CNN (`cnn`) e CNN temporal (`temporal_cnn`). X-vector será solicitado pelo usuário à noite; não iniciar automaticamente. Não há arquitetura chamada apenas `temporal` no runner atual.

30 MFCCs + Δ + ΔΔ, 111 quadros, mesma coorte, cinco folds e sementes, GPU obrigatória, lote 128.

## Ordem atual

1. Z-score sozinho nas duas redes: reaproveitar os baselines já concluídos e compatíveis.
2. CMN por gravação, sem z-score global, nas duas redes.
3. CMVN por gravação, sem z-score global, nas duas redes.
4. RASTA sem CMN/CMVN nem z-score global, nas duas redes.

Combinações, X-vector e 40 MFCCs adiados. Os resultados anteriores continuam preservados; suas condições CMN/RASTA incluíam z-score de treino e não equivalem aos testes isolados novos.

## Execução

Serviço: sr-vctk16-isolated-suite. Script: experiments/run_vctk16_isolated_suite.py.
Monitor a cada 30 minutos: sr-vctk-watchdog.timer; a presença de output/vctk16_isolated_request.json substitui integralmente as filas antigas no monitor.
Status e tabela: output/vctk16_isolated_suite_status.json e output/vctk16_isolated_suite_comparison.md.
Saídas isoladas próprias no HDD; checkpoints anteriores preservados. Trava compartilhada impede workers concorrentes na GPU. Retomada automática a partir do melhor checkpoint, com validação do protocolo de normalização.

## Literatura

Gusev et al., Interspeech 2020, seção 2.1, usam CMN por janela e depois normalização por média e desvio por gravação, em log-mel: https://www.isca-archive.org/interspeech_2020/gusev20_interspeech.pdf
Isso demonstra uso conjunto, mas não reproduz exatamente nosso CMN de MFCC por janela seguido de z-score estimado apenas no treino. O usuário decidiu testar condições separadas mesmo assim.

## Próximo teste autorizado após a rodada isolada

O usuário autorizou em 28/09 iniciar, depois da conclusão validada da rodada atual, CMVN por gravação e por banda log-mel ANTES da DCT. Estatísticas somente nos quadros de fala do próprio áudio, com máscara e fallback documentados. Preservar coorte, cinco folds, sementes e janela pareada de 111 quadros. Extrair 30 MFCCs e deltas na janela existente. Treinar CNN e CNN temporal em saídas novas, GPU obrigatória e lote 128, sem z-score global adicional, sem CMN extra, sem RASTA. Não iniciar X-vector nem 40 MFCCs.

A automação Codex existente `acompanhar-experimento-vctk` foi atualizada para implementar, verificar e iniciar esse teste após validar a conclusão da rodada isolada, sem reiniciar as filas antigas. Frequência mantida em 30 minutos.

## Processamento vigente confirmado nos manifestos — não usar os perfis antigos

O usuário corrigiu explicitamente o uso do perfil antigo do BRSD: ler e aplicar o ÚLTIMO processamento executado, não configs/brsd.env. Manifesto conferido: output/vctk16_corrected_features/p225/241/manifest.json e protocolo /media/lsmsqt/HDD/sr_project/vctk16_rasta_features/protocol.json.

Cadeia vigente para VCTK e nova rodada BRSD: trim suave só nas bordas por RMS (-30 dB, cinco quadros consecutivos), margens 100 ms/250 ms e fade 8 ms; decimate(q=3,n=8,ftype=iir,zero_phase=True) 48→16 kHz, com antialiasing interno Chebyshev I. SEM Butterworth separado. Pré-ênfase .97; Hamming 32 ms/16 ms; center=False; 30 MFCCs+Δ+ΔΔ; janela contínua de 111 quadros escolhida por RMS; deltas recalculados apenas nessa janela. No BRSD, Silero restringe a seleção a trechos contínuos de fala e suas máscaras/estatísticas são reutilizadas nas condições. BRSD tem cinco arquivos nativos 44,1 kHz (106–110): resample_poly diretamente para16k, com seu filtro interno, é a exceção documentada pois não existe fator inteiro de decimação. Nenhum Butterworth adicional também nesses arquivos.

BRSD: 80 locutores×5 textos; cinco folds leave-one-text-out e validação por locutor com semente42, mesma partição em todas as condições. Referência Silero+z-score; CMN, CMVN MFCC, RASTA e CMVN log-mel antes da DCT separados, sem z-score adicional. Só CNN e CNN temporal, GPU obrigatória, lote128. Um aparelho por locutor: não interpretar resultado como eliminação comprovada do confundimento de microfone.

## Execução concreta do restante

Script único experiments/run_normalization_remaining.py, serviço sr-normalization-remaining, status output/normalization_remaining_status.json, tabela output/normalization_remaining_comparison.md. O watchdog usa output/normalization_remaining_request.json e acompanha somente essa fila. Primeiro consultar status/serviço; NÃO iniciar filas duplicadas pela automação.

Ordem: extração CMVN log-mel VCTK, CNN e CNN temporal; extração BRSD Silero pelo processamento vigente; CNN e CNN temporal para zscore/CMN/CMVN/RASTA/CMVN-logmel. Preservar resultados anteriores e checkpoints. X-vector e combinações ficam adiados; não iniciar 40 MFCCs nem attention.

Testes preliminares com o perfil antigo BRSD8k/40/Hann em /tmp NÃO devem ser usados. A rodada completa desse perfil NÃO foi iniciada. Saídas corrigidas têm protocolo versionado próprio. Qualquer mudança de protocolo precisa ser explícita no manifesto e na documentação.

Correção de leitura em 28/09: arquivos BRSD multicanais são convertidos em mono pela média dos canais, como no carregamento librosa anterior. Áudios já mono permanecem intactos; filtro, recorte, decimação e features não mudam. Quantidade de canais originais registrada nos novos audits.


## Rodada de 40 MFCCs autorizada em 28/09 à noite

O usuário pediu: "roda os 40 ... todos menos o xvec". A rodada de 30 está concluída. Executar agora VCTK e BRSD com 40 MFCCs+Δ+ΔΔ (120 características por quadro), somente CNN e CNN temporal. Condições separadas: z-score, CMN, CMVN MFCC, RASTA, CMVN log-mel antes da DCT; nenhum z-score adicional nas quatro intervenções. Combinações e X-vector permanecem adiados.

Mesmo processamento vigente, janelas de 111 quadros, folds, sementes e lote128; GPU obrigatória. BRSD reaproveita as máscaras/segmentos Silero dos audits de 30 e confere igualdade dos recortes e janelas. Diretórios novos e exclusivos com 40: vctk40_isolated_features, brsd40_silero_isolated_features e isolated_*vctk40/brsd40* no HDD. Treinos anteriores preservados.

Script: experiments/run_normalization40.py; serviço sr-normalization40. Status/tabela: output/normalization40_status.json e output/normalization40_comparison.md. O watchdog dá prioridade a output/normalization40_request.json e verifica só esta fila, a cada30 minutos; não retomar as antigas filas combinadas de40.

Verificação de amostras: VCTK baseline/RASTA/CMVN-logmel e BRSD cinco condições mantêm os primeiros30 coeficientes, janelas e máscaras da referência; os10 coeficientes adicionais são reextraídos do mesmo áudio.
