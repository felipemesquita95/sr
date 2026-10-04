# Prompt de contexto — histórico dos experimentos de reconhecimento de locutor

Use este contexto para auditar, discutir e continuar o projeto em `/home/lsmsqt/Documents/sr`. Descreva o que realmente foi executado. Não confunda uma fila planejada, checkpoint parcial, relatório antigo e resultado completo. Este prompt registra o histórico; não autoriza apagar dados, alterar o protocolo ou reiniciar serviços.

## Fontes e regra de leitura

Áudios originais em `/media/lsmsqt/HDD/datasets/`; derivados em `/media/lsmsqt/HDD/sr_project/`, `runs/` e `output/`. Consultar os manifestos, divisões, métricas, `completed.json`, `summary.json` e progresso real antes de afirmar conclusão. As tabelas nos anexos são cópias dos relatórios locais no horário deste snapshot; declarações antigas de “em andamento” podem estar desatualizadas. Resultados históricos documentados não foram todos recalculados nesta auditoria.

## 1. Histórico anterior às reexecuções corrigidas

O documento `docs/resultados.md` registra: matriz de transferência com CNN, CNN temporal, CNN temporal com statistics pooling, CNN temporal wide, attention cepstral e attention temporal; referência linear estática de média/desvio dos MFCCs; cross-microfone em ambas as direções, texto disjunto e sem VAD; diagnóstico fala/não fala por classificador linear; análise de erros por gênero/sotaque e locutor; transformação afim entre capturas calibrada em 36 locutores e avaliada em 72; permutação de rótulos e verificação de alinhamento; e BRSD com CNN, CNN temporal e attention. Os detalhes e números registrados estão no anexo desse documento.

São protocolos históricos próprios, diferentes das rodadas de 16 kHz abaixo. Não transferir seus números, divisões, comprimento de 300 quadros ou número de sementes para os experimentos novos. O texto antigo usa expressões como “silêncio puro” e conclusões causais fortes: considerar esses trechos como alegações do relatório histórico, não como comprovação atual. Ausência de detecção de fala não garante ausência física de fala. Houve correções anteriores de numeração/rótulos e da validação VCTK; resultados dos protocolos substituídos devem ser identificados como antigos.

## 2. Reexecução de 8 kHz

VCTK: 108 locutores, 21.523 pares selecionados. BRSD: 80 locutores, 400 gravações. Áudio mono, sem VAD na extração base; Butterworth ordem 8/fase zero/corte 3,6 kHz na taxa nativa, resample_poly para 8 kHz, pré-ênfase 0,97, 40 MFCCs, janela 32 ms/256 amostras, hop 16 ms/128. O perfil histórico usa Hann, diferentemente da rodada posterior Hamming a 16 kHz.

CNN, CNN temporal e attention; cinco partições; normalização somente no treino. BRSD: 3/1/1 gravações por locutor, totais 240/80/80, mínimo 1007 quadros de entrada. VCTK: primeiros 77 quadros. A validação VCTK anterior com somente uma gravação por locutor foi arquivada e substituída por cinco grupos intercalados e divisão 60/20/20. Os resultados corrigidos estão no anexo `docs/relatorio_8k.md`.

Foi executada também a comparação pareada 77 versus 153 quadros: mesmas 18.067 gravações elegíveis, mesmos papéis nos cinco folds e redes retreinadas em cada duração. Foram feitas avaliações cross-microfone dos checkpoints; consultar os diretórios `runs/models/vctk8k_cross_pareado` e `vctk153_cross_pareado`. Os 77 quadros do corpus completo não equivalem à coorte reduzida 77 pareada com 153.

## 3. Atividade e baixa atividade por energia a 8 kHz

Não eram seleções por Silero. Detector de energia com top_db=30, análise no áudio completo após filtragem/reamostragem, margem de um quadro nas transições. Quadros selecionados podem ser descontínuos e foram concatenados em ordem temporal.

- 17.272 pares/108 locutores: 10 quadros sem seleção, 10 de atividade, 10 de baixa atividade e misto 5+5. CNN, CNN temporal e attention, ambos microfones, cinco partições e cross nas duas direções.
- 17.270 pares: 20 de atividade versus 10 atividade+10 baixa, mesmas três redes e avaliações.
- Controle linear: todos os quadros sem seleção, todos de atividade, todos de baixa, 20 atividade e 10+10; cada gravação resumida por média/desvio dos 40 MFCCs, 80 números, classificador ridge linear.
- Coorte balanceada de 100 locutores × 25 gravações: 15/5/5 por fold; 20 atividade versus 20 baixa, e 40 sem seleção versus 40 atividade versus 20+20; três redes e cross-mic.

Os anexos incluem as tabelas completas. Não chamar baixa energia de silêncio comprovado nem misturar essa coorte de 100 locutores com a de 108.

## 4. Nova extração VCTK a 16 kHz e primeiras redes

Coorte de 108 locutores × 120 leituras pareadas = 12.960 pares. Seleção das leituras mais longas após trim por locutor, com desempate documentado. Cinco grupos de 24 leituras por pessoa; por fold, 7.776 treino/2.592 validação/2.592 teste, mesmos papéis para mic1 e mic2, semente 42.

Trim somente nas bordas por RMS relativo −30 dB, cinco quadros consecutivos, margens 100 ms/250 ms, fade 8 ms; união da atividade dos canais, pausas internas preservadas. 48→16 kHz com decimate q=3, ordem 8 IIR Chebyshev I/fase zero e antialias interno, sem Butterworth separado. Pré-ênfase 0,97; Hamming 32 ms/512, hop 16 ms/256, center=False; 128 bandas mel, DCT-II ortonormal, 30 MFCCs.

Janela contínua pareada de 111 quadros, escolhida pela soma de RMS normalizado dos dois microfones, sem restrição Silero. Deltas e delta-deltas recalculados dentro da janela, largura nove. 111 quadros cobrem 1,792 s. X-vector foi comparado com MFCC estático e MFCC+Δ+ΔΔ; houve seleção por validação do primeiro fold e posteriormente seleção aninhada por fold. Também foram avaliadas CNN cepstral e CNN temporal. Primeiros resultados estão em `output/vctk16_all_models_results.md`; não são os mesmos treinos GPU/lote128 da ablação posterior.

## 5. Piloto Silero VCTK — protocolo separado

Existe um piloto real com Silero: 108 locutores × 105 pares, 70 quadros, x-vector dinâmico, somente fold1 concluído para origem mic1. Controle sem VAD versus Silero. No piloto, trechos de fala foram concatenados antes da extração, independentemente por microfone: não é a janela contínua pareada de 111 quadros.

O relatório preliminar registra 96,38%/53,44% para controle mic1→mic1/mic2 e 96,47%/54,67% com Silero. Nesta auditoria há `completed.json` somente de fold1/mic1 em cada condição. Não apresentar como cinco folds concluídos, nem como estudo completo de atividade versus não atividade por Silero. O checkpoint de controle mic2 foi interrompido antes da conclusão; o estado “Silero mic2 em andamento” é uma declaração antiga do relatório, não prova de processo ativo hoje.

## 6. Rodada VCTK30 baseline, CMN e RASTA COM z-score

GPU/lote128, cinco folds, x-vector, CNN e CNN temporal. Três condições: baseline; CMN MFCC por janela seguido de z-score do treino de origem; RASTA log-mel antes da DCT seguido de z-score do treino de origem. RASTA não combinado com CMN nessa rodada. Mesma coorte/janela RMS de 111 quadros.

Há piloto CMN e uma execução anterior `cmn_vctk16_results`, além da rodada posterior GPU/lote128; não misturar seus checkpoints. A tabela final `output/vctk16_channel_suite_comparison.md` registra todas as nove condições concluídas. X-vector baseline chega a 98,84% intra mic1, mas isso não é desempenho cross-mic.

## 7. Combinação CMN+RASTA e fila antiga de 40

CMN+RASTA com x-vector30 chegou a iniciar: há checkpoint e progresso em `cmn_rasta_vctk16_gpu_b128/dynamic/fold1/mic1`, sem `completed.json` ou summary nessa pasta na inspeção. Foi adiado pelo usuário. Não afirmar que a combinação nunca rodou, nem que terminou.

A fila `vctk16_followup_suite` pretendia CMN+RASTA e CMVN nas três redes com z-score adicional. A fila antiga `vctk40_suite` pretendia 40 com três redes e combinações; foi adiada com etapas pendentes. Ambas estão `deferred_by_user`; seus status internos antigos não demonstram processo ativo. Não confundir essa fila40 com `normalization40` atualmente executada.

## 8. Normalizações ISOLADAS VCTK30

Rodada concluída somente CNN e CNN temporal. Z-score de referência, CMN, CMVN MFCC, RASTA e CMVN log-mel. Nas quatro intervenções não há z-score global adicional; baseline z-score foi reaproveitado da rodada compatível. CMN/CMVN MFCC usam estatísticas da própria janela. CMVN escala derivadas pelo desvio dos estáticos, equivalente a recalculá-las após a transformação afim.

RASTA no log-mel completo do recorte antes da DCT: b=[.2,.1,0,-.1,-.2], a=[1,-.94], inicialização estacionária pelo primeiro quadro, sem compensação de atraso. CMVN log-mel usa média/desvio por banda nos quadros Silero de fala da gravação recortada, antes da DCT; piso 1e-8; fallback para todos os quadros quando não há fala é registrado. Mesmo nessa condição a janela VCTK foi mantida por RMS.

CMN foi melhor no cross-microfone com CNN temporal30: 83,00% e 79,62%; z-score foi melhor intra. CMN não foi o melhor na CNN cepstral nem no BRSD. Todas as tabelas estão anexadas.

## 9. BRSD16k/Silero30 e cinco normalizações separadas

80 locutores × cinco textos, 400 WAVs; mono por média dos canais. Mesmo processamento 16 kHz vigente, com trim RMS nas bordas. Arquivos 106–110 a 44,1 kHz usam resample_poly direto para 16 kHz; demais48k usam decimate, nenhum Butterworth separado.

Silero6.2.3, threshold0,5, fala mínima250ms, silêncio mínimo100ms, margem30ms, máscara por centro dos quadros. A janela contínua de111 quadros é escolhida por RMS dentro de uma região Silero de fala. Falta de trecho suficiente interrompe a extração; não há concatenação para completá-lo. Mesmas máscaras, janelas e folds nas cinco condições. Não foi executada condição de não atividade separada nessa rodada.

CNN e CNN temporal, GPU/lote128, cinco folds leave-one-text-out, validação por locutor semente42, totais240/80/80; teto1000épocas/patience30. Z-score só na referência; CMN, CMVN, RASTA e CMVNlog-mel separados. Rodada concluída, tabela `normalization_remaining_comparison.md`. Referência temporal58,75%; CMN29,75%; não concluir superioridade de CMN nessa base. O BRSD tem um aparelho por locutor, sem captação pareada cross-dispositivo.

## 10. Rodada atual40, sem X-vector

Autorização: “roda os40 ... todos menos o xvec”. CNN e CNN temporal em VCTK e BRSD; cinco normalizações separadas como acima. 40MFCC+Δ+ΔΔ =120 características por quadro; mesmas janelas111, coorte, folds, sementes e parâmetros dos30, arquivos próprios. VCTK mantém seleção RMS; BRSD deve reaproveitar segmentos e máscaras Silero dos30 e validar igualdade das janelas. GPU obrigatória/lote128; VCTK teto150épocas/patience15.

Orquestrador `experiments/run_normalization40.py`, serviço `sr-normalization40.service`, status `output/normalization40_status.json`; watchdog `sr-vctk-watchdog.timer`, a cada30min, prioridade da solicitação `normalization40_request.json`, trava compartilhada da GPU. Estado exato das etapas neste snapshot está abaixo. A tabela só inclui etapas completas; progresso ativo exige ler arquivos por fold, pois o status global só muda nas transições.

## 11. Divergência de protocolo identificada pelo usuário

O usuário esclareceu que Silero deveria escolher amostras de atividade E não atividade. A rodada VCTK de normalizações30/40 descrita acima não implementou isso: escolheu por RMS. O piloto Silero e o CMVN log-mel não tornam essas rodadas equivalentes ao pedido. BRSD selecionou fala com Silero, mas não comparou uma condição separada de não atividade. Documentar essa diferença explicitamente; não mudar o rótulo dos resultados existentes.

Uma nova varredura foi solicitada:100 arquivos aleatórios BRSD e100 pares aleatórios VCTK, Silero independente em mic1/mic2, waveform original e classificação +1atividade/−1nãoatividade alinhadas no tempo, visualização alternando a cada2s. Script `experiments/audit_silero_random100.py`; saída `auditoria_silero_100_20260928` no HDD. A varredura analisa áudio completo a16k, sem novo trim RMS ou pré-ênfase; não é treino nem extração MFCC. Consultar seu progresso real, pois foi iniciada enquanto este contexto era preparado.

## 12. Intercorrências e preservação

Houve falta de espaço no disco do sistema; modelo local Qwen do Ramesses foi removido com autorização do usuário. Houve reboot e falha NTFS/MFTMirr no HDD; reparo Linux autorizado e remontagem permitiram retomada. Pasta corrompida de RASTA fold1/mic2 foi preservada com sufixo de corrupção e etapa refeita. Não interpretar os diretórios incompletos antigos como resultados válidos.

O usuário cogitou apagar derivados depois, mas o escopo e momento de “apagar tudo” não foram definidos. Este prompt não é instrução de exclusão. Áudios originais e código devem ser preservados. O PDF atualizado `output/pdf/relatorio_experimental_reconhecimento_locutor.pdf` descreve o protocolo realizado e contém snapshot parcial40; PDF anterior preservado em `output/pdf/arquivo_protocolo_anterior/`.

## Estado das filas no snapshot

Horário: 2026-09-28T22:09:34-03:00

### vctk16_channel_suite_status.json

Global: `complete`; etapa atual: `None`; atualização registrada: `2026-09-28T18:40:48.008926+00:00`.

| Etapa | Estado registrado |
|---|---|
| cnn/cmn | complete |
| cnn/baseline | complete |
| temporal_cnn/cmn | complete |
| temporal_cnn/baseline | complete |
| prepare/RASTA | complete |
| xvector/rasta | complete |
| cnn/rasta | complete |
| temporal_cnn/rasta | complete |

### vctk16_followup_suite_status.json

Global: `deferred_by_user`; etapa atual: `xvector/cmn_rasta`; atualização registrada: `2026-09-28T18:40:56.473674+00:00`.

| Etapa | Estado registrado |
|---|---|
| xvector/cmn_rasta | running |
| cnn/cmn_rasta | pending |
| temporal_cnn/cmn_rasta | pending |
| xvector/cmvn | pending |
| cnn/cmvn | pending |
| temporal_cnn/cmvn | pending |

### vctk40_suite_status.json

Global: `deferred_by_user`; etapa atual: `None`; atualização registrada: `2026-09-28T16:29:55.945731+00:00`.

| Etapa | Estado registrado |
|---|---|
| prepare/MFCC40 | pending |
| prepare/RASTA40 | pending |
| xvector/baseline | pending |
| cnn/baseline | pending |
| temporal_cnn/baseline | pending |
| xvector/cmn | pending |
| cnn/cmn | pending |
| temporal_cnn/cmn | pending |
| xvector/rasta | pending |
| cnn/rasta | pending |
| temporal_cnn/rasta | pending |
| xvector/cmn_rasta | pending |
| cnn/cmn_rasta | pending |
| temporal_cnn/cmn_rasta | pending |
| xvector/cmvn | pending |
| cnn/cmvn | pending |
| temporal_cnn/cmvn | pending |

### vctk16_isolated_suite_status.json

Global: `complete`; etapa atual: `None`; atualização registrada: `2026-09-28T19:41:26.977725+00:00`.

| Etapa | Estado registrado |
|---|---|
| cnn/zscore | complete |
| temporal_cnn/zscore | complete |
| cnn/cmn | complete |
| temporal_cnn/cmn | complete |
| cnn/cmvn | complete |
| temporal_cnn/cmvn | complete |
| cnn/rasta | complete |
| temporal_cnn/rasta | complete |

### normalization_remaining_status.json

Global: `complete`; etapa atual: `None`; atualização registrada: `2026-09-28T22:22:43.745142+00:00`.

| Etapa | Estado registrado |
|---|---|
| vctk/prepare_logmel_cmvn | complete |
| vctk/cnn/cmvn_logmel | complete |
| vctk/temporal_cnn/cmvn_logmel | complete |
| brsd/prepare_silero | complete |
| brsd/cnn/zscore | complete |
| brsd/temporal_cnn/zscore | complete |
| brsd/cnn/cmn | complete |
| brsd/temporal_cnn/cmn | complete |
| brsd/cnn/cmvn | complete |
| brsd/temporal_cnn/cmvn | complete |
| brsd/cnn/rasta | complete |
| brsd/temporal_cnn/rasta | complete |
| brsd/cnn/cmvn_logmel | complete |
| brsd/temporal_cnn/cmvn_logmel | complete |

### normalization40_status.json

Global: `running`; etapa atual: `vctk/cnn/rasta`; atualização registrada: `2026-09-29T01:05:35.096103+00:00`.

| Etapa | Estado registrado |
|---|---|
| vctk/prepare_baseline | complete |
| vctk/prepare_rasta | complete |
| vctk/prepare_cmvn_logmel | complete |
| vctk/cnn/zscore | complete |
| vctk/temporal_cnn/zscore | complete |
| vctk/cnn/cmn | complete |
| vctk/temporal_cnn/cmn | complete |
| vctk/cnn/cmvn | complete |
| vctk/temporal_cnn/cmvn | complete |
| vctk/cnn/rasta | running |
| vctk/temporal_cnn/rasta | pending |
| vctk/cnn/cmvn_logmel | pending |
| vctk/temporal_cnn/cmvn_logmel | pending |
| brsd/prepare_silero40 | pending |
| brsd/cnn/zscore | pending |
| brsd/temporal_cnn/zscore | pending |
| brsd/cnn/cmn | pending |
| brsd/temporal_cnn/cmn | pending |
| brsd/cnn/cmvn | pending |
| brsd/temporal_cnn/cmvn | pending |
| brsd/cnn/rasta | pending |
| brsd/temporal_cnn/rasta | pending |
| brsd/cnn/cmvn_logmel | pending |
| brsd/temporal_cnn/cmvn_logmel | pending |

## Anexos — tabelas e detalhes registrados nos relatórios locais

Os anexos preservam os registros, inclusive comentários antigos sobre andamento. Use a síntese e o snapshot acima para distinguir estado atual de histórico. As interpretações fortes de documentos antigos precisam ser auditadas, especialmente os rótulos de silêncio e inferências sobre voz/sessão.

---

### Fonte: `docs/resultados.md`

# Resultados

Todos os números vêm dos artefatos em `runs/models/`, e não desta documentação. O
nível do acaso é 1/*N*, onde *N* é o número de locutores: **0,93% no VCTK** (108
locutores) e **1,25% no BrSD** (80 locutores).

> **Procedência.** Os resultados anteriores a setembro de 2026 foram medidos sob a
> numeração deslocada descrita em [`limitacoes.md`](limitacoes.md), contra um teto de
> acurácia de 48,2% em vez de 100%. Eles não constam mais deste documento. O que está
> aqui foi medido na implementação atual, com 108 locutores, e cada seção indica o
> diretório de onde os números saem.

---

## 1. O resultado principal

| condição | acurácia | o que mede |
|---|---|---|
| Mesmo microfone | 98,4 – 99,4% | voz **e** sessão, indistinguíveis |
| Só silêncio, mesmo microfone | 85,4% | a sessão isolada, sem fala alguma |
| Trocando o microfone | 37,0 – 59,6% | o que resta quando a captação muda |
| Rótulos embaralhados | 0,70% | controle do arcabouço |
| Acaso | 0,93% | — |

A leitura conjunta é o resultado do trabalho: **entre 98% e algo na faixa de 37% a
60% está a diferença entre o que um protocolo convencional reporta e o que sobrevive
quando o canal de gravação deixa de ser constante.** Nenhuma das medidas é errada;
elas respondem a perguntas diferentes.

O que **não** se pode afirmar é que a faixa inferior seja identidade vocal pura. A
seção 6 mostra que não é.

---

## 2. Matriz de transferência

`runs/models/matriz9_*` — nove arquiteturas, das quais seis medidas até aqui.

É a medida mais controlada do trabalho. A divisão por enunciado é **fixa e idêntica
nas duas trilhas**, o mesmo checkpoint é avaliado nas duas capturas, a validação e a
normalização ficam sempre na trilha de origem, e cada célula é a média de três
sementes de treino. O microfone é a única variável que muda.

Isso corrige um defeito das comparações anteriores: contrastar 98% com 38% misturava
a troca de trilha com tamanhos de treino diferentes, conjuntos de teste diferentes e
regras de validação diferentes.

Acurácia em %, média ± desvio entre sementes. Lote 256, paciência de parada 10.

| arquitetura | parâmetros | intra `mic1` | → `mic2` | → `mic1` | intra `mic2` |
|---|---:|---:|---:|---:|---:|
| `temporal_cnn` | 114.732 | 98,38 | 40,84 | 50,43 | 96,62 |
| `temporal_cnn_stats` | 147.500 | 98,39 | **44,73** | 52,41 | 96,54 |
| `cnn` | 369.548 | 98,37 | 39,38 | 47,84 | 95,81 |
| **`temporal_attention`** | 402.156 | **99,43** | 43,12 | **59,57** | **98,94** |
| `temporal_cnn_wide` | 866.412 | 99,00 | 41,92 | 52,09 | 98,07 |
| `attention` | 1.647.340 | 98,54 | 37,00 | 47,59 | 96,00 |

Perda pareada, em pontos percentuais — a queda do mesmo checkpoint ao ser avaliado na
outra captura das **mesmas** gravações:

| arquitetura | treino em `mic1` | treino em `mic2` |
|---|---:|---:|
| `temporal_attention` | 56,31 ± 0,77 | **39,37 ± 1,29** |
| `temporal_cnn_stats` | **53,66 ± 3,50** | 44,13 ± 1,82 |
| `temporal_cnn_wide` | 57,08 ± 1,24 | 45,99 ± 1,33 |
| `temporal_cnn` | 57,55 ± 0,26 | 46,19 ± 2,56 |
| `cnn` | 58,99 ± 0,54 | 47,98 ± 0,61 |
| `attention` | 61,54 ± 0,55 | 48,42 ± 1,36 |

### 2.1 A direção do cruzamento importa, e não é reportada

Treinar no `mic2` e avaliar no `mic1` perde de 39 a 48 pontos; a direção oposta perde
de 54 a 62. **A diferença chega a 17 pontos na mesma arquitetura.**

O `mic1` é um DPA 4035 omnidirecional e o `mic2` é um Sennheiser MKH 800, de banda
muito mais larga. Aprender sobre o sinal mais rico e aplicar ao mais pobre não é o
mesmo problema que o inverso, e a literatura que usa este protocolo não costuma
declarar em que sentido rodou.

Consequência: **o piso não é um número, é um intervalo**, e a direção convencional é
a pessimista.

### 2.2 O eixo da convolução decide; a capacidade quase não

As quatro arquiteturas do eixo temporal ocupam as quatro primeiras posições no
cruzamento. As duas do eixo cepstral — `cnn` e `attention` — são as duas últimas, e a
`attention` é a pior apesar de ser a maior de todas as medidas.

O controle de capacidade está em `temporal_cnn_wide`: é a `temporal_cnn` com quatro
vezes mais filtros e nada mais alterado. **Sete vezes e meia mais parâmetros compram
1,1 e 1,7 pontos.** Não é capacidade que separa as arquiteturas.

A justificativa é a que o projeto declara desde o início: o eixo cepstral **não possui
estrutura de vizinhança** — coeficientes adjacentes são projeções de bases distintas
da DCT. Convolução pressupõe localidade e é mal-condicionada ali.

### 2.3 A agregação vale mais que a capacidade

`temporal_cnn_stats` difere de `temporal_cnn` por **uma única camada**: agrega média
**e** desvio ao longo do tempo, em vez de só a média. O ganho é de 3,9 pontos numa
direção e 2,0 na outra, com 33 mil parâmetros a mais.

Para comparação, quadruplicar os filtros rende 1,1 ponto. **A dispersão temporal que a
média descarta carrega mais informação transferível do que sete vezes mais
capacidade.**

### 2.4 Atenção sobre o tempo é a melhor configuração medida

`temporal_attention` tem a maior acurácia dentro de cada microfone (99,43% e 98,94%) e
o melhor cruzamento em uma direção, por margem larga: **59,57%**, sete pontos acima da
segunda colocada. Com 402 mil parâmetros.

Ela existe para fechar a grade de ablação, que antes tinha convolução nos dois eixos e
atenção apenas no cepstral:

| | eixo cepstral | eixo temporal |
|---|---|---|
| convolução | `cnn` | `temporal_cnn` |
| atenção | `attention` | `temporal_attention` |

O resultado precisa ser lido com a ressalva que a própria arquitetura carrega: se o
modelo pode escolher **onde** olhar, e a identidade se prediz melhor a partir dos
trechos sem fala, atenção temporal é o mecanismo mais capaz de explorar o confundidor
— não uma defesa contra ele.

### 2.5 Uma referência estática linear supera a CNN nesta divisão

`runs/models/vctk_static_reference` — regressão logística sobre os mesmos
tensores de 40 MFCCs e 300 quadros de `vctk_transfer_matrix`. Cada gravação vira
80 números: média e desvio no tempo dos tensores já normalizados pela origem. A
divisão por enunciado, a repetição/truncamento e a normalização são os da matriz;
o instrumento confere as estatísticas reconstruídas contra o artefato da CNN.

A grade `C = (0,1; 1; 10)` foi fixada antes da execução e a validação da origem
selecionou `C = 10` nas duas direções. Acurácia em %, sobre as mesmas 4.305
gravações de teste por captura:

| treino | teste na origem | teste na outra captura | perda pareada |
|---|---:|---:|---:|
| `mic1` | 99,56 | 52,87 | 46,69 pp |
| `mic2` | 99,12 | 58,44 | 40,67 pp |

Na mesma divisão, as três sementes da CNN ficaram em 33,84–37,54% no sentido
`mic1 → mic2` e em 41,51–42,93% no inverso. Esta referência não usa uma semente
CNN escolhida depois: o artefato conserva as três. O resultado refuta, nesta
representação e divisão, tratar o patamar cruzado da CNN como teto do protocolo.
Ele não identifica qual coeficiente, banda ou aspecto da sessão causa a vantagem,
nem demonstra que a ordem temporal seja irrelevante para outros modelos ou dados.

---

## 3. Diagnóstico de canal

`runs/models/vctk_cross_mic/diagnostico_travessia` — regressão logística sobre média e
desvio dos 40 coeficientes cepstrais, 80 números por gravação.

| condição | dentro da trilha | atravessando o microfone |
|---|---:|---:|
| **só silêncio** | **85,40%** | 4,33% |
| só fala | 99,64% | 55,38% |
| sinal completo | 99,81% | 51,44% |

**A identidade é predita com 85,4% de acurácia a partir de trechos sem fala alguma**,
por um classificador linear, contra 0,93% de acaso. A maior parte da separabilidade
entre locutores está disponível fora da voz.

Vale notar que 80 números superam uma rede de 370 mil parâmetros dentro da mesma
trilha. Não há nada de profundo a aprender nessa condição — é uma assinatura
linearmente legível.

O corpus é gravado em câmara hemi-anecoica, com o mesmo equipamento para todos. O
"silêncio" aqui não é ruído de sala: é o piso de ruído da cadeia de captação, a
respiração e o corpo da pessoa numa posição fixa. Além disso, a variante usada do VCTK
é `wav48_silence_trimmed`, com silêncio de início e fim já removido pelos autores —
o que resta são **pausas internas**.

---

## 4. Cross-microfone, protocolos auxiliares

`runs/models/vctk_cross_mic*` — partição única, sem estimativa de desvio.

| protocolo | `cnn` | `temporal_cnn` | `attention` |
|---|---:|---:|---:|
| `mic1` → `mic2` | 38,12 | 36,65 | 37,54 |
| `mic2` → `mic1` | 43,31 | 49,43 | 47,97 |
| Texto disjunto | 38,69 | 35,47 | 33,66 |
| Sem detecção de voz | 38,20 | 18,50 | 30,29 |

**Texto disjunto.** Cada locutor do VCTK lê um conjunto **diferente** de sentenças de
jornal; só a *rainbow passage* e o parágrafo de elicitação são comuns. O texto fica
correlacionado ao rótulo, e no protocolo convencional a mesma frase está no treino
(uma trilha) e no teste (a outra). Cortando os enunciados ao meio, a `cnn` não se
move — 38,69% contra 38,12% — e faz isso com metade dos dados de treino. O atalho
lexical não estava sendo explorado. *Ressalva:* o corte é por índice e não lê
transcrição.

**Sem detecção de voz.** A `cnn` não se altera; a `temporal_cnn` cai pela metade. Ela
agrega por média, e o silêncio devolvido entra nessa média. *Ressalva:* o truncamento
em 300 quadros corta mais nesta variante, cujo percentil 95 é 406 contra 299 — a
causa não está isolada.

---

## 5. Estrutura dos erros

`runs/models/vctk_transfer_matrix/estrutura_erros` — sobre as predições já
persistidas, sem treino nem inferência.

### 5.1 Os erros respeitam o gênero, não o sotaque

Fração dos erros que cai em alguém do mesmo grupo do alvo, contra um acaso que
preserva **tanto** a distribuição dos grupos alvo **quanto** a frequência das classes
preditas — sem essa correção, um sistema que apenas prefira prever classes de grupos
grandes já pareceria organizado por grupo.

| direção | campo | observado | esperado | excesso |
|---|---|---:|---:|---:|
| `mic1` → `mic2` | gênero | 94,4% | 52,0% | **+42,3** |
| | sotaque | 23,4% | 18,1% | +5,3 |
| `mic2` → `mic1` | gênero | 86,6% | 50,8% | **+35,9** |
| | sotaque | 22,9% | 17,6% | +5,3 |

O que sobrevive à troca de captação é fortemente organizado por gênero e quase nada
por sotaque. **A medida descarta que os erros ignorem o grupo; não estabelece decisão
por grupo** — errar entre pessoas do mesmo gênero também é o que faz um identificador
que confunde vozes parecidas, e vozes parecidas tendem a ser do mesmo gênero.

### 5.2 A média não descreve nenhum locutor

Revocação por locutor na célula cruzada:

| treino em | média | mediana | quartis | nunca acertado | abaixo de 10% | acima de 90% |
|---|---:|---:|---:|---:|---:|---:|
| `mic1` | 35,1% | 28,7% | 6,7% / 61,0% | 4 | 32 | 5 |
| `mic2` | 42,1% | 40,0% | 15,6% / 69,4% | 6 | 21 | 5 |

Quatro locutores não são acertados **uma única vez**, e cinco passam de 90%. A
distribuição é bimodal: há pessoas cuja identidade atravessa a troca de captação quase
intacta e pessoas cuja identidade era inteiramente do canal.

Dizer que "o sistema reconhece 35% das vezes" não descreve nenhuma das duas
populações. É a tese do trabalho — não confiar em acurácia agregada — aplicada ao
próprio resultado do trabalho.

---

## 6. O protocolo cross-microfone não remove o confundidor de sessão

`runs/models/sessao_ou_transdutor` — **a revisão mais importante deste documento.**

A seção 3 mostra que a pista do silêncio cai de 85,4% para 4,3% ao trocar de
microfone, e disso foi concluído que ela era do transdutor e não da sessão. **A
inferência não se sustenta**, por duas razões que a estrutura do corpus impõe:

1. Todos os locutores foram gravados com o **mesmo modelo** de microfone. Um modelo de
   microfone é idêntico para todos e não pode distinguir locutores — se a pista fosse
   o equipamento enquanto tal, a acurácia seria a do acaso.
2. As duas trilhas gravam a mesma sessão **simultaneamente**. Tudo o que é de sessão
   está nas duas. Se a pista não transfere, isso não pode significar que ela não
   exista do outro lado.

O que varia por pessoa é a **realização da sessão**: o ganho ajustado para aquele
locutor, a distância e a postura, o corpo, a respiração, as condições daquele dia. E
as duas cadeias de captação têm ganhos independentes, de modo que a mesma informação
aparece em coordenadas diferentes em cada trilha.

O teste: aprender uma transformação afim entre os espaços das duas trilhas em **36
locutores de calibração**, sem usar rótulo de identidade, e aplicá-la aos 72 restantes,
que a transformação nunca viu. O acaso passa a ser 1/72 = 1,39%.

| condição | mesma trilha | outra trilha | outra, **transformada** | recupera |
|---|---:|---:|---:|---:|
| **só silêncio** | 81,6% | 6,0% | **29,7%** | **31,4%** |
| só fala *(controle)* | 99,5% | 55,9% | 69,2% | 30,5% |

**A informação estava lá.** Uma transformação afim simples devolve quase um terço da
queda: o silêncio sai de 4,3× o acaso para 21× o acaso. A condição `speech` recupera a
mesma fração, o que confirma que o instrumento funciona e que o efeito não é artefato
do silêncio.

Três consequências:

1. **O cross-microfone não elimina o confundidor de sessão — ele o torna ilegível**
   para um classificador treinado nas coordenadas de uma trilha.
2. **Os números da seção 2 não são identidade vocal pura.** Carregam informação de
   sessão que sobrevive à troca de captação em forma transformável.
3. **Os 31,4% são um piso.** A transformação é afim, com regularização fixada antes da
   avaliação. Uma transformação não-linear recuperaria mais.

Este resultado não vale apenas para este trabalho. Quem usa o protocolo
cross-microfone como controle de canal está supondo que trocar o transdutor elimina o
confundidor, e a medida acima diz que não elimina.

---

## 7. Controles

| controle | resultado | o que estabelece |
|---|---|---|
| Permutação de rótulos, intra-mic | 0,70% ± 0,02 | sem vazamento no arcabouço |
| Permutação de rótulos, cross-mic | 0,81% | idem, no outro protocolo |
| Alinhamento entre trilhas | 100/100 recuperados | as trilhas são o mesmo evento acústico |

**Permutação.** A perda estabiliza em 4,68 = ln(108). *Ressalva:* cada gravação recebe
um rótulo embaralhado, incluindo as duas versões de um mesmo enunciado, o que destrói
também a dependência que um vazamento por gravações correlacionadas exploraria. É
evidência favorável, não prova de ausência de qualquer forma de vazamento.

**Alinhamento.** `runs/models/verificacao_alinhamento` — a gravação correspondente é
recuperada entre 20 distratores por correlação, com acaso conhecido de 4,8%.
Correlação média de 0,878 nos pares contra 0,130 nos distratores, sem sobreposição. O
mesmo instrumento registra que o VAD, aplicado independentemente a cada trilha, desfaz
a correspondência quadro a quadro: das 6.200 gravações comparadas, apenas 212 mantêm
duração idêntica, com divergência mediana de 11,5%.

---

## 8. BrSD

`runs/models/brsd` — 80 locutores, acaso 1,25%, cinco partições.

| arquitetura | acurácia |
|---|---:|
| `temporal_cnn` | **83,00% ± 3,22** |
| `cnn` | 73,75% ± 4,81 |
| `attention` | 61,25% ± 5,65 |

O BrSD tem um confundidor **diferente** do VCTK: cada locutor gravou com o próprio
dispositivo, de modo que o modelo do aparelho é constante por classe e distinto entre
classes. É o caso em que a assinatura do equipamento realmente pode identificar.

Não há aqui protocolo cross-dispositivo, porque não existe segunda captação da mesma
gravação. O corpus permanece subutilizado nas análises recentes.

---

## 9. O que este documento **não** afirma

- Que a faixa de 37% a 60% seja identidade vocal. A seção 6 mostra que carrega sessão.
- Que a barreira seja do dado e não do modelo. As arquiteturas medidas vão de 115 mil
  a 1,6 milhão de parâmetros, todas treinadas do zero em 108 locutores. Nenhum modelo
  pré-treinado foi avaliado.
- Que os desvios entre sementes sejam intervalos de confiança. Não são, e não há
  reamostragem por locutor.
- Nada sobre gravações em sessões diferentes. **Cada locutor do VCTK foi gravado uma
  única vez**, e nenhum protocolo separa voz de sessão quando existe uma sessão por
  pessoa. É a limitação mais séria do trabalho, e ela exige outro corpus.

---

## 10. Reprodução

```bash
# Protocolos principais
SR_CONFIG=configs/vctk.env                python experiments/run_experiment.py
SR_CONFIG=configs/vctk_cross_mic.env      python experiments/run_experiment.py
SR_CONFIG=configs/vctk_cross_mic_inverso.env  python experiments/run_experiment.py
SR_CONFIG=configs/vctk_cross_mic_disjunto.env python experiments/run_experiment.py

# Controle do arcabouço
SR_CONFIG=configs/vctk.env python experiments/run_experiment.py --permute-labels

# Matriz de transferência, uma arquitetura por execução
ARCHITECTURES=temporal_attention MODELS_PATH=runs/models/matriz9_temporal_attention \
  python experiments/transfer_matrix.py --config configs/vctk_transfer_matrix.env

# Diagnósticos
SR_CONFIG=configs/vctk_cross_mic.env python experiments/channel_transfer.py
python experiments/error_structure.py
python experiments/session_or_transducer.py
```

Todos exigem `KERAS_BACKEND=torch`.

---

### Fonte: `docs/relatorio_8k.md`

# Reexecução comparativa · BrSD e VCTK a 8 kHz

Os dados e as features estão no HDD. O BrSD contém 400 gravações de 80 locutores; o VCTK usa 21.523 pares de gravações de 108 locutores, em dois microfones. Neste relatório, uma **gravação** (ou **amostra**) é um arquivo de áudio e sua matriz de MFCC. “Enunciado” significa o trecho falado nesse arquivo; não designa uma duração fixa.

## Parâmetros de extração

| Parâmetro | BrSD | VCTK mic1 | VCTK mic2 |
|---|---:|---:|---:|
| Taxa nominal do áudio (Hz) | 48000 | 48000 | 48000 |
| Taxa final (Hz) | 8000 | 8000 | 8000 |
| VAD | false | false | false |
| Limiar VAD (dB; desligado) | 30 | 30 | 30 |
| Pré-ênfase | 0.97 | 0.97 | 0.97 |
| Coeficientes MFCC | 40 | 40 | 40 |
| Janela (amostras) | 256 | 256 | 256 |

Cinco WAVs do BrSD (106–110) têm taxa nativa de 44,1 kHz; todos os outros áudios usados têm 48 kHz. Cada arquivo é filtrado na própria taxa nativa, antes de ser reamostrado diretamente para 8 kHz.

Filtro anti-aliasing: Butterworth de ordem 8, fase zero, corte de 3,6 kHz. Reamostragem polifásica; salto de 128 amostras (16 ms) para janelas de 256 amostras (32 ms). O limite do modelo é o menor enunciado do respectivo corpus; os dois microfones VCTK compartilham o menor valor observado entre eles.

## Comprimento das gravações após a extração

| Trilha | Gravações | Mínimo (quadros) | Média | Máximo (quadros) | Gravações extremas |
|---|---:|---:|---:|---:|---|
| brsd | 400 | 1007 | 2112.95 | 7241 | mín. L28/E4; máx. L2/E1 |
| vctk8k_mic1 | 21523 | 77 | 223.98 | 1035 | mín. L7/E51; máx. L45/E23 |
| vctk8k_mic2 | 21523 | 77 | 223.98 | 1035 | mín. L7/E51; máx. L45/E23 |
| vctk_combinado | 43046 | 77 | 223.98 | 1035 | mín. L7/E51; máx. L45/E23 |

A duração de cada gravação está no arquivo `runs/features/relatorio_comprimentos_8k.csv`. Um quadro avança 16 ms.

## Amostras usadas pelas redes

As três redes recebem as mesmas divisões. No BrSD, cada rede usa as cinco leituras por locutor. No VCTK, cada rede do experimento mic1 usa só mic1; cada rede do experimento mic2 usa só mic2. Não há mistura de microfones nestes experimentos.

| Trilha | Redes | Partições | Treino por partição | Validação | Teste |
|---|---|---:|---:|---:|---:|
| brsd | CNN, Temporal CNN, Attention | 5 | 240 | 80 | 80 |
| vctk8k_mic1 | CNN, Temporal CNN, Attention | 5 | 12913–12914–12915 | 4304–4305 | 4304–4305 |
| vctk8k_mic2 | CNN, Temporal CNN, Attention | 5 | 12913–12914–12915 | 4304–4305 | 4304–4305 |

Contagens exatas por partição (cada microfone do VCTK tem os mesmos totais):

| Corpus | Partição | Treino | Validação | Teste |
|---|---:|---:|---:|---:|
| BrSD | 1 | 240 | 80 | 80 |
| BrSD | 2 | 240 | 80 | 80 |
| BrSD | 3 | 240 | 80 | 80 |
| BrSD | 4 | 240 | 80 | 80 |
| BrSD | 5 | 240 | 80 | 80 |
| VCTK por microfone | 1 | 12913 | 4305 | 4305 |
| VCTK por microfone | 2 | 12913 | 4305 | 4305 |
| VCTK por microfone | 3 | 12914 | 4304 | 4305 |
| VCTK por microfone | 4 | 12915 | 4304 | 4304 |
| VCTK por microfone | 5 | 12914 | 4305 | 4304 |

No BrSD, cada locutor tem 5 gravações: em cada partição, 3 vão para treino, 1 para validação e 1 para teste (60%/20%/20%). O teste gira de E1 a E5 entre as cinco partições; a validação é sorteada entre as outras quatro com semente 42. Cada teste contém o mesmo número de leitura de todos os locutores.

No VCTK, 107 locutores têm 200 gravações por microfone: 120 para treino, 40 para validação e 40 para teste em cada partição (60%/20%/20%). O locutor p362 tem 123 gravações; por isso seus números e os totais variam ligeiramente. Os índices ordenados são distribuídos em cinco grupos intercalados: na partição k, o grupo k é teste, o seguinte é validação e os três restantes são treino. Mic1 e mic2 usam exatamente os mesmos papéis para cada par de gravações.

| Trilha | Treino por locutor (mín.–máx.) | Validação por locutor | Teste por locutor |
|---|---:|---:|---:|
| brsd | 3–3 | 1–1 | 1–1 |
| vctk8k_mic1 | 73–120 | 24–40 | 24–40 |
| vctk8k_mic2 | 73–120 | 24–40 | 24–40 |

O arquivo `runs/features/relatorio_particoes_8k.csv` informa o papel de **cada** gravação em **cada** partição. As tabelas acima mostram o intervalo de quantidades, não um intervalo contínuo de números de arquivo: os grupos do VCTK são intercalados.

Para os locutores ilustrados, os cinco primeiros índices da partição 1 são:

| Trilha | E1 | E2 | E3 | E4 | E5 |
|---|---|---|---|---|---|
| brsd | teste | validacao | treino | treino | treino |
| vctk8k_mic1 | teste | validacao | treino | treino | treino |
| vctk8k_mic2 | teste | validacao | treino | treino | treino |

## Tamanho das redes

| Trilha | Rede | Entrada (MFCC × quadros) | Classes | Parâmetros | Modelo médio (MiB) |
|---|---|---:|---:|---:|---:|
| brsd | cnn | 40 × 1007 | 80 | 452.848 | 5.21 |
| brsd | temporal_cnn | 40 × 1007 | 80 | 107.536 | 1.27 |
| brsd | attention | 40 × 1007 | 80 | 1.730.640 | 19.93 |
| vctk8k_mic1 | cnn | 40 × 77 | 108 | 341.004 | 3.93 |
| vctk8k_mic1 | temporal_cnn | 40 × 77 | 108 | 114.732 | 1.35 |
| vctk8k_mic1 | attention | 40 × 77 | 108 | 1.618.796 | 18.65 |
| vctk8k_mic2 | cnn | 40 × 77 | 108 | 341.004 | 3.93 |
| vctk8k_mic2 | temporal_cnn | 40 × 77 | 108 | 114.732 | 1.35 |
| vctk8k_mic2 | attention | 40 × 77 | 108 | 1.618.796 | 18.65 |

## Resultados das redes

| Trilha | Rede | Acurácia média ± desvio | F1 macro médio | Cinco partições (%) |
|---|---|---:|---:|---|
| brsd | cnn | 72.25 ± 2.78% | 0.646 | 67.5, 72.5, 75.0, 71.2, 75.0 |
| brsd | temporal_cnn | 76.75 ± 4.00% | 0.703 | 71.2, 81.2, 76.2, 81.2, 73.8 |
| brsd | attention | 47.50 ± 7.03% | 0.388 | 37.5, 58.8, 43.8, 50.0, 47.5 |
| vctk8k_mic1 | cnn | 93.09 ± 0.36% | 0.931 | 93.1, 93.1, 93.5, 92.4, 93.2 |
| vctk8k_mic1 | temporal_cnn | 96.04 ± 0.41% | 0.960 | 96.1, 95.7, 96.6, 96.3, 95.5 |
| vctk8k_mic1 | attention | 95.06 ± 0.23% | 0.951 | 95.3, 95.3, 94.8, 95.1, 94.8 |
| vctk8k_mic2 | cnn | 86.21 ± 0.46% | 0.862 | 86.8, 85.7, 85.9, 86.7, 85.9 |
| vctk8k_mic2 | temporal_cnn | 92.39 ± 0.72% | 0.924 | 91.7, 91.6, 92.6, 93.5, 92.6 |
| vctk8k_mic2 | attention | 88.76 ± 0.59% | 0.887 | 88.1, 88.6, 89.3, 89.6, 88.3 |

Os corpora diferem em número de locutores, duração e conteúdo. As acurácias são apresentadas lado a lado como resultados de tarefas fechadas próprias de cada corpus, sem tratá-las como uma medição direta da mesma população.

## Cadeia de processamento lado a lado

À esquerda está o locutor 1 do BrSD; nas outras colunas, p225 do VCTK nos microfones 1 e 2. Os números de locutor dos dois corpora não identificam a mesma pessoa. Cada número abaixo é apenas um índice para organizar as figuras: as leituras 1–5 do BrSD são textos diferentes entre si, e também não correspondem aos textos do VCTK. No VCTK, mic1 e mic2 são as duas captações da mesma leitura.

As formas de onda e os MFCCs mostram cada gravação inteira; a escala de tempo varia entre as colunas. A linha vermelha tracejada marca o fim do trecho entregue à rede: os primeiros 1007 quadros (aproximadamente 16.11 s) no BrSD e 77 quadros (aproximadamente 1.23 s) no VCTK. A extração guarda o MFCC completo; o corte ocorre na entrada da rede.

| Figura | BrSD: arquivo e duração | VCTK mic1: arquivo e duração | VCTK mic2: arquivo e duração |
|---:|---|---|---|
| 1 | 1.wav · 71.54 s | p225_001_mic1.flac · 2.05 s | p225_001_mic2.flac · 2.05 s |
| 2 | 2.wav · 38.78 s | p225_002_mic1.flac · 3.94 s | p225_002_mic2.flac · 3.94 s |
| 3 | 3.wav · 32.83 s | p225_003_mic1.flac · 7.59 s | p225_003_mic2.flac · 7.59 s |
| 4 | 4.wav · 32.18 s | p225_004_mic1.flac · 4.41 s | p225_004_mic2.flac · 4.41 s |
| 5 | 5.wav · 35.76 s | p225_005_mic1.flac · 6.32 s | p225_005_mic2.flac · 6.32 s |

### Gravação ilustrativa 1


### Gravação ilustrativa 2


### Gravação ilustrativa 3


### Gravação ilustrativa 4


### Gravação ilustrativa 5


## Fontes dos corpora

- [BrSD, página dos autores](https://sites.google.com/view/brsduem).
- [VCTK 0.92, University of Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3443).

---

### Fonte: `docs/comparacao_pareada_77_153.md`

# Comparação pareada: 77 versus 153 quadros no VCTK

As duas durações usam as mesmas 18.067 gravações e, em cada uma das
cinco partições, os mesmos conjuntos de treino, validação e teste.
Todos os modelos foram treinados novamente com a duração correspondente.
Ambos usam os primeiros quadros de cada gravação; não há seleção de
atividade vocal neste experimento.

| Microfone | Rede | 77 quadros | 153 quadros | Diferença (p.p.) |
|---|---|---:|---:|---:|
| mic1 | cnn | 93.91% | 97.40% | +3.49 |
| mic1 | temporal_cnn | 95.96% | 98.60% | +2.64 |
| mic1 | attention | 94.66% | 98.25% | +3.59 |
| mic2 | cnn | 86.89% | 93.91% | +7.02 |
| mic2 | temporal_cnn | 92.54% | 97.08% | +4.54 |
| mic2 | attention | 89.10% | 95.25% | +6.15 |

## Quantidade de quadros de entrada

| Seleção | Gravações | Quadros por gravação | Total de quadros |
|---|---:|---:|---:|
| Corpus completo, 77 | 21.523 | 77 | 1.657.271 |
| Seleção pareada, 77 | 18.067 | 77 | 1.391.159 |
| Seleção pareada, 153 | 18.067 | 153 | 2.764.251 |

Na seleção pareada, 153 entrega 1.373.092 quadros adicionais, ou
98,7% mais quadros de entrada que 77. A unidade de treino é a
gravação, de modo que ambas as durações continuam com 18.067 exemplos.
Entre todos os cortes inteiros possíveis no corpus, 153 maximiza o
produto gravações elegíveis × quadros por gravação.
A acurácia dentro do mesmo microfone pode refletir voz e canal;
o teste entre microfones avalia essa transferência separadamente.

---

### Fonte: `docs/resultado_atividade_baixa_vctk.md`

# VCTK: quadros sem seleção, com atividade e com baixa atividade

Foram usadas as mesmas 17.272 gravações pareadas, dos 108 locutores, com 10 quadros por gravação em cada condição.
As posições de atividade e baixa atividade foram escolhidas a partir do
áudio original inteiro, após o filtro e a reamostragem para 8 kHz.
A condição sem seleção amostra uniformemente o áudio inteiro;
as outras duas amostram posições com rótulo comum aos dois microfones.
Os quadros são espalhados pela gravação e concatenados em uma entrada
curta; eles não formam necessariamente um trecho contínuo de fala.
Há uma margem de um quadro em cada transição. Treino, validação e teste
usam as mesmas gravações nas três condições e nos dois microfones.

O detector mede energia, não uma anotação humana de fala. Baixa atividade
não deve ser interpretada como silêncio garantido.

## Teste no mesmo microfone

| Microfone | Rede | Sem seleção | Atividade | Baixa atividade |
|---|---|---:|---:|---:|
| mic1 | cnn | 78,20% | 87,39% | 68,23% |
| mic1 | temporal_cnn | 74,89% | 85,06% | 65,86% |
| mic1 | attention | 83,60% | 90,95% | 69,77% |
| mic2 | cnn | 66,80% | 78,55% | 54,68% |
| mic2 | temporal_cnn | 64,34% | 77,42% | 54,49% |
| mic2 | attention | 72,60% | 82,11% | 56,58% |

## Teste ao trocar de microfone

| Treino → teste | Rede | Sem seleção | Atividade | Baixa atividade |
|---|---|---:|---:|---:|
| mic1 → mic2 | cnn | 18,30% | 26,23% | 9,65% |
| mic1 → mic2 | temporal_cnn | 15,28% | 25,84% | 7,56% |
| mic1 → mic2 | attention | 19,20% | 29,38% | 9,98% |
| mic2 → mic1 | cnn | 23,14% | 35,66% | 8,06% |
| mic2 → mic1 | temporal_cnn | 20,75% | 29,88% | 6,67% |
| mic2 → mic1 | attention | 24,35% | 36,65% | 10,62% |

Acaso em 108 classes: 0,93%. Cada célula resume cinco partições.
A detecção usa o mesmo critério de energia nas três condições;
a interpretação deve levar em conta classificação incorreta de fala fraca,
respiração e ruído. Os modelos têm inicialização estocástica; uma execução
por partição não mede toda a variabilidade de treino.

Arquivo de seleção: `docs/vctk_activity_probe_selection.json`; SHA-256 das chaves: `2bb3393d25d339e7cec8692156009f2a6ad1d39935bf294994bb78fa01eb7368`.

---

### Fonte: `docs/resultado_atividade_mais_baixa_vctk.md`

# VCTK: efeito de combinar atividade e baixa atividade

As condições foram obtidas dos mesmos arquivos completos, após o
filtro para 8 kHz. O detector usa energia (`top_db=30`); baixa
atividade não equivale necessariamente a silêncio puro.
Os quadros escolhidos são ordenados no tempo, mas podem vir de
trechos separados da mesma gravação.

## Entradas de 10 quadros

17.272 gravações pareadas; 108 locutores; mínimo de 3 gravações por locutor em cada grupo de teste.

O misto contém **5 quadros de atividade + 5 de baixa atividade**.
A comparação com 10 de atividade testa a substituição de
metade dos quadros, mantendo a largura fixa.

### Teste no mesmo microfone

| Microfone | Rede | Sem seleção | Atividade | Baixa atividade | Atividade + baixa atividade |
|---|---|---:|---:|---:|---:|
| mic1 | cnn | 78,20% | 87,39% | 68,23% | 80,77% |
| mic1 | temporal_cnn | 74,89% | 85,06% | 65,86% | 77,69% |
| mic1 | attention | 83,60% | 90,95% | 69,77% | 85,58% |
| mic2 | cnn | 66,80% | 78,55% | 54,68% | 69,48% |
| mic2 | temporal_cnn | 64,34% | 77,42% | 54,49% | 68,17% |
| mic2 | attention | 72,60% | 82,11% | 56,58% | 75,02% |

### Teste ao trocar de microfone

| Treino → teste | Rede | Sem seleção | Atividade | Baixa atividade | Atividade + baixa atividade |
|---|---|---:|---:|---:|---:|
| mic1 → mic2 | cnn | 18,30% | 26,23% | 9,65% | 20,32% |
| mic1 → mic2 | temporal_cnn | 15,28% | 25,84% | 7,56% | 16,85% |
| mic1 → mic2 | attention | 19,20% | 29,38% | 9,98% | 20,69% |
| mic2 → mic1 | cnn | 23,14% | 35,66% | 8,06% | 25,29% |
| mic2 → mic1 | temporal_cnn | 20,75% | 29,88% | 6,67% | 21,60% |
| mic2 → mic1 | attention | 24,35% | 36,65% | 10,62% | 25,35% |

## Entradas de 20 quadros

17.270 gravações pareadas; 108 locutores; mínimo de 3 gravações por locutor em cada grupo de teste.

O misto contém **10 quadros de atividade + 10 de baixa atividade**.
O controle tem 20 quadros de atividade. Ambos usam a mesma
seleção de gravações, as mesmas partições e a mesma largura.

### Teste no mesmo microfone

| Microfone | Rede | Atividade | Atividade + baixa atividade |
|---|---|---:|---:|
| mic1 | cnn | 92,50% | 90,14% |
| mic1 | temporal_cnn | 93,51% | 90,90% |
| mic1 | attention | 95,17% | 93,35% |
| mic2 | cnn | 86,80% | 82,26% |
| mic2 | temporal_cnn | 88,37% | 83,39% |
| mic2 | attention | 89,00% | 87,04% |

### Teste ao trocar de microfone

| Treino → teste | Rede | Atividade | Atividade + baixa atividade |
|---|---|---:|---:|
| mic1 → mic2 | cnn | 30,28% | 23,97% |
| mic1 → mic2 | temporal_cnn | 27,41% | 17,52% |
| mic1 → mic2 | attention | 31,23% | 21,07% |
| mic2 → mic1 | cnn | 40,17% | 29,28% |
| mic2 → mic1 | temporal_cnn | 32,04% | 21,33% |
| mic2 → mic1 | attention | 39,64% | 27,23% |

O acaso em 108 classes é 0,93%. Cada célula resume cinco partições.
Os dois grupos de tamanho de entrada usam coortes ligeiramente
diferentes (17.272 e 17.270 gravações); compare principalmente
condições **dentro de cada grupo**. A inicialização das redes é
estocástica e não foi repetida com várias sementes.

---

### Fonte: `docs/resultado_todos_quadros_vctk.md`

# Todos os quadros de atividade e baixa atividade — VCTK

Mesmas 17.270 gravações pareadas, 108 locutores e cinco partições em
todas as condições. Cada gravação vira um vetor de 80 números:
média e desvio dos 40 MFCCs calculados sobre **todos os quadros** da
condição indicada. O classificador linear é o mesmo em todas as linhas.
A normalização é ajustada apenas no treino do microfone de origem.
O detector é de energia; baixa atividade não significa silêncio puro.

| Condição | Quadros medianos por gravação | Mesmo mic1 | Mesmo mic2 | Mic1 → Mic2 | Mic2 → Mic1 |
|---|---:|---:|---:|---:|---:|
| Todos, sem seleção | 209 | 95,30% | 89,29% | 34,31% | 37,32% |
| Todos de atividade | 88 | 91,26% | 82,72% | 40,04% | 45,91% |
| Todos de baixa atividade | 55 | 74,56% | 61,80% | 13,06% | 22,13% |
| 20 de atividade | 20 | 87,01% | 75,98% | 35,99% | 42,72% |
| 10 atividade + 10 baixa | 20 | 85,71% | 72,84% | 26,44% | 32,47% |

Acaso: 0,93% em 108 classes. As condições de 20 quadros usam
os mesmos registros e o mesmo classificador das condições com
todos os quadros; diferenças de duração são parte do efeito medido.
A seleção de quadros de baixa atividade ainda pode conter fala fraca,
respiração e ruído.

---

### Fonte: `docs/resultado_vctk_bal100_atividade.md`

# VCTK: atividade e baixa atividade, 100 locutores balanceados

Cada uma das 100 pessoas tem 25 gravações pareadas nos dois microfones.
Em cada uma das cinco partições, por pessoa são 15 para treino, 5 para
validação e 5 para teste. Todas as condições usam as mesmas gravações
e os mesmos papéis. Mapeamento em `vctk_bal100_selection.json`.
Baixa atividade é uma medida de energia, não silêncio anotado.

A comparação de 20 quadros confronta atividade e baixa atividade.
A comparação de 40 confronta atividade pura, mistura 20+20 e quadros
uniformemente amostrados do áudio inteiro. São quadros de posições
possivelmente não contíguas, ordenados pela posição temporal.

## 20 quadros

### Teste no mesmo microfone

| Microfone | Rede | 20 atividade | 20 baixa |
|---|---|---:|---:|
| mic1 | cnn | 91,64% | 75,40% |
| mic1 | temporal_cnn | 83,24% | 68,08% |
| mic1 | attention | 89,04% | 68,96% |
| mic2 | cnn | 82,32% | 58,08% |
| mic2 | temporal_cnn | 75,44% | 54,12% |
| mic2 | attention | 74,96% | 53,20% |

### Troca de microfone

| Treino → teste | Rede | 20 atividade | 20 baixa |
|---|---|---:|---:|
| mic1 → mic2 | cnn | 31,40% | 10,84% |
| mic1 → mic2 | temporal_cnn | 26,72% | 5,92% |
| mic1 → mic2 | attention | 30,04% | 7,48% |
| mic2 → mic1 | cnn | 40,92% | 10,76% |
| mic2 → mic1 | temporal_cnn | 27,56% | 6,60% |
| mic2 → mic1 | attention | 38,16% | 7,24% |

## 40 quadros

### Teste no mesmo microfone

| Microfone | Rede | 40 sem seleção | 40 atividade | 20 atividade + 20 baixa |
|---|---|---:|---:|---:|
| mic1 | cnn | 93,68% | 93,60% | 93,96% |
| mic1 | temporal_cnn | 86,64% | 88,08% | 87,48% |
| mic1 | attention | 92,04% | 89,84% | 91,60% |
| mic2 | cnn | 85,36% | 85,60% | 84,32% |
| mic2 | temporal_cnn | 78,84% | 83,32% | 81,28% |
| mic2 | attention | 79,00% | 77,76% | 80,60% |

### Troca de microfone

| Treino → teste | Rede | 40 sem seleção | 40 atividade | 20 atividade + 20 baixa |
|---|---|---:|---:|---:|
| mic1 → mic2 | cnn | 28,08% | 34,92% | 25,28% |
| mic1 → mic2 | temporal_cnn | 17,08% | 29,60% | 17,08% |
| mic1 → mic2 | attention | 20,64% | 32,44% | 20,92% |
| mic2 → mic1 | cnn | 35,64% | 39,96% | 33,76% |
| mic2 → mic1 | temporal_cnn | 20,80% | 27,28% | 19,56% |
| mic2 → mic1 | attention | 28,44% | 36,60% | 29,16% |

Acaso: 1% em 100 classes. Cada célula agrega cinco partições.
O experimento não testa os 108 locutores do corpus original;
compare apenas condições desta coorte de 100 locutores.

---

### Fonte: `output/vctk16_all_models_results.md`

# VCTK 16 kHz — comparação das três redes

108 locutores; 120 gravações pareadas por locutor; 111 quadros por gravação; cinco folds 60/20/20. A divisão é por gravação e idêntica nos dois microfones. Cada modelo e seu z-score usam só o microfone de origem no treino. Teste cruzado usa o mesmo checkpoint.

Os quadros foram escolhidos pela maior soma de RMS normalizado num intervalo contínuo de 111 quadros, comum ao par de microfones.

| Rede | Mic1→Mic1 | Mic1→Mic2 | Mic2→Mic1 | Mic2→Mic2 |
|---|---:|---:|---:|---:|
| x-vector, MFCC | 98.69 ± 0.46% | 54.21 ± 2.08% | 57.57 ± 3.63% | 98.60 ± 0.20% |
| x-vector, MFCC+Δ+ΔΔ | 98.56 ± 0.86% | 61.57 ± 4.12% | 64.97 ± 2.46% | 98.34 ± 0.37% |
| x-vector, escolha por fold | 98.98 ± 0.28% | 60.08 ± 5.76% | 60.09 ± 5.41% | 98.51 ± 0.36% |
| CNN temporal, MFCC+Δ+ΔΔ | 96.23 ± 0.78% | 46.51 ± 2.19% | 51.44 ± 2.10% | 95.99 ± 0.82% |
| CNN cepstral, MFCC+Δ+ΔΔ | 93.59 ± 0.41% | 41.50 ± 2.56% | 46.96 ± 1.17% | 91.16 ± 0.61% |

## Precisão macro

| Rede | Mic1→Mic1 | Mic1→Mic2 | Mic2→Mic1 | Mic2→Mic2 |
|---|---:|---:|---:|---:|
| x-vector, MFCC | 98.79 ± 0.38% | 62.73 ± 2.08% | 62.47 ± 4.21% | 98.69 ± 0.16% |
| x-vector, MFCC+Δ+ΔΔ | 98.65 ± 0.77% | 68.14 ± 2.93% | 69.71 ± 3.06% | 98.44 ± 0.35% |
| x-vector, escolha por fold | 99.03 ± 0.27% | 66.37 ± 5.31% | 65.38 ± 5.02% | 98.60 ± 0.34% |
| CNN temporal, MFCC+Δ+ΔΔ | 96.48 ± 0.70% | 54.96 ± 3.20% | 52.82 ± 2.65% | 96.22 ± 0.77% |
| CNN cepstral, MFCC+Δ+ΔΔ | 94.14 ± 0.35% | 49.53 ± 2.24% | 50.75 ± 2.24% | 91.94 ± 0.48% |

## Recall macro

| Rede | Mic1→Mic1 | Mic1→Mic2 | Mic2→Mic1 | Mic2→Mic2 |
|---|---:|---:|---:|---:|
| x-vector, MFCC | 98.69 ± 0.46% | 54.21 ± 2.08% | 57.57 ± 3.63% | 98.60 ± 0.20% |
| x-vector, MFCC+Δ+ΔΔ | 98.56 ± 0.86% | 61.57 ± 4.12% | 64.97 ± 2.46% | 98.34 ± 0.37% |
| x-vector, escolha por fold | 98.98 ± 0.28% | 60.08 ± 5.76% | 60.09 ± 5.41% | 98.51 ± 0.36% |
| CNN temporal, MFCC+Δ+ΔΔ | 96.23 ± 0.78% | 46.51 ± 2.19% | 51.44 ± 2.10% | 95.99 ± 0.82% |
| CNN cepstral, MFCC+Δ+ΔΔ | 93.59 ± 0.41% | 41.50 ± 2.56% | 46.96 ± 1.17% | 91.16 ± 0.61% |

## F1 macro

| Rede | Mic1→Mic1 | Mic1→Mic2 | Mic2→Mic1 | Mic2→Mic2 |
|---|---:|---:|---:|---:|
| x-vector, MFCC | 98.69 ± 0.45% | 51.72 ± 1.87% | 53.81 ± 3.88% | 98.60 ± 0.20% |
| x-vector, MFCC+Δ+ΔΔ | 98.55 ± 0.86% | 59.14 ± 4.08% | 61.85 ± 3.01% | 98.34 ± 0.38% |
| x-vector, escolha por fold | 98.98 ± 0.28% | 57.36 ± 6.32% | 56.61 ± 5.57% | 98.51 ± 0.37% |
| CNN temporal, MFCC+Δ+ΔΔ | 96.22 ± 0.78% | 44.62 ± 2.80% | 46.27 ± 2.15% | 95.95 ± 0.84% |
| CNN cepstral, MFCC+Δ+ΔΔ | 93.56 ± 0.40% | 39.05 ± 2.21% | 43.47 ± 1.38% | 91.08 ± 0.61% |

## EER um-contra-todos (softmax)

| Rede | Mic1→Mic1 | Mic1→Mic2 | Mic2→Mic1 | Mic2→Mic2 |
|---|---:|---:|---:|---:|
| x-vector, MFCC | 0.14 ± 0.08% | 8.88 ± 0.97% | 9.29 ± 0.91% | 0.13 ± 0.04% |
| x-vector, MFCC+Δ+ΔΔ | 0.15 ± 0.10% | 7.28 ± 1.07% | 6.79 ± 0.68% | 0.19 ± 0.06% |
| x-vector, escolha por fold | 0.12 ± 0.06% | 7.67 ± 1.72% | 8.42 ± 1.77% | 0.17 ± 0.04% |
| CNN temporal, MFCC+Δ+ΔΔ | 0.46 ± 0.08% | 9.85 ± 1.17% | 10.77 ± 0.67% | 0.44 ± 0.04% |
| CNN cepstral, MFCC+Δ+ΔΔ | 0.86 ± 0.08% | 10.23 ± 0.45% | 10.16 ± 0.49% | 1.04 ± 0.11% |

As linhas com MFCC+Δ+ΔΔ usam a mesma representação para comparar arquiteturas. Essa representação foi escolhida inicialmente pela validação do fold 1 da x-vector; portanto, essa escolha global não é uma seleção aninhada independente dos cinco testes. A linha “escolha por fold” corrige isso para a x-vector: em cada fold, decide entre MFCC e MFCC+Δ+ΔΔ usando somente a validação daquele fold e microfone de origem.

A seleção das 120 gravações mais longas por locutor favorece áudios longos. Estes números se aplicam a essa coorte, não a todas as gravações do VCTK. A janela temporal comum usa o RMS dos dois microfones; isso também faz parte do protocolo offline pareado.

EER e minDCF por classe, quando presentes nos arquivos de métricas, foram calculados com softmax em conjunto fechado e não equivalem ao protocolo aberto de verificação por embeddings.

---

### Fonte: `output/vctk16_silero/resultado_preliminar.md`

# Silero VAD no reconhecimento de locutor — resultado preliminar

Coorte derivada do experimento VCTK a 16 kHz: 108 locutores, 105 gravações pareadas por locutor, 70 quadros por gravação. Os cinco grupos originais foram preservados, com 21 gravações por locutor em cada grupo. Neste relatório só o fold 1 foi executado: 60% treino, 20% validação, 20% teste. A x-vector usa 30 MFCC + delta + delta-delta e z-score calculado apenas no treino do microfone de origem. Não foram aplicados CMN nem RASTA.

O controle e a condição Silero usam exatamente as mesmas gravações, locutores, folds, rede e parâmetros de treino. No controle, a janela de 70 quadros é escolhida no áudio após o trim; na condição Silero, os trechos de fala detectados em cada microfone são reunidos em ordem temporal antes da extração e escolha da janela. As escolhas de trechos são independentes por microfone.

| Condição, treino mic1 | Teste mic1 | Teste mic2 |
|---|---:|---:|
| Sem VAD: acurácia | 96,38% | 53,44% |
| Com Silero: acurácia | 96,47% | 54,67% |
| Sem VAD: F1 macro | 96,37% | 51,48% |
| Com Silero: F1 macro | 96,42% | 51,91% |
| Sem VAD: EER um-contra-todos | 0,40% | 8,86% |
| Com Silero: EER um-contra-todos | 0,32% | 8,20% |

O controle com treino mic2 também produziu um checkpoint de validação, mas a
execução foi interrompida na época 74 antes de concluir a parada antecipada.
A avaliação desse checkpoint deu 96,34% em mic2→mic2 e 60,19% em mic2→mic1.
A condição Silero com treino mic2 ainda está em andamento.

No teste cruzado mic1→mic2, a diferença preliminar é +1,23 ponto percentual de acurácia e +0,43 ponto de F1 macro. Um fold não estabelece que o ganho se repete nos cinco. O EER vem de scores softmax um-contra-todos em conjunto fechado; não equivale a um protocolo aberto de verificação por embeddings.

O gráfico [example_p225_241_vad_original.png](example_p225_241_vad_original.png) mostra a classificação de fala do Silero no arquivo de entrada e após o trim. A fonte `wav48_silence_trimmed` já passou pelo tratamento distribuído com o VCTK. Trechos sem fala detectada não são silêncio comprovado.

---

### Fonte: `output/vctk16_cmn_pilot.md`

# CMN no VCTK 16 kHz — piloto

Condição: subtração integral da média temporal dos 30 MFCCs estáticos de cada
gravação, após a seleção dos 111 quadros. Deltas e delta-deltas não mudam sob
subtração de constante. O z-score continua estimado apenas no treino de origem.
Mesma coorte, divisão, arquitetura x-vector dinâmica e semente da referência.

Resultado concluído: fold 1, origem mic1, 2.592 gravações de teste pareadas.

| Teste | Referência | CMN | Diferença |
|---|---:|---:|---:|
| mic1→mic1 | 98,77% | 93,40% | −5,36 pp |
| mic1→mic2 | 62,54% | 85,57% | +23,03 pp |

Na troca de microfone, CMN corrigiu 780 erros da referência e introduziu 183
erros novos. A melhora ocorreu em 69 dos 108 locutores e a piora em 31; os
demais empataram. Intervalo exploratório de 95% por bootstrap de locutores para
a diferença: +16,63 a +29,51 pontos percentuais. Dentro de mic1, a diferença
foi −6,94 a −3,90 pontos. Este intervalo descreve apenas o fold piloto;
os folds não são amostras independentes de locutores.

O modelo foi escolhido pela validação (94,21% com CMN, 99,46% na referência),
sem seleção por teste. Ainda faltam os outros quatro folds e o treino de origem
mic2. O serviço `sr-vctk16-cmn-all.service` executa essas condições e grava
`/media/lsmsqt/HDD/sr_project/cmn_vctk16_results/dynamic/summary.json`
quando terminar. Verificar com `systemctl --user status
sr-vctk16-cmn-all.service` e `journalctl --user -u
sr-vctk16-cmn-all.service -n 30 --no-pager`.

O piloto mostra ganho de transferência neste protocolo, acompanhado de perda
intra-microfone. Ele não demonstra que a identidade vocal foi isolada dos
efeitos de sessão. RASTA ainda não foi executado.

---

### Fonte: `output/vctk16_cmn_gpu_b128_comparison.md`

# CMN × referência — VCTK 16 kHz

X-vector, MFCC+Δ+ΔΔ, lote 128; 108 locutores, 120 leituras por locutor, 111 quadros; cinco folds 60/20/20. Média ± desvio entre folds. Os folds compartilham locutores e não são independentes.

| Treino → teste | Referência | CMN | Diferença de acurácia |
|---|---:|---:|---:|
| mic1->mic1 | 98.84 ± 0.64% | 94.30 ± 1.79% | -4.54 pp |
| mic1->mic2 | 63.12 ± 3.25% | 86.60 ± 3.63% | +23.47 pp |
| mic2->mic1 | 65.85 ± 2.97% | 81.81 ± 5.34% | +15.96 pp |
| mic2->mic2 | 98.19 ± 1.07% | 93.90 ± 2.68% | -4.29 pp |

CMN subtrai a média temporal dos 30 MFCCs estáticos de cada gravação. As duas condições mantêm os deltas, folds e sementes e ajustam o z-score somente no treino de origem. O checkpoint é escolhido pela validação. Melhora de transferência não demonstra isolamento da identidade vocal dos efeitos de sessão.

CMN: `/media/lsmsqt/HDD/sr_project/cmn_vctk16_gpu_b128`
Referência: `/media/lsmsqt/HDD/sr_project/baseline_vctk16_gpu_b128`

---

### Fonte: `output/vctk16_channel_suite_comparison.md`

# VCTK 16 kHz — referência, CMN e RASTA

108 locutores × 120 leituras pareadas; 111 quadros; cinco folds 60/20/20; MFCC30+Δ+ΔΔ; lote 128. Média ± desvio entre folds. Uma célula só é preenchida após os cinco folds concluídos.

| Rede | Condição | mic1→mic1 | mic1→mic2 | mic2→mic1 | mic2→mic2 |
|---|---|---:|---:|---:|---:|
| xvector | baseline | 98.84 ± 0.64% | 63.12 ± 3.25% | 65.85 ± 2.97% | 98.19 ± 1.07% |
| xvector | cmn | 94.30 ± 1.79% | 86.60 ± 3.63% | 81.81 ± 5.34% | 93.90 ± 2.68% |
| xvector | rasta | 93.67 ± 3.28% | 85.44 ± 4.36% | 83.54 ± 3.17% | 94.90 ± 0.82% |
| cnn | baseline | 95.10 ± 0.28% | 43.53 ± 1.40% | 48.27 ± 1.27% | 92.69 ± 0.28% |
| cnn | cmn | 16.47 ± 0.61% | 14.89 ± 0.43% | 14.83 ± 0.48% | 15.56 ± 0.43% |
| cnn | rasta | 61.82 ± 0.92% | 45.54 ± 0.85% | 44.55 ± 1.50% | 59.13 ± 1.28% |
| temporal_cnn | baseline | 96.78 ± 0.57% | 47.80 ± 3.19% | 54.84 ± 1.35% | 96.27 ± 0.47% |
| temporal_cnn | cmn | 83.80 ± 1.28% | 74.21 ± 2.10% | 72.84 ± 1.44% | 84.88 ± 1.42% |
| temporal_cnn | rasta | 86.00 ± 1.22% | 73.05 ± 1.98% | 73.17 ± 2.05% | 87.85 ± 0.72% |

RASTA é aplicado no log-mel do áudio recortado completo, antes da DCT. Os mesmos índices de janela da referência são mantidos. Deltas são recalculados somente nos 111 quadros selecionados. RASTA não é combinado com CMN nesta ablação.

Todos os modelos usam apenas treino de origem para o z-score e validação de origem para escolher o checkpoint. Folds compartilham locutores; melhora cruzada não demonstra isolamento da sessão.

---

### Fonte: `output/vctk16_followup_suite_comparison.md`

# VCTK 16 kHz — comparação com CMN+RASTA e CMVN

30 MFCCs + Δ + ΔΔ, 111 quadros, lote 128; mesma coorte, folds e sementes. Acurácia média ± desvio nos cinco folds.

| Rede | Condição | mic1→mic1 | mic1→mic2 | mic2→mic1 | mic2→mic2 |
|---|---|---:|---:|---:|---:|
| xvector | baseline | 98.84 ± 0.64% | 63.12 ± 3.25% | 65.85 ± 2.97% | 98.19 ± 1.07% |
| xvector | cmn | 94.30 ± 1.79% | 86.60 ± 3.63% | 81.81 ± 5.34% | 93.90 ± 2.68% |
| xvector | rasta | pendente | pendente | pendente | pendente |
| xvector | cmn_rasta | pendente | pendente | pendente | pendente |
| xvector | cmvn | pendente | pendente | pendente | pendente |
| cnn | baseline | 95.10 ± 0.28% | 43.53 ± 1.40% | 48.27 ± 1.27% | 92.69 ± 0.28% |
| cnn | cmn | 16.47 ± 0.61% | 14.89 ± 0.43% | 14.83 ± 0.48% | 15.56 ± 0.43% |
| cnn | rasta | pendente | pendente | pendente | pendente |
| cnn | cmn_rasta | pendente | pendente | pendente | pendente |
| cnn | cmvn | pendente | pendente | pendente | pendente |
| temporal_cnn | baseline | 96.78 ± 0.57% | 47.80 ± 3.19% | 54.84 ± 1.35% | 96.27 ± 0.47% |
| temporal_cnn | cmn | 83.80 ± 1.28% | 74.21 ± 2.10% | 72.84 ± 1.44% | 84.88 ± 1.42% |
| temporal_cnn | rasta | pendente | pendente | pendente | pendente |
| temporal_cnn | cmn_rasta | pendente | pendente | pendente | pendente |
| temporal_cnn | cmvn | pendente | pendente | pendente | pendente |

CMN+RASTA: características RASTA existentes, depois subtração da média dos 30 MFCCs estáticos na janela; deltas mantidos.

CMVN: características baseline; média e desvio de cada MFCC estático calculados somente na janela da própria gravação. Estáticos centrados e divididos pelo desvio (piso 1e-8); Δ e ΔΔ divididos pelo mesmo desvio estático, equivalente à transformação dos deltas após CMVN. Não há combinação CMVN+RASTA.

O z-score global continua estimado somente no treino de origem. Checkpoint escolhido na validação de origem. Folds compartilham locutores; resultados não isolam efeitos de sessão.

---

### Fonte: `output/vctk40_suite_comparison.md`

# VCTK 16 kHz — comparação com CMN+RASTA e CMVN

40 MFCCs + Δ + ΔΔ, 111 quadros, lote 128; mesma coorte, folds e sementes. Acurácia média ± desvio nos cinco folds.

| Rede | Condição | mic1→mic1 | mic1→mic2 | mic2→mic1 | mic2→mic2 |
|---|---|---:|---:|---:|---:|
| xvector | baseline | pendente | pendente | pendente | pendente |
| xvector | cmn | pendente | pendente | pendente | pendente |
| xvector | rasta | pendente | pendente | pendente | pendente |
| xvector | cmn_rasta | pendente | pendente | pendente | pendente |
| xvector | cmvn | pendente | pendente | pendente | pendente |
| cnn | baseline | pendente | pendente | pendente | pendente |
| cnn | cmn | pendente | pendente | pendente | pendente |
| cnn | rasta | pendente | pendente | pendente | pendente |
| cnn | cmn_rasta | pendente | pendente | pendente | pendente |
| cnn | cmvn | pendente | pendente | pendente | pendente |
| temporal_cnn | baseline | pendente | pendente | pendente | pendente |
| temporal_cnn | cmn | pendente | pendente | pendente | pendente |
| temporal_cnn | rasta | pendente | pendente | pendente | pendente |
| temporal_cnn | cmn_rasta | pendente | pendente | pendente | pendente |
| temporal_cnn | cmvn | pendente | pendente | pendente | pendente |

CMN+RASTA: características RASTA existentes, depois subtração da média dos 40 MFCCs estáticos na janela; deltas mantidos.

CMVN: características baseline; média e desvio de cada MFCC estático calculados somente na janela da própria gravação. Estáticos centrados e divididos pelo desvio (piso 1e-8); Δ e ΔΔ divididos pelo mesmo desvio estático, equivalente à transformação dos deltas após CMVN. Não há combinação CMVN+RASTA.

O z-score global continua estimado somente no treino de origem. Checkpoint escolhido na validação de origem. Folds compartilham locutores; resultados não isolam efeitos de sessão.

---

### Fonte: `output/vctk16_isolated_suite_comparison.md`

# VCTK 16 kHz — normalizações separadas

30 MFCCs + Δ + ΔΔ; 111 quadros; cinco folds 60/20/20; GPU, lote 128.
Z-score apenas na condição zscore. CMN, CMVN e RASTA sem z-score global.
Z-score reaproveita os baselines concluídos com o mesmo protocolo.

| Rede | Condição | mic1→mic1 | mic1→mic2 | mic2→mic1 | mic2→mic2 |
|---|---|---:|---:|---:|---:|
| cnn | zscore | 95.10 ± 0.28% | 43.53 ± 1.40% | 48.27 ± 1.27% | 92.69 ± 0.28% |
| temporal_cnn | zscore | 96.78 ± 0.57% | 47.80 ± 3.19% | 54.84 ± 1.35% | 96.27 ± 0.47% |
| cnn | cmn | 12.25 ± 0.68% | 10.93 ± 0.48% | 11.30 ± 0.58% | 12.07 ± 0.46% |
| temporal_cnn | cmn | 92.43 ± 0.23% | 83.00 ± 1.29% | 79.62 ± 1.77% | 93.55 ± 1.13% |
| cnn | cmvn | 17.31 ± 0.24% | 15.79 ± 0.90% | 16.19 ± 0.47% | 17.32 ± 0.62% |
| temporal_cnn | cmvn | 85.91 ± 1.14% | 74.62 ± 1.85% | 75.12 ± 1.42% | 85.87 ± 1.28% |
| cnn | rasta | 52.67 ± 0.80% | 39.49 ± 1.03% | 39.26 ± 1.56% | 50.69 ± 1.74% |
| temporal_cnn | rasta | 92.89 ± 0.87% | 78.00 ± 2.26% | 75.91 ± 1.38% | 93.23 ± 0.63% |

CMN/CMVN por janela de gravação; CMVN escala Δ/ΔΔ pelo desvio dos estáticos.
RASTA no log-mel antes da DCT, com as janelas pareadas existentes.
X-vector, combinações e 40 MFCCs adiados por solicitação do usuário.

---

### Fonte: `output/normalization_remaining_comparison.md`

# Normalizações restantes — VCTK e BRSD

CNN e CNN temporal; GPU obrigatória, lote 128, cinco folds.
VCTK: 16 kHz, 30 MFCCs, mesmas janelas de 111 quadros.
BRSD: 16 kHz, 30 MFCCs + Δ + ΔΔ, Silero, 80 locutores × 5 textos.
Z-score somente na referência BRSD zscore; outros tratamentos separados.

| Corpus | Rede | Condição | Direção | Acurácia média ± desvio |
|---|---|---|---|---:|
| vctk | cnn | cmvn_logmel | mic1->mic1 | 24.92 ± 0.30% |
| vctk | cnn | cmvn_logmel | mic1->mic2 | 22.29 ± 0.54% |
| vctk | cnn | cmvn_logmel | mic2->mic1 | 22.55 ± 0.68% |
| vctk | cnn | cmvn_logmel | mic2->mic2 | 25.41 ± 0.93% |
| vctk | temporal_cnn | cmvn_logmel | mic1->mic1 | 85.25 ± 1.09% |
| vctk | temporal_cnn | cmvn_logmel | mic1->mic2 | 76.77 ± 1.65% |
| vctk | temporal_cnn | cmvn_logmel | mic2->mic1 | 72.17 ± 2.97% |
| vctk | temporal_cnn | cmvn_logmel | mic2->mic2 | 86.50 ± 1.57% |
| brsd | cnn | zscore | audio->audio | 47.00 ± 4.97% |
| brsd | temporal_cnn | zscore | audio->audio | 58.75 ± 5.80% |
| brsd | cnn | cmn | audio->audio | 3.25 ± 2.88% |
| brsd | temporal_cnn | cmn | audio->audio | 29.75 ± 6.34% |
| brsd | cnn | cmvn | audio->audio | 2.25 ± 0.56% |
| brsd | temporal_cnn | cmvn | audio->audio | 27.50 ± 4.42% |
| brsd | cnn | rasta | audio->audio | 3.75 ± 2.17% |
| brsd | temporal_cnn | rasta | audio->audio | 29.75 ± 9.62% |
| brsd | cnn | cmvn_logmel | audio->audio | 1.25 ± 0.88% |
| brsd | temporal_cnn | cmvn_logmel | audio->audio | 8.25 ± 4.11% |

BRSD possui um aparelho por locutor; resultados não demonstram eliminação do confundimento de canal.

---

### Fonte: `output/normalization40_comparison.md`

# Normalizações com 40 MFCCs — VCTK e BRSD

CNN e CNN temporal; GPU obrigatória, lote 128, cinco folds.
VCTK: 16 kHz, 40 MFCCs, mesmas janelas de 111 quadros.
BRSD: 16 kHz, 40 MFCCs + Δ + ΔΔ, Silero, 80 locutores × 5 textos.
Z-score somente na referência BRSD zscore; outros tratamentos separados.

| Corpus | Rede | Condição | Direção | Acurácia média ± desvio |
|---|---|---|---|---:|
| vctk | cnn | zscore | mic1->mic1 | 96.67 ± 0.41% |
| vctk | cnn | zscore | mic1->mic2 | 45.32 ± 1.20% |
| vctk | cnn | zscore | mic2->mic1 | 54.21 ± 1.87% |
| vctk | cnn | zscore | mic2->mic2 | 93.33 ± 0.64% |
| vctk | temporal_cnn | zscore | mic1->mic1 | 97.62 ± 0.58% |
| vctk | temporal_cnn | zscore | mic1->mic2 | 53.16 ± 3.49% |
| vctk | temporal_cnn | zscore | mic2->mic1 | 56.44 ± 1.41% |
| vctk | temporal_cnn | zscore | mic2->mic2 | 96.71 ± 0.21% |
| vctk | cnn | cmn | mic1->mic1 | 15.42 ± 1.77% |
| vctk | cnn | cmn | mic1->mic2 | 14.15 ± 1.02% |
| vctk | cnn | cmn | mic2->mic1 | 13.51 ± 0.76% |
| vctk | cnn | cmn | mic2->mic2 | 14.46 ± 0.81% |
| vctk | temporal_cnn | cmn | mic1->mic1 | 92.28 ± 0.92% |
| vctk | temporal_cnn | cmn | mic1->mic2 | 83.09 ± 1.15% |
| vctk | temporal_cnn | cmn | mic2->mic1 | 80.73 ± 2.68% |
| vctk | temporal_cnn | cmn | mic2->mic2 | 93.09 ± 1.25% |
| vctk | cnn | cmvn | mic1->mic1 | 21.19 ± 0.96% |
| vctk | cnn | cmvn | mic1->mic2 | 19.65 ± 0.87% |
| vctk | cnn | cmvn | mic2->mic1 | 19.95 ± 0.80% |
| vctk | cnn | cmvn | mic2->mic2 | 20.97 ± 0.40% |
| vctk | temporal_cnn | cmvn | mic1->mic1 | 82.82 ± 1.19% |
| vctk | temporal_cnn | cmvn | mic1->mic2 | 71.37 ± 1.59% |
| vctk | temporal_cnn | cmvn | mic2->mic1 | 73.39 ± 2.07% |
| vctk | temporal_cnn | cmvn | mic2->mic2 | 83.50 ± 2.06% |
| vctk | cnn | rasta | — | pendente |
| vctk | temporal_cnn | rasta | — | pendente |
| vctk | cnn | cmvn_logmel | — | pendente |
| vctk | temporal_cnn | cmvn_logmel | — | pendente |
| brsd | cnn | zscore | — | pendente |
| brsd | temporal_cnn | zscore | — | pendente |
| brsd | cnn | cmn | — | pendente |
| brsd | temporal_cnn | cmn | — | pendente |
| brsd | cnn | cmvn | — | pendente |
| brsd | temporal_cnn | cmvn | — | pendente |
| brsd | cnn | rasta | — | pendente |
| brsd | temporal_cnn | rasta | — | pendente |
| brsd | cnn | cmvn_logmel | — | pendente |
| brsd | temporal_cnn | cmvn_logmel | — | pendente |

BRSD possui um aparelho por locutor; resultados não demonstram eliminação do confundimento de canal.

---

## Atualização posterior: auditoria Silero100 concluída

Concluída em 2026-09-29T01:14:55.425746+00:00. 300 trilhas. Sorteio uniforme sem reposição, semente42, universo de400 BRSD e43.873 pares VCTK do inventário completo.

## Visualização

Abra visualizacao.html: original na taxa nativa em cima; decisão Silero alinhada embaixo (+1verde, −1vermelho); troca a cada2s; filtros BRSD/mic1/mic2; pausa manual e ao escutar. index.html contém áudios separados.

## Método

Áudio completo, mono por média dos canais quando necessário, conversão16k por decimate IIR ordem8/fasezero para fatoresinteiros ou resample_poly para44,1k. Sem novo trim RMS, fades, pré-ênfase ou MFCC. VCTK da distribuição wav48_silence_trimmed já passou por recorte dos autores. Silero6.2.3 CPU; threshold0,5; neg_threshold implícito0,35; fala mínima250ms; silêncio mínimo100ms; margem30ms. Não atividade é o complemento dos intervalos detectados, não silêncio comprovado.

## Resultados

| Trilha | Amostras | Duração total (s) | Atividade (s) | Não atividade (s) | Com fala contínua ≥1,792s | Com não atividade contínua ≥1,792s |
|---|---:|---:|---:|---:|---:|---:|
| brsd/audio | 100 | 3244.92 | 2561.18 | 683.74 | 100 | 39 |
| vctk/mic1 | 100 | 344.70 | 225.63 | 119.07 | 51 | 1 |
| vctk/mic2 | 100 | 344.70 | 224.13 | 120.57 | 50 | 1 |

Concordância média entre máscaras mic1/mic2: 99.40%.

Os dois previews WAV de cada trilha concatenam intervalos para escuta. Não são janelas contínuas de treino. Para reconstruir cada trecho, usar original_16k.wav e os limites em segmentos.csv/audit.json. mascara.npz contém rótulo por amostra16k; as prévias preservam o total de amostras de cada classe. Não houve treino nesta auditoria. Os treinos existentes permaneceram ativos.

selection.json registra os arquivos sorteados; protocol.json registra o método; resumo_amostras.csv descreve300 trilhas; comparacao_pares.csv descreve100 pares; verification.json registra a validação final.
