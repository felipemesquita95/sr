# Revisão para a nova execução no VCTK

## Situação dos dados

Em 24/09/2026, o volume `/media/lsmsqt/HDD` não estava montado. Os arquivos FLAC
originais do VCTK não estavam disponíveis nesta sessão. Por isso, nenhuma
gravação real foi reprocessada ou usada para obter uma nova acurácia.

O processador preliminar está em `experiments/process_vctk_pair_corrected.py`.
Ele faz corte suave nas bordas, limites compartilhados para dois canais já
alinhados, decimação Chebyshev I, pré-ênfase, MFCC Hamming 32/16 ms e deltas.
Um sinal sintético de 3 s produziu limites 0,572–2,420 s e 114 quadros em cada
microfone. É apenas uma verificação funcional, sem validade como resultado de
reconhecimento. O processador recusa pares de durações diferentes: o alinhamento
dos FLACs reais ainda precisa ser medido antes da execução do corpus inteiro.

## Delta e delta-delta

O delta aproxima a variação temporal de cada coeficiente MFCC; o delta-delta
aproxima a variação do delta. Eles mantêm o mesmo número de quadros e elevam a
dimensão de `C` para `3C` quando concatenados com os MFCCs estáticos. No código
preliminar, cada derivada usa regressão local de nove quadros. Com hop de 16 ms,
a janela de nove quadros abrange aproximadamente 128 ms entre o primeiro e o
último centro. Calcular essas características apenas após o corte do áudio.

O x-vector original de Snyder et al. (2018) usa 24 filtros de banco mel, sem
descrever concatenação explícita de delta/delta-delta. O estudo de Gu et al.
(2020) usa 30 dimensões incluindo delta e delta-delta em um x-vector adaptativo.
Chauhan et al. (2023) combinam várias famílias de características e derivadas,
com melhora relatada em seu protocolo; isso não isola o efeito de delta no nosso
VCTK. Portanto, delta é uma hipótese de teste, não um ganho presumido.

Proposta: manter a mesma rede temporal, pares de gravação e folds; comparar
`MFCC` com `MFCC+delta+delta-delta` apenas nos dados de treino/validação; escolher
uma entrada antes de avaliar os testes intra e cross-mic. Ajustar o z-score de
cada dimensão exclusivamente com o treino. Não tratar quadros da mesma gravação
como exemplos independentes em partições diferentes. Evitar usar o teste para
selecionar entre os dois conjuntos de características.

## Trabalhos e números publicados

| Trabalho | Dados e entrada | Avaliação publicada | Limite da comparação |
|---|---|---|---|
| Snyder et al., ICASSP 2018, x-vector | SWBD/SRE, 24 filterbanks, SAD, rede temporal com pooling de média e desvio | SITW Core EER de 9,40% no sistema inicial e 4,16% após treinamento com aumento e VoxCeleb; SRE16 EER de 8,00% e 5,71% nas mesmas condições | Verificação por pares, corpus e volume de treino diferentes; EER não é acurácia de identificação fechada. |
| Gu et al., Interspeech 2020, x-vector adaptativo | VoxCeleb para treino, SITW/VOiCES para teste, 30 características com deltas | x-vector base: EER 3,28% no SITW eval e 8,34% no VOiCES eval; também minDCF e actDCF | Dados e tarefa de verificação diferentes; artigo não demonstra ganho isolado dos deltas. |
| Chauhan et al., SN Computer Science 2023 | VCTK, 109 locutores, somente 5 gravações por locutor: 4 treino e 1 teste; fusão de MFCC, LPC, PLP, energia e derivados | Até 100% de acurácia de identificação e EER 0% no subconjunto de 545 gravações | Não é rede x-vector, nem usa nossos 5 folds, seleção ou teste cross-mic; valor não serve como meta direta. |
| Khan et al., Computers, Materials & Continua 2023 | VCTK e outros corpora; fusão de 7 famílias incluindo 40 MFCC, delta e delta-delta; Transformer | 97,4% de acurácia média relatada para VCTK, divisão 80/20; artigo informa também precisão, recall e F1 | Sem ablação que isole deltas; protocolo e entradas diferentes; sem avaliação cross-mic equivalente. |
| Vaessen e van Leeuwen, Interspeech 2022 | VoxCeleb2, subconjuntos de 50 mil arquivos; compara x-vector, ECAPA e wav2vec2 | EER de verificação; por exemplo, no subconjunto `tiny-few-sessions`, ECAPA 19,52% e x-vector 24,00% após 400 mil passos | Mostra dificuldade de treino com poucas sessões; não prevê desempenho no VCTK. |

## Como avaliar o nosso experimento

É identificação fechada de locutores conhecidos. A unidade de teste deve ser a
**gravação**, com uma previsão por gravação. Relatar acurácia top-1, F1 macro,
matriz de confusão e resultados por locutor nos cinco folds; média e dispersão
entre folds. Manter treino/validação/teste 60/20/20 por gravação e locutor.
Separar os quatro caminhos mic1→mic1, mic2→mic2, mic1→mic2 e mic2→mic1.
Se forem calculados scores de verificação no futuro, apresentar EER e minDCF
separadamente com protocolo explícito de pares positivos/negativos. Não converter
EER em acurácia.

O teste cruzado usa a mesma leitura gravada nos dois microfones, em partições
disjuntas por leitura. Ele mede mudança de transdutor nesse corpus, mas ainda
pode carregar características da sessão e do texto. Não descrevê-lo como prova
isolada de identidade vocal independente de ambiente.

## Fontes primárias

- Snyder et al. (2018), *X-Vectors: Robust DNN Embeddings for Speaker Recognition*:
  https://www.danielpovey.com/files/2018_icassp_xvectors.pdf
- Gu et al. (2020), *An Adaptive X-Vector Model for Text-Independent Speaker Verification*:
  https://www.isca-archive.org/interspeech_2020/gu20_interspeech.pdf
- Chauhan et al. (2023), *Text-Independent Speaker Recognition System Using Feature-Level Fusion for Audio Databases of Various Sizes*:
  https://link.springer.com/article/10.1007/s42979-023-02056-w
- Khan et al. (2023), *An Efficient Text-Independent Speaker Identification Using Feature Fusion and Transformer Model*:
  https://www.techscience.com/cmc/v75n2/52098/html
- Vaessen e van Leeuwen (2022), *Training speaker recognition systems with limited data*:
  https://www.isca-archive.org/interspeech_2022/vaessen22_interspeech.pdf
- NIST (2021), *Speaker Recognition Evaluation Plan*:
  https://www.nist.gov/publications/nist-2021-speaker-recognition-evaluation-plan
