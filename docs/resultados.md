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
