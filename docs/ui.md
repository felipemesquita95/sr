# Interface de inspeção do pipeline — especificação

Documento de implementação. Descreve uma aplicação que torna **visível cada etapa**
do sistema, do áudio bruto até a rede treinada, para que o processo possa ser
percorrido e discutido em vez de descrito.

Não é uma ferramenta de análise nova: tudo que ela mostra já é produzido pelo
sistema. O que ela acrescenta é navegação — escolher um locutor e uma amostra e ver
o que aconteceu com aquele sinal em cada estágio, na ordem em que aconteceu.

Destino: diretório `ui/` na raiz do repositório.

---

## 1. Decisões já tomadas

| decisão | escolha | motivo |
|---|---|---|
| Linguagem | Python 3.13 | mesma do sistema; reaproveita `src/sr` diretamente |
| Framework | **Streamlit** | abas nativas, estado de sessão, zero front-end; `st.pyplot` aceita as figuras matplotlib que o projeto já sabe gerar |
| Execução | local, `streamlit run ui/app.py` | é uma ferramenta de inspeção, não um serviço |
| Fonte dos dados | `runs/` | nada é reprocessado do corpus (ver §2) |
| Idioma da interface | português | mesma convenção do restante do projeto |

Dependências novas, para `requirements-dev.txt`: `streamlit`. Tudo mais
(`numpy`, `librosa`, `matplotlib`, `keras`) já é dependência do sistema.

> **Backend do Keras.** A aplicação precisa de `KERAS_BACKEND=torch` no ambiente
> antes de qualquer import de `keras`. Definir em `ui/app.py`, na primeira linha
> executável, com `os.environ.setdefault('KERAS_BACKEND', 'torch')`.

---

## 2. A restrição que molda tudo

**O áudio bruto não existe mais em disco.** O VCTK é ingerido apagando cada arquivo
logo após convertê-lo em features — o corpus não cabe em disco junto do próprio zip
de 11 GB. O áudio do BrSD também já não está presente.

A consequência é direta e a interface precisa ser honesta quanto a ela:

| estágio | disponível para | como |
|---|---|---|
| Forma de onda, espectros, VAD | **apenas enunciados 1–4** de cada locutor | PNGs gravados na ingestão |
| MFCC | **todos** os 21.523 enunciados por trilha | `mfccs.npy`, replotável ao vivo |
| Assinatura de canal | **todos** | `assinaturas.npz`, replotável ao vivo |
| Tensores, treino, avaliação | todos | recalculável / `runs/models/` |

A cobertura de 1–4 é uniforme: **os 108 locutores têm figuras dos enunciados 1 a 4
em todas as quatro trilhas**. Não é uma amostra enviesada de locutores; é uma
amostra dos primeiros enunciados de cada um.

**Regra de projeto:** a interface nunca inventa um estágio que não pode mostrar.
Quando o enunciado escolhido não tem figuras, o estágio aparece com um aviso
explicando o motivo e indicando os enunciados que têm — e não com um gráfico vazio,
um placeholder ou uma reconstrução aproximada. A ausência de dado é informação sobre
o experimento e deve ser lida como tal.

---

## 3. Layout dos dados em disco

O implementador não deve inferir estes caminhos: eles são o contrato.

```
runs/
├── features/
│   ├── vctk_manifesto.json          # índice → nome do corpus
│   ├── vctk_mic1/                   # trilha DPA 4035, com VAD
│   ├── vctk_mic2/                   # trilha Sennheiser MKH 800, com VAD
│   ├── vctknovad_mic1/              # idem, sem VAD
│   ├── vctknovad_mic2/
│   └── brsd/
│       └── <locutor>/<enunciado>/
│           ├── mfccs.npy            # (num_mfccs, num_quadros) float32  — sempre
│           ├── assinaturas.npz      # silence|speech|full, (80,) float32 — sempre
│           ├── sinal_original.png   # ─┐
│           ├── sinal_vad.png        #  │ só quando o enunciado ≤ 4
│           ├── espectro_original.png#  │ (ausente nas demais gravações)
│           ├── espectro_filtrado.png#  │
│           ├── espectro_reamostrado.png
│           ├── espectro_preenfase.png
│           └── mfccs.png            # ─┘
└── models/
    └── <experimento>/<arquitetura>/particao<N>/
        ├── metricas.json
        ├── modelo.keras
        ├── progresso.json           # escrito a cada época — permite acompanhar ao vivo
        ├── progresso.png
        ├── curvas_treino.png
        ├── matriz_confusao.png
        └── acuracia_por_locutor.png
```

`_resumo/` existe dentro de cada trilha (`duracao.png`,
`enunciados_por_locutor.png`) e **não é um locutor** — precisa ser filtrado ao
listar os diretórios.

### Esquemas

`runs/features/vctk_manifesto.json`:

```jsonc
{
  "zip": "...",
  "microfones": ["mic1", "mic2"],
  "num_locutores": 108,
  "locutores":  { "1": "p225", "2": "p226", ... },   // índice → nome do corpus
  "enunciados": { "p225": { "1": "001", "2": "002", ... }, ... },
  "excluidos":  { "locutores_sem_todas_as_trilhas": ["p280", "p315"],
                  "enunciados_parciais_por_locutor": {} }
}
```

`metricas.json`: `acuracia`, `precisao_macro`, `revocacao_macro`, `f1_macro`,
`acaso`, `vezes_o_acaso`, `num_classes`, `num_amostras_teste`.

`progresso.json`: `arquitetura`, `particao`, `epoca`, e `historico` com as listas
`loss`, `val_loss`, `accuracy`, `val_accuracy` — uma entrada por época concluída.

---

## 4. Barra lateral — seleção persistente

Fica visível em todas as abas e define o que elas mostram. Estado em
`st.session_state`, para que trocar de aba **não** perca a seleção.

1. **Corpus/trilha** — `vctk_mic1`, `vctk_mic2`, `vctknovad_mic1`,
   `vctknovad_mic2`, `brsd`. Lida do que existe em `runs/features/`, não fixa em
   código.
2. **Locutor** — 1 a *N*. Exibir sempre o nome do corpus ao lado do índice
   (`12 — p236`), lido do manifesto. O índice é interno; o nome é o que o orientador
   reconhece.
3. **Enunciado** — 1 a *N*. Marcar quais têm figuras completas, por exemplo
   `1 ✚ figuras` contra `57 — só MFCC`.
4. **Atalho "amostra completa"** — botão que sorteia um locutor aleatório com
   enunciado em 1–4, garantindo uma seleção que preenche todas as abas. É o botão a
   usar durante uma apresentação.

Rodapé da barra lateral: contagem de locutores, de enunciados e de gravações com
figuras na trilha corrente. Orienta sem precisar navegar.

---

## 5. As abas

A ordem é a ordem do sinal. Cada aba abre com **uma frase** dizendo o que aquele
estágio faz e por que ele existe — é essa frase que sustenta a conversa com o
orientador, não o gráfico.

### Aba 1 — Corpus

Panorama, sem seleção. Responde "o que estamos olhando".

- Cartões: número de locutores, enunciados por locutor, total de gravações, taxa de
  amostragem de origem e alvo, e o **acaso** (1/*N*).
- `_resumo/duracao.png` e `_resumo/enunciados_por_locutor.png` da trilha corrente.
- Tabela do manifesto: índice → nome do corpus, com filtro por nome.
- **Aviso de exclusão**, em destaque: `p280` e `p315` não possuem trilha `mic2`
  (problema técnico documentado no registro do corpus, nas gravações com o
  MKH 800). A numeração é definida sobre a interseção das trilhas, e por isso são
  108 locutores e não 110. É o defeito que invalidava o protocolo cross-microfone, e
  merece estar visível.

### Aba 2 — Sinal bruto

- `sinal_original.png` — forma de onda no tempo.
- `espectro_original.png` — espectro do sinal como veio do corpus.
- Ficha: taxa de origem (48 kHz), duração em segundos, número de amostras.

### Aba 3 — Detecção de atividade vocal

- Lado a lado: `sinal_original.png` e `sinal_vad.png`.
- Texto do estágio: o VAD corta trechos abaixo de `VAD_TOP_DB` dB do pico.
- **Ponto de discussão a deixar explícito na aba:** o VAD é aplicado a cada trilha
  isoladamente, com limiar relativo ao pico *daquela* trilha. Como os microfones
  diferem em nível e ruído de fundo, os cortes caem em pontos distintos e as trilhas
  deixam de ser quadro a quadro correspondentes — medido em
  `experiments/verify_alignment.py`: apenas 212 de 6.200 gravações mantêm duração
  idêntica entre as trilhas, com divergência mediana de 11,5%.
- Nas trilhas `vctknovad_*` a aba informa que o estágio foi desativado no perfil, em
  vez de sumir. O contraste entre as trilhas é conteúdo, não um caso de erro.

### Aba 4 — Filtragem e reamostragem

- `espectro_filtrado.png` — após o filtro anti-aliasing.
- `espectro_reamostrado.png` — após decimação para 16 kHz.
- Texto: filtrar antes de decimar evita que conteúdo acima de Nyquist rebata sobre a
  banda útil. É a ordem que importa, e o par de figuras a torna visível.

### Aba 5 — Pré-ênfase

- `espectro_preenfase.png`, preferencialmente sobreposto ao reamostrado para que o
  ganho em alta frequência apareça como diferença e não como duas figuras a comparar
  de memória.
- Texto: filtro de primeira ordem, coeficiente 0,97, compensa a queda espectral
  natural da voz.

### Aba 6 — MFCC

A primeira aba que funciona para **qualquer** enunciado, porque `mfccs.npy` sempre
existe.

- Mapa de calor da matriz, plotado ao vivo com
  `sr.preprocessing.visualization.plot_mfccs`.
- Ficha: forma da matriz (coeficientes × quadros), duração da janela, salto.
- **Comparação entre trilhas**: mostrar a mesma gravação nas duas trilhas do mesmo
  perfil, lado a lado. É a demonstração mais direta de "mesma voz, mesma frase, mesmo
  instante, transdutor diferente" — e é a base de todo o protocolo cross-microfone.

### Aba 7 — Assinatura de canal

- As três condições do `assinaturas.npz` (`silence`, `speech`, `full`) como três
  curvas de 80 valores — média e desvio de cada coeficiente cepstral.
- Texto: se o locutor pode ser identificado a partir da condição `silence`, a pista
  não está na voz.
- Resultado já medido, exibido como tabela fixa (108 locutores, acaso 0,93%):

  | condição | dentro da trilha | atravessando o microfone |
  |---|---|---|
  | só silêncio | 85,40% | 4,33% |
  | só fala | 99,64% | 55,38% |
  | sinal completo | 99,81% | 51,44% |

  Ler de `runs/models/vctk_cross_mic/diagnostico_travessia/travessia_canal.json`, não
  fixar em código.
- `runs/models/<exp>/diagnostico_travessia/travessia_canal.png` quando existir.

### Aba 8 — Montagem dos tensores

O estágio que costuma ficar invisível e onde moram as garantias contra vazamento.

- **Alinhamento de comprimento**: as matrizes têm largura variável (81 a 827 quadros
  no VCTK) e a rede exige largura fixa. Mostrar a matriz da seleção antes e depois do
  `pad_or_truncate`, com o teto de `MAX_FRAMES_CAP=300` desenhado sobre o
  histograma de durações. Explicitar que o preenchimento **repete o conteúdo** em vez
  de inserir zeros, e que o comprimento comum sai do **máximo do treino**, nunca do
  conjunto todo.
- **Partição**: diagrama da validação cruzada — os enunciados de cada locutor
  distribuídos em 5 grupos por posição; na partição *k*, o grupo *k* vai para teste.
  Marcar onde cai o enunciado selecionado.
- **Normalização**: média e desvio por coeficiente, calculados **somente sobre o
  treino**.
- Cartões com os tamanhos reais dos três conjuntos no protocolo escolhido.

### Aba 9 — Protocolos

A aba que explica *por que* existem quatro perfis, antes de mostrar resultados.

Cartões, um por protocolo, cada um dizendo o que mede e o que **não** controla:

| protocolo | treino | teste | o que isola |
|---|---|---|---|
| Intra-microfone | `mic1`, partições por enunciado | `mic1` | nada — voz e canal juntos |
| Cross-microfone | `mic1` | `mic2` | voz, do modelo de microfone |
| Multi-microfone | ambos, partição por enunciado | ambos | descorrelaciona canal e rótulo |
| Permutação | rótulos embaralhados | — | o próprio arcabouço |

### Aba 10 — Aprendizado profundo

A aba final, e a que o orientador vai querer manipular.

**Arquiteturas.** Cartão por rede, com o eixo da convolução, a agregação e a
contagem de parâmetros (obter de `sr.models.build_model`, ao vivo, com a forma de
entrada corrente — não fixar):

| arquitetura | parâmetros | eixo | agregação |
|---|---|---|---|
| `temporal_cnn` | 114.732 | tempo | `GlobalAveragePooling1D` |
| `temporal_cnn_stats` | 147.500 | tempo | média + desvio |
| `cnn` | 369.548 | coeficientes | `Flatten` |
| `xvector` | 1.203.564 | tempo (TDNN) | estatísticas |
| `xvector_attentive` | 1.401.068 | tempo (TDNN) | estatísticas com atenção |
| `attention` | 1.647.340 | coeficientes | `Flatten` |

Incluir a nota de projeto: o eixo cepstral **não tem estrutura de vizinhança** —
coeficientes adjacentes são projeções de bases distintas da DCT. Convolução
pressupõe localidade; atenção é equivariante a permutação e não faz essa suposição.
A escolha do eixo é objeto de estudo, não detalhe de implementação.

**Disparar um treino.** Formulário: perfil (`configs/*.env`), arquitetura,
número de partições, épocas. Ao confirmar, executa

```
KERAS_BACKEND=torch SR_CONFIG=<perfil> ARCHITECTURES=<arq> MAX_FOLDS=<n> \
  .venv/bin/python experiments/run_experiment.py
```

como subprocesso, com a saída em fluxo na tela. Requisitos:

- **Nunca bloquear a interface.** `subprocess.Popen`, leitura incremental do
  `stdout`, e o PID guardado em `st.session_state` para permitir interromper.
- **Acompanhamento ao vivo** relendo `progresso.json` do diretório da partição e
  redesenhando perda e acurácia a cada atualização. O arquivo já é escrito a cada
  época pelo callback de treino; não é preciso instrumentar nada.
- **Aviso de sobrescrita** quando `MODELS_PATH` já contiver resultados.
- **Aviso de espaço em disco** antes de começar: cada modelo salvo ocupa alguns MB e
  a partição está com folga pequena.

**Resultados.** Para o experimento selecionado, ler `runs/models/<exp>/`:

- Tabela por arquitetura e partição, com acurácia, F1, acaso e razão sobre o acaso.
- `curvas_treino.png`, `matriz_confusao.png`, `acuracia_por_locutor.png`.
- **Comparação entre protocolos**, que é o resultado do trabalho — a mesma
  arquitetura sob perfis diferentes, na mesma figura:

  | condição | `cnn` | o que mede |
  |---|---|---|
  | Intra-microfone | 97,34% ± 0,24 | voz **e** canal, indistinguíveis |
  | Só silêncio, intra | 85,40% | canal isolado |
  | Cross-microfone | 38,12% | voz, isolada do modelo de microfone |
  | Permutação | 0,70% ± 0,02 | controle — deve ficar no acaso |
  | Acaso | 0,93% | — |

  Montar da leitura dos `metricas.json`, com fallback explícito quando um
  experimento ainda não tiver sido executado.

---

## 6. Estrutura de arquivos sugerida

```
ui/
├── app.py                 # entrada: configura a página, monta a barra lateral, despacha as abas
├── dados.py               # acesso a runs/: listar trilhas, locutores, enunciados; ler manifesto,
│                          #   mfccs, assinaturas, métricas, progresso. Sem matplotlib aqui.
├── figuras.py             # replot ao vivo (MFCC, assinaturas, curvas) usando sr.preprocessing.visualization
├── execucao.py            # subprocesso de treino, streaming de log, leitura de progresso.json
└── abas/
    ├── corpus.py
    ├── sinal.py
    ├── vad.py
    ├── filtragem.py
    ├── preenfase.py
    ├── mfcc.py
    ├── assinatura.py
    ├── tensores.py
    ├── protocolos.py
    └── aprendizado.py
```

Separar `dados.py` de `figuras.py` permite testar o acesso a disco sem depender de
backend gráfico — é o que torna a interface coberta por testes em vez de verificada
a olho.

### Cache

`@st.cache_data` na leitura de `mfccs.npy`, `assinaturas.npz`, do manifesto e dos
`metricas.json`. São dezenas de milhares de arquivos pequenos: sem cache, cada
troca de aba relê disco. **Não** cachear `progresso.json` — é justamente o arquivo
que muda durante o treino.

### Testes

`tests/test_ui.py`, no mesmo estilo do resto da suíte: propriedades, não valores.

- Listar locutores de uma trilha **não** inclui `_resumo`.
- Um enunciado sem figuras é relatado como ausente, e não levanta exceção.
- O manifesto mapeia índice → nome em ambas as direções sem colisão.
- A montagem do comando de treino preserva as sobrescritas de ambiente.

---

## 7. Fora do escopo desta versão

Registrado para implementação posterior. Nada aqui bloqueia a versão acima.

**Recomputação ao vivo a partir do áudio.** É o maior salto de qualidade possível:
com o áudio em disco, as abas 2 a 6 deixam de ser figuras gravadas e passam a
recalcular o estágio, com controles para `VAD_TOP_DB`, `PRE_EMPHASIS_COEF`,
`TARGET_SAMPLING_RATE`, `NUM_MFCCS` e `FRAME_SIZE` — o orientador mexe no limiar e
vê o corte mudar. Pré-requisitos: rebaixar o corpus (~16 min) e ter espaço, hoje
inexistente (2,8 GB livres). Alternativa barata: manter uma **vitrine** de algumas
dezenas de WAVs (poucas centenas de MB) só para essa finalidade, em vez do corpus
inteiro.

**Reprodução do áudio** (`st.audio`) — depende do mesmo pré-requisito.

**Metadados dos locutores.** O `speaker-info.txt` do VCTK traz idade, gênero,
sotaque e região. Foi apagado com o corpus, mas são poucos KB e pode ser obtido
isoladamente. Permitiria colorir a matriz de confusão por gênero e sotaque e
responder uma pergunta que nenhuma acurácia agregada responde: se os 38% do
cross-microfone vêm de separar gênero e sotaque, são traço vocal grosso; se os erros
se espalham, é identidade.

**Exportação de relatório** — reunir as figuras da seleção corrente em um PDF.

**Comparação lado a lado de duas seleções** — dois locutores, ou o mesmo locutor em
duas trilhas, com as abas espelhadas.

**Aba de ingestão** — acompanhar `experiments/ingest_vctk.py`. Só faz sentido se o
corpus voltar ao disco.

---

## 8. Estado dos experimentos que a interface consome

Nem todo perfil tem resultado gravado. A interface deve degradar com elegância, e o
implementador precisa saber o que esperar encontrar:

| experimento | `MODELS_PATH` | estado |
|---|---|---|
| BrSD, 3 arquiteturas × 5 partições | `runs/models/brsd` | pronto |
| VCTK intra-mic, 5 partições | `runs/models/vctk_mic1` | pronto |
| VCTK cross-mic, com VAD | `runs/models/vctk_cross_mic` | pronto |
| VCTK cross-mic, sem VAD | `runs/models/vctk_cross_mic_novad` | pronto |
| Travessia de canal | `runs/models/vctk_cross_mic/diagnostico_travessia` | pronto |
| Controle de permutação | `runs/models/controle_permutacao_*` | pronto |
| Verificação de alinhamento | `runs/models/verificacao_alinhamento` | pronto |
| VCTK multi-mic | `runs/models/vctk_multi_mic` | **pendente** |
| VCTK sem VAD, intra-mic | `runs/models/vctk_novad` | **pendente** |
| Busca de arquiteturas (6 redes) | — | **pendente** |

Ver `EXPERIMENTOS.md` para a sequência e o que cada passo decide.
