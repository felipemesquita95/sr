# Interface de inspeção do pipeline — especificação

Documento da implementação. Descreve uma aplicação que torna **visível cada etapa**
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
| Framework | **PySide6-Essentials** (Qt Widgets) | janela de desktop com navegação lateral e páginas em `QStackedWidget`; qualidade visual e widgets prontos |
| Gráficos | matplotlib com `backend_qtagg`, via `FigureCanvasQTAgg` | figuras embutidas nos painéis com barra de zoom, deslocamento e exportação |
| Execução | `.venv/bin/python ui/app.py` | é uma ferramenta de inspeção, não um serviço |
| Fonte dos dados | `runs/` | nada é reprocessado do corpus (ver §2) |
| Idioma da interface | português | mesma convenção do restante do projeto |

**Custo aceito: 233 MB de dependência adicional** com PySide6-Essentials, declarada
em `requirements-dev.txt`. Tkinter não teria esse custo; a escolha por Qt foi
aceita pela qualidade visual e pelos widgets prontos. `matplotlib` já é
dependência do sistema.

> Não usar Streamlit, Dash, Gradio, Flask ou qualquer coisa que sirva página em
> navegador. A interface tem de abrir em janela própria. Esta é uma decisão de
> projeto, não uma preferência de implementação.

> **Backend do Keras.** A aplicação precisa de `KERAS_BACKEND=torch` no ambiente
> antes de qualquer import de `keras`. `ui/app.py` o define logo após importar
> `os`, com `os.environ.setdefault('KERAS_BACKEND', 'torch')`.

> **Backend do matplotlib.** `ui/figuras.py` instancia `Figure` e
> `FigureCanvasQTAgg` diretamente de `backend_qtagg`, sem usar `pyplot` nem trocar
> seu backend global. `ui/app.py` define `QT_API=pyside6` por padrão. Os módulos do
> sistema que gravam figuras continuam usando `Agg`.

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

## 4. Barra de seleção — persistente

Uma faixa fixa no topo da janela, **fora** do `QStackedWidget`, continua visível ao
trocar de página. Uma `QListWidget` lateral determina a página visível.

`MainWindow.selection` guarda uma dataclass `Selection` congelada, com trilha,
locutor, enunciado, inventário, manifesto e perfil. Os valores dos seletores ficam
em `QComboBox.currentData()`; seus sinais `currentIndexChanged` recompõem a seleção.
Durante o preenchimento dos combos, `blockSignals` evita atualizações intermediárias.
Um `QTimer` de disparo único reúne mudanças próximas em uma atualização após 75 ms.

Revisões da seleção e do inventário permitem descartar respostas antigas das
tarefas assíncronas. Na carga das etapas, **só a página visível desenha widgets e
figuras**: as demais guardam os dados recebidos até serem abertas. O acompanhamento
do treino tem timer próprio e continua atualizando mesmo com sua página oculta.
A chave de carregamento reúne seleção,
revisão e parâmetros da página; voltar a uma página sem mudanças reaproveita seu
conteúdo. `QSettings` persiste a geometria da janela, trilha, locutor e enunciado
entre aberturas.

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

A faixa mostra o caminho da seleção e um indicador de figuras disponíveis. No
rodapé lateral ficam as contagens de locutores e gravações da trilha; a tabela da
visão geral detalha enunciados e cobertura de figuras por locutor.

---

## 5. As abas

A ordem é a ordem do sinal. As abas são páginas persistentes acessadas pela lateral;
treino e resultados têm páginas próprias. Cada página abre com **uma frase** dizendo
o que aquele estágio faz e por que ele existe — é essa frase que sustenta a conversa com o
orientador, não o gráfico.

### Aba 1 — Corpus

Panorama, sem seleção. Responde "o que estamos olhando".

- Cartões: número de locutores, total de gravações e **acaso** (1/*N*). A tabela
  detalha enunciados e figuras por locutor; taxas aparecem nas etapas do sinal.
- `_resumo/duracao.png` e `_resumo/enunciados_por_locutor.png` da trilha corrente.
- Tabela do manifesto: índice → nome do corpus, com filtro por nome.
- **Aviso de exclusão**, em destaque, com os nomes lidos do manifesto. No VCTK,
  `p280` e `p315` não possuem trilha `mic2`
  (problema técnico documentado no registro do corpus, nas gravações com o
  MKH 800). A numeração é definida sobre a interseção das trilhas, e por isso são
  108 locutores e não 110. É o defeito que invalidava o protocolo cross-microfone, e
  merece estar visível.

### Aba 2 — Sinal bruto

- `sinal_original.png` — forma de onda no tempo.
- `espectro_original.png` — espectro do sinal como veio do corpus.
- Ficha: taxa de leitura declarada no perfil. Duração exata e número de amostras
  são indicados como não persistidos, sem inferi-los dos MFCCs.

### Aba 3 — Detecção de atividade vocal

- Painéis de antes e depois: `sinal_original.png` e `sinal_vad.png`.
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

- `espectro_reamostrado.png` e `espectro_preenfase.png` em painéis separados. Não há
  curvas numéricas persistidas para uma sobreposição fiel; a página orienta a
  comparação das escalas das imagens.
- Texto: filtro de primeira ordem, coeficiente 0,97, compensa a queda espectral
  natural da voz.

### Aba 6 — MFCC

A primeira aba que funciona para **qualquer** enunciado, porque `mfccs.npy` sempre
existe.

- Mapa de calor da matriz, plotado ao vivo por `ui.figuras.mfcc`, com escala de cor
  compartilhada ao comparar trilhas.
- Ficha: forma da matriz (coeficientes × quadros), duração da janela, salto.
- **Comparação entre trilhas**: mostrar a mesma gravação nas duas trilhas do mesmo
  perfil, lado a lado. É a demonstração mais direta de "mesma voz, mesma frase, mesmo
  instante, transdutor diferente" — e é a base de todo o protocolo cross-microfone.

### Aba 7 — Assinatura de canal

- As três condições do `assinaturas.npz` (`silence`, `speech`, `full`) como três
  curvas de 80 valores — média e desvio de cada coeficiente cepstral.
- Aviso: baixa energia pode conter respiração ou fala fraca; o acerto indica pistas
  residuais e não mede uma parcela causal de canal ou voz.
- Resultado já medido, exibido a partir do relatório (108 locutores, acaso 0,93%):

  | condição | dentro da trilha | atravessando o microfone |
  |---|---|---|
  | só silêncio | 85,40% | 4,33% |
  | só fala | 99,64% | 55,38% |
  | sinal completo | 99,81% | 51,44% |

  Ler de `runs/models/vctk_cross_mic/diagnostico_travessia/travessia_canal.json`, não
  fixar em código.
- A página combina as curvas da seleção com a tabela lida do diagnóstico; não
  embute o PNG de travessia.

### Aba 8 — Montagem dos tensores

O estágio que costuma ficar invisível e onde moram as garantias contra vazamento.

- **Alinhamento de comprimento**: as matrizes têm largura variável (81 a 827 quadros
  no VCTK) e a rede exige largura fixa. Mostrar a matriz da seleção antes e depois do
  `pad_or_truncate`, com o teto de `MAX_FRAMES_CAP=300` desenhado sobre o
  histograma de durações. Explicitar que o preenchimento **repete o conteúdo** em vez
  de inserir zeros, e que o comprimento comum sai do **máximo do treino**, nunca do
  conjunto todo.
- **Partição**: tabela dos grupos da validação cruzada, conforme o número de
  partições do perfil; na partição *k*, o grupo *k* vai para teste. Um aviso indica
  os destinos da seleção ou se ela não participa do perfil escolhido.
- **Normalização**: média e desvio por coeficiente, calculados sob demanda em duas
  passagens **somente sobre o treino**, sem alocar todos os tensores.
- Cartões com os tamanhos reais dos três conjuntos no protocolo escolhido.

### Aba 9 — Protocolos

A aba que explica *por que* existem quatro perfis, antes de mostrar resultados.

Cartões, um por protocolo, cada um dizendo o que mede e o que **não** controla:

| protocolo | treino | teste | o que mede |
|---|---|---|---|
| Intra-microfone | `mic1`, partições por enunciado | `mic1` | identificação em condições conhecidas, com voz e sessão associadas |
| Cross-microfone | `mic1` | `mic2` | transferência entre transdutores, com sessão e enunciados compartilhados |
| Multi-microfone | ambos, partição por enunciado | ambos | exposição a ambos os microfones, ainda com sessão compartilhada |
| Permutação | rótulos embaralhados | — | controle negativo do arcabouço, sem provar ausência de todo vazamento |

### Aba 10 — Aprendizado profundo

O trecho final da navegação tem duas páginas: arquiteturas e treino, e resultados.

**Arquiteturas.** Tabela com eixo, agregação e contagem de parâmetros obtida de
`sr.models.build_model`, sob demanda, para a forma escolhida. Referência de contagens:

| arquitetura | parâmetros | eixo | agregação |
|---|---|---|---|
| `temporal_cnn` | 114.732 | tempo | `GlobalAveragePooling1D` |
| `temporal_cnn_stats` | 147.500 | tempo | média + desvio |
| `cnn` | 369.548 | coeficientes | `Flatten` |
| `xvector` | 1.203.564 | tempo (TDNN) | estatísticas |
| `xvector_attentive` | 1.401.068 | tempo (TDNN) | estatísticas com atenção |
| `attention` | 1.647.340 | coeficientes | `Flatten` |

A nota da página lembra que o eixo cepstral ordena bases da DCT e que atenção com
viés posicional e `Flatten` não torna a rede completa invariante à ordem. A
contagem é calculada sob demanda na CPU, para a forma escolhida no formulário;
os números acima são referências, não valores fixados pela interface.

**Disparar um treino.** Formulário: perfil (`configs/*.env`), arquitetura,
número de partições, épocas. Ao confirmar, executa

```
KERAS_BACKEND=torch SR_CONFIG=<perfil> ARCHITECTURES=<arq> MAX_FOLDS=<n> EPOCHS=<épocas> \
  .venv/bin/python experiments/run_experiment.py
```

como subprocesso, com a saída em fluxo na tela. Requisitos:

- **Nunca bloquear a interface.** Leituras e cálculos demorados usam
  `QThreadPool`/`QRunnable`; um `Signal` entrega resultado ou erro ao objeto `Tasks`
  na thread da janela. O treino usa `subprocess.Popen`, guardado em `Job` para
  permitir interrupção. Uma thread leitora separada drena `stdout` e, como estado
  compartilhado, só altera `job.log` sob lock, limitado às últimas 800 linhas.
  **Só a thread da janela toca em widgets.** Um `QTimer` de 700 ms chama `poll`,
  que consulta o processo, copia o log sob lock e atualiza a tela. A leitora nunca
  recebe widgets; o encerramento com espera e eventual `kill` também ocorre fora
  da thread da janela.
- **Acompanhamento ao vivo** relendo `progresso.json` do diretório da partição e
  redesenhando perda e acurácia a cada atualização. O arquivo já é escrito a cada
  época pelo callback de treino; não é preciso instrumentar nada.
- **Aviso de sobrescrita** quando `MODELS_PATH` já contiver resultados.
- **Aviso de espaço em disco** antes de começar: cada modelo salvo ocupa alguns MB e
  a partição está com folga pequena.

**Resultados.** Para o experimento selecionado, ler `runs/models/<exp>/`:

- Tabela por arquitetura e partição, com acurácia, F1, acaso e razão sobre o acaso.
- Curvas reconstruídas de `progresso.json`, `matriz_confusao.png` e
  `acuracia_por_locutor.png`, com aviso quando faltar artefato.
- **Comparação entre protocolos**, que é o resultado do trabalho — a mesma
  arquitetura sob perfis diferentes, na mesma figura:

  | condição | `cnn` | o que mede |
  |---|---|---|
  | Intra-microfone | 97,34% ± 0,24 | voz **e** canal, indistinguíveis |
  | Só silêncio, intra | 85,40% | canal isolado |
  | Cross-microfone | 38,12% | transferência entre transdutores, com sessão e enunciados compartilhados |
  | Permutação | 0,70% ± 0,02 | controle — deve ficar no acaso |
  | Acaso | 0,93% | — |

  A comparação é montada dos `metricas.json`, filtrando arquitetura e número de
  classes, com aviso quando não houver resultados. As linhas acima são referências
  históricas; o painel mostra apenas os relatórios disponíveis. Uma única partição
  aparece com desvio não estimável, e os perfis atuais não substituem snapshots
  históricos de configuração.

---

## 6. Estrutura de arquivos implementada

```
ui/
├── __init__.py            # identifica o pacote da interface
├── app.py                 # entrada, janela, seleção, navegação e persistência com QSettings
├── dados.py               # leitura e cache de artefatos, partições e estatísticas sem widgets
├── figuras.py             # gráficos matplotlib embutidos em canvas Qt e barras de navegação
├── execucao.py            # comando, ambiente, subprocesso, log sob lock e interrupção
├── conteudo.py            # Selection, catálogo de etapas e preparação dos dados em segundo plano
├── paginas.py             # páginas do pipeline e inspeção de tensores e normalização
├── experimentos.py        # formulário de treino, acompanhamento e comparação de resultados
├── componentes.py         # cartões, tabelas, rótulos e painéis de imagem com zoom
├── estilo.py              # folha de estilo dos widgets Qt
├── tarefas.py             # QThreadPool/QRunnable e entrega de resultados por Signal
├── README.md              # instruções locais de uso
└── assets/chevron.svg     # indicador dos seletores no tema
```

Separar `dados.py` de `figuras.py` permite testar o acesso a disco sem depender de
backend gráfico — é o que torna a interface coberta por testes em vez de verificada
a olho.

### Cache

`functools.lru_cache` na leitura de `mfccs.npy`, `assinaturas.npz`, do manifesto e
dos `metricas.json`, com chave formada pelo caminho e por `(mtime_ns, tamanho)`.
Assim, um arquivo substituído pode ser relido sem invalidar todo o cache.
`progresso.json` é lido com `live=True`, **fora do cache**, pois muda durante o treino;
uma leitura incompleta retorna ausência de dados e será tentada novamente.

Inventários e referências às formas dos MFCCs também têm cache limitado. `Ctrl + R`
limpa os caches e relê a trilha. As figuras permanecem nos widgets das páginas já
carregadas, com a chave descrita na §4; não existe um cache separado de `Figure`.

### Testes

`tests/test_ui.py`, no mesmo estilo do resto da suíte: propriedades, não valores.

- Listar locutores de uma trilha **não** inclui `_resumo`.
- Um enunciado sem figuras é relatado como ausente, e não levanta exceção.
- O manifesto mapeia índice → nome em ambas as direções sem colisão.
- A montagem do comando de treino preserva as sobrescritas de ambiente.

Os testes cobrem dados, partições, normalização, montagem do comando e interrupção
do subprocesso. Também instanciam widgets com `QT_QPA_PLATFORM=offscreen`, sem
exigir `DISPLAY`, para verificar navegação, avisos de ausência, confirmação de
sobrescrita, reaproveitamento de páginas e descarte de respostas antigas.

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

**Reprodução do áudio** — depende do mesmo pré-requisito e de integrar um mecanismo
de reprodução, ainda fora desta versão.

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
