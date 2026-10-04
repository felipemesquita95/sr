# SR Studio — aplicativo desktop

Interface PySide6 / Qt Widgets para apresentar o processamento e inspecionar os
artefatos locais. Abra com:

```bash
.venv/bin/python ui/app.py
# Ou pelo atalho que seleciona o ambiente virtual:
python3 abrir_ui.py
```

O ambiente precisa das dependências de `requirements-dev.txt`, incluindo Keras e
PyTorch para ler os checkpoints. O aplicativo não treina nem altera os artefatos.

## Quatro passos

| Passo | Seções | Seleção da gravação |
|---|---|---|
| Sinal | Corpus, sinal original, pré-processamento | Visível |
| MFCC | Matriz e derivadas, assinatura de canal | Visível |
| Rede | Arquiteturas, montagem dos tensores | Visível |
| Resultado | Resultado principal, comparar experimentos, evidências, ressalvas | Oculta |

Use os botões Anterior/Próximo ou `Alt + ↑/↓`. As seções conservam suas próprias
rolagens ao alternar entre elas. Resultado principal mantém visão geral, curvas de
perda/acurácia e matriz/erros. Uma única instância de Evidências contém Matrizes,
Silêncio e travessia, Sessão ou transdutor, Estrutura dos erros, Controles,
Protocolos auxiliares e BrSD. Seus atalhos abrem a seção pertinente de MFCC.

Escolha trilha, locutor e enunciado no topo. `Ctrl + F` foca a busca de locutor.
Confirme a entrada com Enter ou saia do campo; índice e nome exatos são aceitos.
Uma entrada inexistente restaura a seleção anterior e apresenta um aviso. Não há
sorteio nem navegação por setas entre gravações. A geometria e a seleção são
lembradas via `QSettings`.

A roda sobre campos não troca seleções. Sobre gráficos, rola a página; use a
barra matplotlib para zoom, deslocamento e exportação. PNGs permitem ampliar,
arrastar e ajustar. Os MFCCs pareados compartilham a escala de cor.

## Parâmetros e ausências

Os passos Sinal, MFCC e Rede apresentam os campos da configuração efetiva do
perfil associado à trilha, com seu caminho. Sinal mostra taxas, VAD e pré-ênfase;
MFCC mostra quantidade de coeficientes e janela em amostras; Rede mostra protocolo
e treino, incluindo partições, locutores, enunciados, lote, taxa de aprendizado e
paciência da parada antecipada. Corpus relaciona a seleção ao manifesto, quando
existente, e informa a procedência do inventário.

O perfil atual não é um snapshot histórico. A interface explicita essa diferença
e não atribui ao arquivo de áudio medições que só constam do perfil. Se não houver
perfil associado, mostra o aviso e o diretório esperado de configuração; não
escolhe outro perfil. A partição dos tensores usa o perfil dessa mesma trilha e
informa que a divisão foi reconstruída das features atuais. Os parâmetros das
redes vêm dos checkpoints salvos.

Quando faltam PNGs do sinal original, o app procura o áudio da seleção e gera uma
prévia com taxa, duração e quantidade de amostras lidas do arquivo. Sem áudio ou
figura, mostra ausência. No pré-processamento, desativar VAD preserva a apresentação
de filtragem, reamostragem e pré-ênfase.

Delta e delta-delta só aparecem se `delta.npy` e `delta_delta.npy` existirem ao lado
de `mfccs.npy`. Na ausência, a tela mostra os caminhos esperados e explica que o
perfil que gerou as features não persistiu as derivadas. Elas nunca são calculadas
pela interface. Se forem disponibilizadas, `Ctrl + R` permite relê-las.

## Comparar experimentos

Em **Resultado → Comparar experimentos**, a tabela recebe todas as linhas de
`dados.metrics(runs/models)`: experimento, arquitetura, partição, acurácia, F1 macro,
acaso, classes, amostras de teste e diretório. Cada linha corresponde a um
`runs/models/<experimento>/<arquitetura>/particao*/metricas.json` legível e completo.
Não há lista fixa de experimentos nem preenchimento de métricas ausentes.

Clique no cabeçalho para ordenar; as medidas são numéricas na escala do arquivo.
O filtro textual pesquisa todas as colunas. Selecione duas ou mais linhas com
Ctrl/Shift para vê-las lado a lado. O contraste mantém uma coluna por registro,
com diretório visível e caminho de `metricas.json` na dica de cada valor. Filtrar
retira as linhas ocultas do contraste; limpar o filtro recupera as selecionadas.
Comparações entre protocolos ou números de classes diferentes ficam permitidas,
com aviso para interpretar essas diferenças. A comparação não agrega partições.

## Carregamento e testes

Leituras, imagens e montagem das partições ocorrem em segundo plano. Cada carga
captura a seleção e sua revisão; respostas antigas são descartadas. As páginas
permanecem em memória e reutilizam o conteúdo ao voltar à mesma seleção. Evidências
e documentos são carregados uma vez por carga do passo Resultado, que também
compartilha uma leitura de métricas entre resumo e comparação. `Ctrl + R` invalida
os caches e relê os artefatos.

```bash
KERAS_BACKEND=torch QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest
```

A especificação e as decisões de composição estão em [`docs/ui.md`](../docs/ui.md).
