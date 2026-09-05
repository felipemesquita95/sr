# SR Studio — aplicativo desktop

Interface Python em janela própria, implementada com PySide6 / Qt Widgets.
O aplicativo lê os artefatos locais de `runs/`.

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python ui/app.py
```

Ou use o atalho na raiz, que seleciona automaticamente o ambiente virtual:

```bash
python3 abrir_ui.py
```

O ambiente do projeto precisa ter Keras 3 e PyTorch. A inspeção de sinais não
inicializa as redes; o backend torch é carregado sob demanda para a contagem de
parâmetros e em subprocesso para treinamento.

## Navegação

- Seleção fixa no topo: trilha, locutor com busca e enunciado.
- Etapas na lateral; treino e resultados têm páginas próprias.
- Setas esquerda/direita: enunciado anterior/próximo, fora de campos de texto.
- `Alt + ↑/↓`: etapa anterior/próxima; `Ctrl + F`: buscar locutor.
- `Ctrl + R`: atualizar os artefatos e invalidar os caches.
- **Amostra completa**: sorteia uma gravação com figuras dos primeiros enunciados.
- PNGs: botões de zoom, arrastar, ajustar e ampliar em uma janela maior.
- Gráficos: barra matplotlib para zoom, deslocamento e exportação. Os MFCCs
  pareados compartilham a escala de cor.

A aplicação mantém as páginas já desenhadas ao navegar. Leitura de índices,
decodificação de imagens, montagem das partições, normalização e construção de
modelos ocorrem em segundo plano. Uma resposta de uma seleção anterior é
descartada se o usuário trocar de gravação durante o carregamento. Somente a
página visível desenha seus gráficos. A geometria da janela e a seleção são
lembradas entre aberturas com `QSettings`.

## Treinamento

O formulário respeita perfil, arquitetura, limite de partições e épocas. Usa
`--features-only` para impedir reprocessamento de áudio. A saída do subprocesso
é lida em outra thread e as últimas 800 linhas aparecem no painel. Um timer
acompanha o progresso das épocas. Navegar não interrompe o treino.

Resultados existentes exigem confirmação antes de sobrescrita. Partições não
executadas novamente permanecem em disco; confira a origem ao comparar rodadas.
A janela permite um treino de cada vez, mas não coordena processos iniciados
externamente. Ao fechar durante um treino, oferece interrompê-lo e aguarda o
encerramento sem bloquear a janela.

## Limites dos artefatos

Sem áudio bruto, duração exata e quantidade de amostras originais não podem ser
recuperadas. As etapas iniciais exibem PNGs persistidos; gráficos inexistentes
são indicados explicitamente. Metadados de processamento vêm dos perfis atuais,
não de snapshots históricos. A reconstrução dos tensores usa a divisão real do
sistema e lê apenas as formas das matrizes; estatísticas de normalização são
calculadas sob solicitação, em duas passagens pelo treino.

O documento `docs/ui.md` passou a especificar Tkinter. Esta versão usa Qt para
oferecer a navegação desktop solicitada, com a dependência `PySide6-Essentials`
declarada em `requirements-dev.txt`; o documento de especificação foi preservado.

## Testes

```bash
KERAS_BACKEND=torch QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest
```

Os testes Qt usam o backend offscreen e não exigem servidor gráfico. Cobrem
navegação, dados ausentes, descarte de respostas antigas, cache de páginas,
confirmação de sobrescrita e subprocesso interrompível, além das propriedades de
leitura e particionamento do sistema.
