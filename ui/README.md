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

Treino, resultados e **Evidências diagnósticas** são divididos em seções com rolagens independentes. Em
**Resultados → Perda e acurácia**, a partição é escolhida no cabeçalho fixo.
**Arquiteturas e treino** separa acompanhamento, configuração e inspeção das redes.
O acompanhamento atualiza as linhas do mesmo gráfico a cada época, preservando
o painel e a posição de leitura.

A roda do mouse sobre campos não altera seleções ou números. Sobre os gráficos,
ela rola a página; para ampliar uma região, use a ferramenta de zoom na barra do
gráfico. Tabelas e log devolvem a rolagem à página ao alcançar suas extremidades.

- Seleção fixa no topo: trilha, locutor com busca e enunciado.
- Para selecionar digitando, confirme com Enter ou saia do campo. O locutor
  aceita índice (`4`, `004`) ou nome exato (`p228`); o enunciado aceita seu número.
  Entradas inexistentes restauram a seleção anterior com um aviso, sem deixar
  um número no campo enquanto outro áudio é exibido.
- Nas etapas de sinal, **Figuras salvas deste locutor** lista os enunciados
  realmente presentes para aquela etapa, sem trocar o locutor escolhido.
  Ao trocar o locutor, se o enunciado anterior não tiver sinal disponível,
  seleciona um exemplo com figuras do novo locutor; o número aparece no topo.
- Etapas na lateral; treino e resultados têm páginas próprias.
- **Roteiro da defesa** abre a sequência argumentativa de `docs/resultados.md` e
  leva diretamente a **Evidências diagnósticas**. As abas desta página cobrem
  matrizes de transferência, silêncio, recuperação por transformação afim,
  estrutura dos erros, controles, protocolos auxiliares e BrSD. Cada gráfico
  é lido dos JSONs em `runs/models/`, informa seu caminho de procedência e
  mantém a ressalva documental junto da evidência. Se um relatório faltar, a
  aba exibe o caminho esperado e pede sua geração; ela não mostra valores
  substitutos nem deriva medições da documentação.
- Setas esquerda/direita: gravação anterior/próxima, inclusive após escolher nos
  seletores. Ao chegar ao último enunciado, seguem para o próximo locutor.
  Durante uma busca digitada, as setas editam o texto; `Alt + ←/→` troca a gravação.
- `Alt + ↑/↓`: etapa anterior/próxima; `Ctrl + F`: buscar locutor.
- `Ctrl + R`: atualizar os artefatos e invalidar os caches.
- **Amostra completa**: sorteia outra gravação com figuras, sem repetir a seleção atual.
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

Em **Sinal bruto**, quando faltam PNGs, o app procura o áudio original da seleção
e prepara a forma de onda e o espectro em segundo plano, sem alterar artefatos.
A prévia usa a taxa nativa do arquivo, informa duração e amostras e resume
extremos/picos para limitar o desenho. No BrSD, a numeração vem do índice do
projeto; no VCTK, exige a correspondência exata do manifesto, inclusive microfone.
Isso permite explorar outros locutores do BrSD, cujos PNGs só cobrem o locutor 1.
Sem o áudio nem o PNG, exibe um aviso e um botão para uma gravação com figuras;
não reconstrói sinais a partir de MFCCs. As demais etapas iniciais usam PNGs salvos.
Metadados de processamento vêm dos perfis atuais,
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
