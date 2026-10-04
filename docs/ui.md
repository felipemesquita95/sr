# Apresentação do processamento no SR Studio

A navegação acompanha o percurso da gravação: Sinal → MFCC → Rede → Resultado.
São quatro páginas seccionadas com estado persistente e rolagens independentes.
Os dados apresentados vêm de arquivos locais; a UI não executa treino, não
reprocessa features e não preenche artefatos ausentes com números de referência.

## Conteúdo de cada passo

| Passo | Conteúdo absorvido | Parâmetros do perfil associados à seleção |
|---|---|---|
| Sinal | Corpus, forma de onda e espectro originais, VAD, filtragem, reamostragem e pré-ênfase | `source_sampling_rate`, `target_sampling_rate`, `enable_vad`, `vad_top_db`, `pre_emphasis_coef` |
| MFCC | Matriz da gravação, par entre trilhas quando houver, derivadas persistidas e assinatura de canal | `num_mfccs`, `frame_size` |
| Rede | Arquiteturas salvas, contagem de parâmetros e camadas, montagem dos tensores e divisão por partição | `num_folds`, `num_speakers`, `num_utterances`, `batch_size`, `learning_rate`, `early_stopping_patience`, campos de protocolo e limite de quadros |
| Resultado | Resultado principal, comparação, diagnósticos e ressalvas | Cada registro conserva a procedência do experimento; não depende do seletor de gravação |

A barra de trilha/locutor/enunciado aparece apenas nos primeiros três passos.
Corpus pertence a Sinal porque descreve a trilha selecionada. O manifesto relaciona
os índices da seleção ao locutor e ao enunciado originais. A cobertura é contada no
inventário de arquivos. A ausência do perfil associado ou do manifesto necessário
é informada com a localização esperada, sem substituição silenciosa.

Os parâmetros exibidos são da configuração efetiva carregada por `track_profile`.
As sobrescritas de ambiente seguem as regras do carregador do projeto. Eles são
identificados como perfil atual: não provam, isoladamente, quais valores foram
usados em uma extração ou execução antiga sem snapshot. Taxa nativa, duração e
quantidade de amostras da prévia de áudio são lidas do próprio arquivo. Os totais
de parâmetros das redes vêm de `modelo.keras`, sem construir uma rede de referência.

Tensores permite escolher a partição do perfil associado à trilha. A divisão é
reconstruída das features atuais pelo código de inspeção existente; a tela informa
isso e mostra o destino da gravação. Comprimento e normalização usam apenas treino.
O protocolo cross-microfone explicita o compartilhamento de enunciados quando
aplicável; a UI não promete isolamento de voz ou de sessão que os dados não sustentam.

## Derivadas dos MFCCs

A tela procura `delta.npy` e `delta_delta.npy` junto de `mfccs.npy`. Cada arquivo é
independente: se existir, sua matriz é lida e desenhada; se faltar, a tela informa
o caminho e que o perfil que gerou as features não persistiu aquela derivada.
Não há cálculo de delta ou delta-delta na UI nem alteração de `runs/features/`.
Disponibilizar os arquivos e atualizar os artefatos basta para exibi-los.

## Resultado e comparação

Resultado principal contém visão geral, perda/acurácia e matriz/erros, mantendo o
seletor de partição no cabeçalho. As médias são calculadas somente das partições
persistidas para os filtros escolhidos; a tabela identifica os diretórios.

Comparar experimentos usa `dados.metrics(runs/models)`. O leitor percorre
`*/*/particao*/metricas.json`, valida os campos obrigatórios e devolve uma linha por
experimento, arquitetura e partição. A interface não limita a quantidade ou os
nomes dos experimentos. Relatórios ausentes, incompletos ou ilegíveis recebem
avisos com os caminhos, sem valores substitutos.

A tabela exibe experimento, arquitetura, partição, acurácia, F1 macro, acaso,
classes, amostras de teste e procedência. A ordenação das medidas é numérica,
na escala original; o filtro textual pesquisa todas as colunas. A seleção múltipla
com Ctrl/Shift monta uma coluna por registro no contraste. Cada valor tem uma dica
com o caminho de `metricas.json`, além da linha visível de diretório. Ordenar não
rompe a associação entre medida e origem. Linhas ocultas pelo filtro não entram no
contraste, mas conservam a seleção para quando o filtro for limpo.

Há uma única `EvidencePage`, com suas seções Matrizes, Silêncio e travessia, Sessão
ou transdutor, Estrutura dos erros, Controles, Protocolos auxiliares e BrSD.
Ressalvas usa os documentos carregados junto das evidências. Os botões de diagnóstico
de canal abrem assinatura ou MFCC dentro do passo correspondente.

## Decisões de composição

- `PassoPage`, derivada de `SectionedPage`, hospeda os componentes já existentes.
  Resultado principal e Evidências conservam suas abas internas para manter as
  rolagens e evitar uma faixa única excessivamente longa de abas.
- A comparação é uma seção própria de Resultado e substitui a antiga comparação
  agregada por protocolo. Permite contrastar arquiteturas, partições e classes
  diferentes, mostrando um aviso de interpretação em vez de excluir registros.
- Tensores segue o perfil da trilha. A seleção independente de outro perfil foi
  removida para não apresentar a divisão de um protocolo como se fosse da amostra.
- Falhas de leitura de uma seção viram aviso nessa seção, permitindo consultar as
  demais evidências disponíveis no mesmo passo.

## Estado, carregamento e atalhos

`Alt + ↑/↓` e os botões Anterior/Próximo percorrem os quatro passos. `Ctrl + F` foca
o locutor. Entradas digitadas são validadas com Enter ou ao perder foco; um valor
inexistente restaura a seleção anterior. `QSettings` lembra geometria e seleção.
A roda sobre seletores não altera o exemplo escolhido. Não há sorteio, navegação
por setas entre gravações nem comandos de treino.

`Tasks` mantém leituras e cálculos em segundo plano. A janela captura seleção,
revisão e partição e descarta respostas obsoletas. Um passo que deixou de estar
visível recebe seu payload pendente e só é desenhado quando voltar a ser aberto.
A carga de Resultado compartilha métricas entre resumo e comparação e diagnósticos
entre Evidências e Ressalvas. Os detalhes de curvas têm seu próprio contador de
solicitações. `Ctrl + R` invalida caches e relê a trilha.

## Arquivos e validação

`ui/app.py` controla navegação e seleção; `ui/passos.py` compõe as páginas;
`ui/conteudo.py` define o roteiro e prepara os payloads; `ui/dados.py` lê os
artefatos; `ui/paginas.py` apresenta amostras e tensores; `ui/experimentos.py`
apresenta redes, resultados e comparação; `ui/evidencias.py` reúne os diagnósticos.
Componentes, gráficos, estilo e tarefas ficam nos módulos homônimos de `ui/`.

A suíte verifica os quatro passos, a visibilidade da barra, o alcance de todas as
evidências numa instância, a carga compartilhada, a comparação com procedência,
ordenação e filtro, os parâmetros da amostra, a ausência de perfil e o comportamento
das derivadas ausentes ou persistidas. Também preserva os testes de seleção,
rolagem, curvas, divisão, normalização e descarte de respostas obsoletas.

```bash
KERAS_BACKEND=torch QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest
```
