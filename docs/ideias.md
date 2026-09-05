# Próximos experimentos, por prioridade

Análise de 4 de setembro de 2026. As propostas abaixo procuram alterar a
interpretação dos resultados ou a escolha do protocolo. Os números citados foram
lidos dos `resumo.json`, `resumo.txt` e `metricas.json` em `runs/models/`.
Nenhum treino ou experimento proposto foi executado nesta análise.

Há duas diferenças entre o estado informado e o disco consultado. Em
`vctk_cross_mic_inverso`, já existem resultados completos da CNN (**43,31%**, F1
macro **0,4000**) e da CNN temporal (**49,43%**, F1 macro **0,4387**), contra
38,12% e 36,65% no sentido direto. O segundo apareceu durante esta consulta;
estes são os resultados inversos completos considerados aqui. O áudio bruto do
**BrSD está presente** em `/home/lsmsqt/datasets/brsd/utterances`: 400 WAVs, cerca
de 2,64 GB, todos com cabeçalhos RIFF/WAVE legíveis. A soma das durações nos
cabeçalhos é de 223 minutos. Isso verifica disponibilidade, não integridade de
todo o conteúdo. O VCTK bruto e seu arquivo ZIP estão ausentes.

## 1. Medir quem é reconhecido e a discriminação dentro dos grupos de metadados

**O que se mede.** Reconstituir as predições das três arquiteturas no cross com
enunciados disjuntos. Reportar a distribuição de revocação por locutor, quantos
locutores nunca são acertados, a concentração dos acertos e quais classes recebem
os erros. Cruzar essa distribuição com os campos de gênero e sotaque do corpus,
preservando a terminologia e as categorias da fonte.

A pista dos metadados merece investigação, mas a hipótese precisa de um controle:
errar entre pessoas do mesmo grupo também é compatível com um identificador que
discrimina indivíduos e tem dificuldade com vozes semelhantes. Gênero e sotaque
não são alternativas mutuamente exclusivas à identidade.

Usar dois referenciais, definidos antes de olhar as matrizes:

- Um classificador que recebe o grupo verdadeiro, mas nenhuma informação que
  distinga pessoas dentro dele. Com peso igual por locutor, seu teto esperado de
  acurácia é **G/S**, onde G é o número de grupos ocupados e S o de locutores com
  metadados. Para dois grupos e 108 locutores, seria 1,85%, não 38%. Para o
  cruzamento gênero × sotaque, é preciso contar G; não presumir o teto. Com pesos
  por gravação, o teto correspondente é a soma do maior suporte de locutor em cada
  grupo dividida pelo suporte total.
- A identificação restrita aos candidatos do mesmo grupo do alvo, usando os
  escores já produzidos pela rede. Comparar a revocação nesse conjunto com
  `1 / número de candidatos`. Excluir grupos unitários dessa comparação e
  informar sua cobertura. Essa restrição usa metadados verdadeiros como um
  diagnóstico; não é uma nova acurácia operacional do sistema.

Na matriz de erros, comparar a proporção de confusões dentro de cada grupo com um
referencial que preserve os tamanhos dos grupos e a frequência das classes
preditas. Contar apenas erros dentro do grupo favoreceria os grupos maiores.

**Por que agora.** A CNN disjunta tem 38,69% de acurácia e F1 macro de 0,3432.
Esses agregados não dizem se quase todos os locutores são parcialmente reconhecidos
ou se poucos concentram os acertos. A diferença muda o que o protocolo deve
reportar, mesmo sem alterar a média.

**O que os resultados decidem.**

- Se a discriminação dentro dos grupos superar seu acaso e os acertos se
  distribuírem entre muitos locutores, a explicação baseada apenas nesses
  metadados fica insuficiente. Há informação discriminativa individual adicional;
  sua origem acústica continua sem identificação causal.
- Se o resultado ficar próximo do referencial de grupos e a identificação
  intragrupo não superar seu acaso, o número agregado não sustenta discriminação
  individual além dessas categorias. O protocolo deve explicitar essa limitação.
- Se o desempenho depender de poucos locutores ou grupos, a conclusão passa a
  ser de transferência heterogênea. Revocação por locutor, cobertura e resultados
  por grupo tornam-se medidas necessárias, em vez de uma única média cross.

**Custo.** Zero treinos. Estimar **3–15 minutos** para reconstrução dos conjuntos
e inferência dos três modelos, mais **1–5 minutos de CPU** para as estatísticas.
A preparação do instrumento deve consumir aproximadamente meio dia. Os tempos
de inferência e CPU são estimativas, não medições nesta máquina.

**O que precisa existir.** Os checkpoints e os MFCCs existem. O manifesto
`runs/features/vctk_manifesto.json` preserva tanto a relação índice → locutor
original quanto a relação índice → enunciado original. O `speaker-info.txt` não
foi encontrado nos diretórios do projeto e dos datasets: é necessário recuperar
esse arquivo da versão correspondente, sem precisar recuperar o áudio. Conferir
a cobertura, inclusive do locutor `s5`; não preencher categorias ausentes por
suposição nem transformar ausência em um grupo artificialmente informativo.

Os JSONs atuais contêm apenas métricas agregadas. A matriz e a revocação por
locutor foram salvas como PNG; alvos, predições e probabilidades não foram
persistidos em formato numérico. Será necessário um instrumento de inferência
que reconstitua a divisão e a normalização do treino. Antes da análise, deve
reproduzir as métricas salvas, dentro da tolerância numérica. Uma divergência
interrompe a interpretação das novas predições; não autoriza atribuí-las ao
experimento antigo.

## 2. Estimar a perda de transferência com treino e teste comparáveis

**O que se mede.** Fixar, por locutor, os mesmos enunciados de treino, validação e
teste nas duas trilhas. Reservar ambas as versões de cada enunciado de teste.
Treinar uma CNN em mic1 e avaliá-la em mic1 e mic2, sobre exatamente esses
enunciados reservados. Repetir com treino em mic2. Isso produz quatro células:

| Microfone de treino | Teste mic1 | Teste mic2 |
|---|---|---|
| mic1 | referência do modelo 1 | transferência do modelo 1 |
| mic2 | transferência do modelo 2 | referência do modelo 2 |

Cada diferença horizontal compara o **mesmo checkpoint**, com a mesma
normalização estimada na origem, sobre duas capturas da mesma gravação inédita
para ele. A validação deve permanecer no microfone de origem. Fixar uma divisão
por enunciado e três sementes de treino registradas: duas direções × três
sementes = seis ajustes. Usar a CNN como instrumento já disponível, sem escolhê-la
novamente por uma competição no teste.

**Por que agora.** `prepare_fold` retira aproximadamente 20% dos enunciados para
teste e um enunciado por locutor para validação. O cross convencional usa todas
as frases de mic2 no teste e reserva dez frases por locutor em mic1 para
validação. O disjunto corta a interseção ordenada ao meio e também reserva dez
frases. Portanto, subtrair 38% de 97% mistura a troca de trilha com diferentes
tamanhos de treino, testes e regras de validação. Os 43,31% e 49,43% no sentido
inverso também tornam inadequado tratar 38% como um número independente da direção
e sugerem que até o ranking das arquiteturas pode depender dela.

**O que os resultados decidem.**

- Uma perda grande nas duas direções, estável entre sementes, permite quantificar
  a penalidade de transferência dessa cadeia de processamento, condicionada à
  divisão escolhida. Substitui a subtração entre protocolos diferentes.
- Uma perda predominantemente em uma direção exige reportar a matriz completa:
  a generalização depende de qual trilha fornece treino e teste. Uma média dos
  dois sentidos esconderia uma propriedade do protocolo.
- Uma perda muito menor, ou uma referência intra também baixa, reduz a parcela
  da diferença antiga atribuível à troca de trilha sob comparação controlada.
  Tamanho do treino e escolha dos enunciados passam a importar na explicação.
- Variação entre sementes comparável às diferenças entre arquiteturas impede
  sustentar seu ranking com as execuções atuais. Não impede reportar a perda
  pareada dentro de cada execução.

Mesmo essa comparação mede a troca de microfone **mais seus efeitos no
pré-processamento**. O relatório de alinhamento já documenta que o VAD muda os
recortes. Não a chamar de porcentagem de informação de canal, nem de separação
entre voz e sessão.

**Custo.** **6–30 minutos de treino em GPU**, pela referência de 1–5 minutos por
ajuste, mais **3–10 minutos** de leitura e inferência. Cerca de meio dia para o
instrumento e o registro das divisões. Uma etapa inicial ainda mais barata é
reavaliar os cinco checkpoints intra da CNN nos enunciados correspondentes de
mic2: zero treinos e aproximadamente **5–15 minutos**. Ela fornece a comparação
pareada mic1 → mic2, mas não substitui a medição de direção e sementes.

**O que precisa existir.** Todos os MFCCs necessários estão em disco; o áudio
bruto não é necessário. Falta um avaliador que conserve os índices e as
estatísticas do treino e aceite os dois testes. Os perfis atuais, sozinhos, não
produzem essa comparação. O treinador também precisa permitir fixar e registrar
a semente de inicialização e de treino: `VALIDATION_SEED` governa a divisão, não
a aleatoriedade da rede. Não basta executar três vezes com esse mesmo campo.

## 3. Testar se a pista de baixa energia desapareceu ou mudou de representação

**O que se mede.** Transferência das assinaturas `silence` após uma transformação
entre microfones aprendida em **outros locutores**. A hipótese a testar é que uma
pista compartilhada entre trilhas ficou inacessível ao classificador na escala
ou nas coordenadas originais.

Separar 36 locutores para estimar uma transformação afim regularizada de
assinaturas mic2 → mic1, usando seus pares, sem rótulos de identidade no ajuste.
Fixar a regularização antes da avaliação. Nos outros 72 locutores, treinar o
classificador de identidade apenas em mic1, com uma parte dos enunciados, e medir
em enunciados reservados: mic1, mic2 original e mic2 transformado. A transformação
não pode usar nenhum par desses 72 locutores. Repetir com três grupos de
calibração definidos previamente e realizar o mesmo procedimento em `speech`
como controle de funcionamento da transformação.

O acaso dessa tarefa é **1/72 = 1,39%**. Comparar as três condições sobre a mesma
população e divisão; os 4,33% anteriores, com 108 classes e outro treino, não são
o controle numérico desse novo experimento. As rotações compartilham locutores de
avaliação e não são três amostras independentes.

**Por que agora.** A conclusão “é o transdutor e não a sessão” é mais forte que o
diagnóstico realizado. O mesmo modelo de microfone é usado para todos os
locutores; sozinho, seu nome não separa as classes. Uma particularidade de
sessão pode ser filtrada diferentemente pelos dois microfones. A falha de um
classificador ao atravessar essa transformação não demonstra que a particularidade
deixou de existir. Essa objeção independe de explicar o resíduo de 4,33%.

**O que os resultados decidem.**

- Se a transformação recuperar uma parcela relevante da identificação por baixa
  energia em locutores que não participaram de seu ajuste, fica refutada a
  leitura de que a pista foi eliminada pela troca de transdutor. Ela permanece
  recuperável após calibração entre canais. Isso tampouco prova que seja sessão:
  `silence` é um recorte por energia, não uma anotação de ausência de voz.
- Se houver boa identificação em mic1, a transformação funcionar em `speech`,
  mas não recuperar `silence`, fica enfraquecida a explicação por uma transformação
  afim compartilhada. A conclusão admitida continua sendo não transferência sob
  os instrumentos testados; não ausência de informação de sessão.
- Se a referência mic1 falhar, ou a transformação não funcionar nem em `speech`,
  a falta de recuperação em `silence` será inconclusiva. Encerrar essa tentativa,
  sem escalar para uma rede de adaptação até encontrar recuperação.

Uma recuperação de, por exemplo, cinco pontos percentuais acima do cross sem
transformação pode ser fixada como efeito de interesse, acompanhada da diferença
por locutor. O limiar não deve ser escolhido depois de observado o resultado.

**Custo.** Zero treino de redes. **5–20 minutos de CPU** para leitura das
assinaturas, seis classificadores de identidade e as transformações nas três
rotações e duas condições. Aproximadamente meio a um dia de instrumentação.

**O que precisa existir.** As assinaturas já existem em ambas as trilhas. Usar a
interseção de pares disponíveis em cada condição e informar ausências por
locutor. Não há dependência de áudio bruto. O script atual
`experiments/channel_transfer.py` não separa locutores para calibração nem
enunciados para esse teste: treina em toda a origem e testa em toda a outra
trilha. A proposta exige outro instrumento, não apenas inverter caminhos.

## 4. Usar o BrSD para testar a unidade de separação em uma tarefa que a permita

**O que se mede.** Classificação do campo `Gender` fornecido pelo BrSD, comparando
separação por gravação com separação por pessoa. A mudança de alvo é deliberada:
permite colocar uma pessoa inteira fora do treino sem eliminar a classe que será
predita. A unidade de generalização passa a ser testável neste corpus.

Usar um classificador linear de estatísticas cepstrais e dois recortes: fala e
baixa energia. Construir cinco divisões de cada tipo:

- Por gravação: três leituras de cada pessoa no treino, uma na validação e uma
  no teste, distribuindo os textos entre essas funções de modo equilibrado.
- Por pessoa: 48 pessoas no treino, 16 na validação e 16 no teste, com todas as
  suas leituras juntas. Estratificar pela categoria alvo e verificar o equilíbrio
  de idade e texto entre os conjuntos.

Os dois esquemas têm 240 gravações de treino, 80 de validação e 80 de teste.
Os recortes usam a mesma extração e o mesmo conjunto elegível de gravações.
Fixar um classificador e a regra de seleção na validação, sem busca de
arquiteturas. Reportar acurácia balanceada, cujo acaso binário é 50%, além do
resultado por pessoa. Se a ausência de baixa energia inviabilizar alguma divisão,
registrar a redução e reconstruir os dois protocolos sobre a mesma população.

**Por que agora.** Os resultados atuais de identificação do BrSD são 73,75%,
83,00% e 61,25% para CNN, temporal e atenção. Eles não medem generalização para
novas pessoas ou dispositivos. Repeti-los pouco acrescentaria ao objetivo do
trabalho. Esta comparação investiga, com outro corpus e outro alvo, se a unidade
que se mantém junta na divisão muda a avaliação de um classificador de sinais.
O confundimento declarado entre pessoa e dispositivo motiva o controle, mas não
determina antecipadamente seu resultado para essa nova categoria alvo.

**O que os resultados decidem.**

- Desempenho alto por gravação e queda por pessoa mostram que a avaliação depende
  da reutilização de pessoas e das condições associadas a elas. O resultado
  fundamenta a separação por unidade de aquisição para essa tarefa. A diferença
  não pode ser atribuída exclusivamente ao aparelho.
- Fala com desempenho preservado por pessoa e baixa energia próxima do acaso
  mostram uma tarefa em que existe generalização para pessoas não vistas, sob o
  controle aplicado. O argumento deixa de ser “o corpus não serve” e passa a
  especificar qual alvo e qual divisão ele permite avaliar.
- Baixa energia preditiva mesmo por pessoa revela associação entre a categoria
  alvo e condições compartilhadas por diferentes pessoas, ou conteúdo vocal que
  permaneceu no recorte. Separar pessoas não basta para declarar o controle
  resolvido; é necessário investigar essa associação antes de interpretar a
  classificação como atributo vocal.
- Ambos os protocolos próximos do acaso tornam o instrumento insuficiente para
  esse alvo. Não acrescentar uma rede maior somente para produzir uma diferença;
  encerrar a proposta sem usá-la como demonstração positiva de viés.

**Custo.** Zero treino de redes; vinte ajustes lineares para duas condições,
dois protocolos e cinco divisões, se a regularização for fixada. Reservar
**10–40 minutos de CPU** para extração de assinaturas dos 223 minutos de áudio e
**1–5 minutos** para os classificadores. São estimativas dependentes da extração;
cronometrar uma pequena amostra antes da execução completa. Preparação dos
metadados e das divisões: aproximadamente um dia.

**O que precisa existir.** O PDF
`/home/lsmsqt/datasets/brsd/utterances_info.pdf` contém locutor, `Gender`, idade,
texto e índice do WAV. Os 400 MFCCs e os WAVs estão presentes, mas **não existem
`assinaturas.npz` do BrSD**, nem features do perfil `brsd_vad`. Será preciso
extrair os dois recortes do áudio disponível e validar a ligação com os metadados.
O sistema atual só classifica identidade; essa tarefa requer instrumento próprio.

Se os WAVs forem removidos antes de rodar, a parte de baixa energia ficará
bloqueada até recuperar o BrSD. Os MFCCs salvos permitem comparar a divisão por
gravação e por pessoa sobre o sinal completo, mas não reconstruir um recorte de
silêncio validado. Nesse caso, a comparação perde seu controle de interpretação
e deve ser apresentada com esse alcance menor.

## 5. Comparar uma referência estática com as redes usando os mesmos MFCCs

**O que se mede.** Regressão logística sobre média e desvio temporal dos **mesmos
tensores de 40 coeficientes e 300 quadros** usados na proposta 2. Preservar seus
enunciados, repetição/truncamento e normalização da origem. O classificador
recebe 80 números por gravação; a diferença deliberada é substituir o tratamento
da sequência por estatísticas estáticas e uma decisão linear.

Comparar as quatro células da proposta 2. Se houver seleção de regularização,
usar uma grade pequena definida previamente e apenas a validação da origem.
Comparar com a distribuição dos resultados da CNN, sem escolher sua melhor
semente. A proposta depende da divisão da prioridade 2, não necessariamente da
conclusão de todas as outras análises.

**Por que agora.** O diagnóstico `speech` registra 55,38% cross, acima das redes.
Essa comparação não identifica a causa: suas assinaturas usam a taxa original e
o recorte inteiro, enquanto as redes usam outra representação e comprimento.
Uma referência sobre a mesma entrada testa a afirmação de que o patamar das três
redes reflete uma barreira do protocolo, em vez de uma propriedade dos modelos
e do regime de ajuste escolhidos.

**O que os resultados decidem.**

- Uma vantagem relevante do modelo linear refuta a leitura dos aproximadamente
  38% como teto imposto pelo corpus ou pelo cross. A análise deve separar o
  problema de transferência do problema de ajuste das redes. Ganhar acurácia é
  consequência do diagnóstico, não seu objetivo.
- Resultado próximo ao das redes mostra que estatísticas sem ordem temporal
  bastam para reproduzir esse nível de transferência. Não se pode usar esse
  nível como evidência de que a estrutura temporal aprendida foi necessária.
- Desempenho claramente inferior mostra que essa representação estática linear
  não explica o resultado das redes. O contraste com os 55,38% das assinaturas
  passa a justificar uma investigação de representação, banda ou duração, em
  vez de uma busca geral de arquiteturas. Não prova, sozinho, que a ordem
  temporal seja a informação responsável.

**Custo.** **3–15 minutos de CPU**, incluindo leitura e estatísticas. Com três
valores de regularização e duas direções, são seis ajustes lineares. Zero treinos
adicionais de redes se a proposta 2 já foi executada. Instrumentação de duas a
quatro horas, reaproveitando a divisão daquela proposta.

**O que precisa existir.** Tudo que é dado de entrada já está em disco. Falta
derivar as estatísticas dos tensores e aplicar o avaliador comum. Não substituir
esses vetores pelas assinaturas de 48 kHz existentes: isso reintroduziria a
diferença que o experimento pretende controlar. Não há necessidade de áudio bruto.

## Alcance das conclusões e regra de leitura

Há duas outras conclusões cujo enunciado deve ser reduzido, sem gastar GPU para
repetir controles:

- **Permutação.** O resultado ao acaso é um controle negativo do procedimento
  executado, não uma prova universal de ausência de vazamento. Em
  `_permute_labels`, cada gravação recebe um rótulo embaralhado, inclusive as duas
  versões de um mesmo enunciado. Isso destrói também a relação que um vazamento
  por gravações correlacionadas poderia explorar. O controle não testa todas as
  formas de dependência entre conjuntos. Manter a evidência favorável sem chamar
  o arcabouço inteiro de provadamente livre de vazamento.
- **Texto disjunto.** `_split_utterances` separa metades da interseção de índices;
  não lê transcrições nem equilibra vocabulário ou conteúdo fonético. Além disso,
  a atenção cai de 37,54% para 33,66%, diferença de 3,89 pontos sem repetição, e
  treino e população de teste mudam juntos. O controle enfraquece a explicação
  por reutilização da mesma gravação, mas não estabelece ausência de qualquer
  pista lexical. O VCTK combina passagens comuns com frases de jornal escolhidas
  por locutor, conforme a [descrição oficial do corpus](https://datashare.ed.ac.uk/handle/10283/3443).
  O manifesto foi preservado, mas as transcrições não foram encontradas no disco.

O alinhamento 100/100 é evidência favorável ao pareamento amostrado no áudio
bruto. Não há motivo para repeti-lo como prioridade; seu alcance não inclui
igualdade dos recortes após o VAD, como o próprio relatório registra.

Para as comparações novas, definir previamente um efeito de interesse, por
exemplo cinco pontos percentuais, e calcular diferenças pareadas por locutor.
Reamostrar locutores inteiros para expressar a heterogeneidade entre pessoas,
mantendo seus enunciados e pares juntos; isso não estima a variação entre sessões.
Separar essa incerteza da variação entre sementes. Cinco partições com treinos
sobrepostos tampouco são cinco replicações independentes.

A análise de erros dos checkpoints é exploratória. Usá-la para escolher o próximo
tratamento não transforma o teste já consultado em evidência confirmatória para
esse tratamento. Registrar os contrastes e todas as condições executadas, sem
selecionar a melhor configuração pelo teste. Os artefatos antigos não incluem
todos os estados necessários à reprodução exata; os novos precisam guardar
divisão, configuração, sementes, normalização e predições por gravação.

## Ideias consideradas e descartadas nesta rodada

- **Completar a busca de seis arquiteturas.** Seis arquiteturas × cinco partições
  custariam 30–150 minutos de GPU por protocolo. Um ranking não decide a origem
  das pistas e a seleção pelo teste prejudicaria justamente a avaliação que se
  pretende defender. A referência controlada da prioridade 5 decide primeiro se
  vale investigar os modelos.
- **Rodar multi-microfone apenas para preencher a tabela.** A exposição aos dois
  microfones de teste mede interpolação entre canais conhecidos. Um resultado
  alto não estabelece transferência para canal novo; um baixo tampouco separa as
  causas relevantes aqui. Misturar as trilhas não elimina por construção pistas
  específicas da combinação locutor e canal. Até o diagnóstico de silêncio
  nesse regime, isoladamente, deixaria a mesma ambiguidade de origem.
- **Repetir permutação ou alinhamento em maior escala.** Não há indício novo que
  justifique esse gasto. Mais exemplos dos mesmos controles não ampliam seu
  alcance lógico. A prioridade é formular conclusões compatíveis com o que
  testam.
- **Repetir o disjunto esperando empate, ou treinar um classificador só de texto
  como prova de atalho da rede.** O primeiro repete uma comparação que muda mais
  de uma variável. O segundo pode revelar texto preditivo, mas não demonstra seu
  uso pela rede acústica; um resultado ao acaso também não exclui todas as pistas
  lexicais. Recuperar transcrições serviria a uma auditoria posterior de conteúdo,
  não a declarar esse mecanismo resolvido com um novo número isolado.
- **Apenas aumentar `MAX_FRAMES_CAP` ou alternar VAD.** O truncamento assimétrico
  já é conhecido. Aumentar o teto muda informação disponível e, em algumas
  arquiteturas, a própria dimensão do modelo. Uma recuperação de acurácia não
  separaria essas causas. Um contraste com máscara de fala comum às duas trilhas
  seria mais específico, mas exige áudio VCTK ausente ou máscaras temporais que
  não foram preservadas. Fica atrás dos contrastes que usam os dados disponíveis.
- **CMVN, cortar para 13 coeficientes ou adicionar perturbações aleatórias sem
  uma hipótese mais específica.** Uma queda pode remover tanto identidade quanto
  canal; um ganho pode regularizar o modelo. Nenhum sinal da diferença identifica
  sozinho sua causa. Tampouco há métricas atuais em disco que sustentem a afirmação
  universal de que CMVN leva ao acaso. Isso pede correção do alcance da afirmação,
  não uma grade de transformações. Cortar os coeficientes existentes é viável sem
  áudio, mas não é prioritário antes da comparação de representações.
- **Diagnóstico de só silêncio no BrSD apenas para confirmar dispositivo por
  locutor.** Um resultado alto não provaria origem no dispositivo, e um baixo
  não eliminaria pistas de canal durante a fala. A prioridade 4 usa esse controle
  em uma comparação que pode mudar a unidade de avaliação e a tarefa defensável.
- **Treinar BrSD → VCTK como classificação das identidades atuais.** Os rótulos
  designam pessoas diferentes. Isso não define um teste de identificação fechada
  com os classificadores existentes. Transferência de representação exigiria
  outra tarefa e outro desenho, sem resolver diretamente as perguntas anteriores.
- **Conjunto aberto, EER, calibração e uma bateria de ruído.** Podem constituir
  outro trabalho, mas ainda avaliariam escores produzidos sob as dependências
  atuais. Não são os primeiros instrumentos para decidir o significado da
  classificação fechada. Perturbações acústicas fiéis no VCTK também dependeriam
  de recuperar o áudio bruto.
- **Baixar outro corpus ou coletar novas sessões imediatamente.** É uma mudança
  de escopo com custo de aquisição, preparação e conferência que não cabe na
  estimativa de minutos por treino. Seria necessária para uma afirmação direta
  sobre novas sessões, mas as propostas acima já podem alterar o protocolo e a
  interpretação com os dados disponíveis. Não apresentar essa aquisição como
  mais um experimento barato de GPU.
