# Prompt de reexecução — rascunho com decisões pendentes

Execute este trabalho em `/home/lsmsqt/Documents/sr`. Não inicie esta nova rodada enquanto as decisões pendentes abaixo não forem respondidas. Este documento não autoriza por si só a exclusão de arquivos nem a interrupção da rodada atual.

## Objetivo e decisões que precisam ser fechadas

Refazer a comparação de normalizações para identificação de locutor em VCTK e BRSD, com Silero orientando a seleção das amostras. O VCTK anterior selecionou janelas por RMS sem restringi-las pela atividade Silero; não o apresente como execução deste novo protocolo.

1. PENDENTE: selecionar somente fala ou comparar fala e não fala como condições separadas? Identificar ambas no audit não equivale a treinar com ambas.
2. PENDENTE: quais diretórios derivados de 30/40 devem ser apagados, e quando termina a rodada atual? Nunca interpretar “tudo” como apagar o projeto inteiro.
3. Se houver condição não fala: fechar duração, quantidade por gravação, regra de seleção, balanceamento e tratamento de arquivos sem trecho suficiente. Não inventar preenchimento, concatenação ou limiares para completar a coorte.
4. Para VCTK: explicitar como combinar as duas máscaras Silero. Proposta a revisar: fala comum pela interseção; não fala comum quando ambos classificam não fala. Usar os mesmos índices temporais nos dois canais.
5. Fechar se o trim por RMS deve permanecer antes do Silero. Para estudar não fala, esse trim pode remover material relevante; não adotá-lo automaticamente porque estava no código antigo.

## Antes de limpar ou executar

- Ler as instruções locais e inventariar serviços, timers, processos, filas e saídas atuais. Registrar o estado sem iniciar filas duplicadas.
- Gerar uma lista exata de caminhos e tamanhos da limpeza. Preservar áudios originais, código, documentação e este protocolo. Excluir somente os derivados abrangidos pela decisão do usuário; não seguir links que levem aos originais.
- Coordenar o fim da rodada antiga e seu monitor antes de remover arquivos que possam estar em uso. Não reutilizar o monitor antigo com autorização ou caminhos desatualizados.
- Criar um manifesto versionado do novo protocolo. Nenhuma seleção de janelas antiga pode ser reaproveitada sem comprovar que atende ao Silero e às decisões acima.

## Parâmetros a preservar, salvo alteração explícita do usuário

- VCTK: coorte de 108 locutores, 120 leituras por locutor, pares mic1/mic2. BRSD: 80 locutores, cinco textos, 400 arquivos. Registrar exclusões; nunca descartar amostras silenciosamente.
- Áudio de análise a 16 kHz. 48→16 kHz: `decimate(q=3,n=8,ftype='iir',zero_phase=True)`, antialias interno, sem Butterworth separado. BRSD nativo 44,1 kHz: `resample_poly` diretamente para 16 kHz. BRSD multicanal: média dos canais.
- Rodar primeiro 30 MFCCs e, após validar sua conclusão, 40 MFCCs; mesma coorte, máscaras, janelas, folds e sementes entre as duas dimensões.
- Pré-ênfase 0,97; Hamming de 32 ms, hop de 16 ms, 512/256 amostras a 16 kHz; `center=False`; 128 bandas mel; DCT-II ortonormal; MFCC + Δ + ΔΔ. Entrada de 90 ou 120 características por quadro.
- Janela de 111 quadros consecutivos, aproximadamente 1,792 s. Não concatenar segmentos descontínuos; deltas de largura nove recalculados dentro da janela.
- Silero 6.2.3, limiar 0,5, fala mínima 250 ms, silêncio mínimo 100 ms, margem de fala 30 ms. Rodar sobre áudio mono a 16 kHz antes da pré-ênfase. Classificar quadros pelo centro temporal e registrar que “não fala” significa ausência de detecção do VAD, não silêncio físico comprovado.
- A escolha de candidatos deve obedecer primeiro à máscara Silero e à condição definida. RMS, se aprovado, pode desempatar candidatos elegíveis; jamais substituir o VAD.
- CNN e CNN temporal; não iniciar X-vector ou attention. GPU obrigatória; lote 128; uma tarefa de treino por vez na GPU.

## Normalizações separadas

Aplicar todas à mesma seleção de amostras. Manter as condições de atividade separadas caso sejam autorizadas.

1. Z-score: média/desvio estimados somente no treino de cada fold, por característica; reutilizar em validação e teste.
2. CMN MFCC: subtrair a média temporal de cada coeficiente na janela da própria gravação, antes das derivadas; sem z-score global adicional.
3. CMVN MFCC: centrar e dividir pelo desvio temporal de cada coeficiente da janela, piso 1e-8; calcular derivadas após essa transformação; sem z-score global adicional.
4. RASTA: no log-mel antes da DCT, sobre a sequência contínua, coeficientes b=[0.2,0.1,0,-0.1,-0.2], a=[1,-0.94], inicialização estacionária pelo primeiro quadro, sem compensação de atraso. Sem CMN, CMVN ou z-score adicional.
5. CMVN log-mel: média/desvio por banda antes da DCT, usando os quadros definidos explicitamente pelo protocolo. Para fala, usar quadros Silero de fala da gravação. Se houver experimento de não fala, decidir e registrar o domínio dessas estatísticas antes de executar. Piso 1e-8; sem normalização extra.

Não substituir ausência de fala por “todos os quadros”. Quando uma regra não puder ser satisfeita, registrar o arquivo e a causa e resolver a política de insuficiência antes de treinar. Não acrescentar CMN+RASTA automaticamente; essa combinação exige inclusão explícita na matriz aprovada.

## Partições e avaliação

- Cinco folds. VCTK: 60/20/20 por leituras, mesmas divisões para ambos os microfones e todas as condições; avaliar mic1→mic1, mic1→mic2, mic2→mic1 e mic2→mic2.
- BRSD: leave-one-text-out; validação com um dos quatro textos restantes por locutor, semente 42. Não misturar janelas da mesma gravação em treino e teste.
- VCTK: teto 150 épocas, patience 15. BRSD: teto 1000 épocas, patience 30. Registrar arquitetura, otimizador, taxa de aprendizado, scheduler, critérios de parada e sementes efetivamente usados; conferir que são iguais entre condições antes de lançar a comparação.
- A tarefa é identificação em conjunto fechado. BRSD tem um aparelho por locutor; não concluir eliminação de confundimento de canal. Reportar médias e desvio amostral dos cinco folds e valores de cada fold.

## Verificação obrigatória antes da execução completa

- Produzir uma amostra visual por corpus: waveform, intervalos Silero, máscara de atividade/não atividade e janelas selecionadas, identificadas pelo arquivo de origem.
- Verificar automaticamente que todos os quadros selecionados satisfazem a regra da condição, que o par VCTK usa os mesmos índices e que nenhuma janela cruza partições.
- Validar dimensões, valores finitos, contagem de amostras, índices e igualdade das máscaras/janelas entre métodos e entre 30/40. Comparar os primeiros 30 coeficientes quando a operação permitir essa equivalência.
- Publicar manifesto, matriz de experimentos e exemplo visual para revisão do usuário antes do treino completo, pois a seleção de atividade ainda está pendente neste rascunho.

## Execução contínua e acompanhamento

- Usar um orquestrador persistente e uma trava compartilhada da GPU. Criar diretórios novos com identificador e hash do protocolo; não sobrescrever a rodada antiga.
- Retomar somente checkpoints cujo manifesto, fold, arquitetura, dimensão e seleção coincidam. Salvar estado atomicamente e manter logs por etapa.
- Instalar verificação a cada 30 minutos: consultar serviço/processo e arquivos de progresso por fold. O status global pode mudar apenas entre etapas; não usá-lo sozinho para inferir travamento.
- Registrar progresso anterior e atual, horário da última época e mudanças nos arquivos. Antes de reiniciar, confirmar ausência do worker; nunca iniciar um segundo treino porque o status global está antigo.
- Verificar montagem do HDD, espaço livre e erros de I/O. Não gravar no ponto de montagem vazio nem reparar o disco automaticamente. Não reiniciar indefinidamente falhas de dados, protocolo ou armazenamento.
- Limitar retomadas de falhas transitórias a três tentativas consecutivas, com erro registrado. Bloqueios que exigem decisão devem ser relatados, sem alterar parâmetros para contorná-los.
- Avisar em português quando uma etapa terminar, houver falha, possível travamento ou conclusão. Ao finalizar, validar todos os folds, checkpoints, manifestos e resumos; gerar tabelas 30×40 e relatório fiel ao protocolo e desativar o acompanhamento desta fila.

## Entrega

Entregar o protocolo final, o inventário de limpeza executada, os exemplos visuais de seleção, a matriz completa de condições, o comando e serviço da execução, a rotina de acompanhamento e os resultados finais. Não declarar um experimento concluído ou aderente ao Silero sem verificar os artefatos reais.
