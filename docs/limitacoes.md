# Limitações e ameaças à validade

Este documento registra o que os resultados **não** permitem concluir, e as correções
metodológicas aplicadas em relação à primeira implementação do sistema.

## 1. Confundidor de canal — limitação inerente ao BrSD

**Natureza:** estrutural, não corrigível por escolha de partição.

Cada um dos 80 contribuintes gravou as suas cinco leituras no próprio aparelho, em
ambiente não controlado. A assinatura de canal — resposta do microfone, ruído de
fundo, ganho, codec — é constante dentro de cada locutor e distinta entre locutores,
constituindo um identificador perfeito do locutor que independe da voz.

Identificação em conjunto fechado exige que cada locutor apareça em treino e em teste.
Logo o seu dispositivo também aparece nos dois lados, por construção. O artigo
original evita o efeito nas suas próprias tarefas mantendo todas as gravações de uma
pessoa na mesma partição — estratégia impossível aqui.

**Consequência:** a acurácia obtida no BrSD mede "reconhecer a gravação" tanto quanto
"reconhecer o locutor", e **não estima** o desempenho sobre uma gravação nova da mesma
pessoa em outro dispositivo.

**Medição:** `experiments/channel_probe.py` quantifica o efeito.

## 2. Confundidor de sessão — persiste no VCTK

**Natureza:** estrutural, atenuado mas não eliminado.

O VCTK remove o confundidor de dispositivo: mesma sala e mesmo equipamento para todos
os locutores, e textos distintos por locutor. Mas cada locutor foi gravado em **uma
única sessão**, e o que for particular dela — ganho ajustado no dia, distância e
postura em relação ao microfone, ruído ambiente do momento, respiração — permanece
constante dentro do locutor.

Trocar de corpus muda a **forma** do confundidor, de dispositivo para sessão, sem
alterar a sua estrutura. Este é um resultado do trabalho, não uma ressalva menor:
corpora tidos como controlados não estão livres do problema, e a ausência de
diagnóstico não é evidência de ausência do efeito.

## 3. Vazamento entre teste e validação — corrigido

**Antes:** o conjunto de teste era passado como `validation_data`, e
`EarlyStopping(restore_best_weights=True)` selecionava a melhor época medindo no
próprio teste. A acurácia reportada era o **pico sobre o teste**, sistematicamente
otimista.

**Correção:** a validação é separada do treino, de forma determinística e com todos os
locutores representados. Parada antecipada e ajuste da taxa de aprendizado observam
apenas a validação; o teste é lido uma única vez, na avaliação final.

**Efeito:** os números do BrSD anteriores à correção não são comparáveis aos
posteriores.

## 4. Vazamento no comprimento de padding — corrigido

**Antes:** o comprimento comum de padding era o máximo sobre o conjunto inteiro,
incluindo a partição de teste. Relevante aqui porque o texto 1 do BrSD é
aproximadamente duas vezes mais longo que os demais.

**Correção:** o comprimento é calculado apenas sobre os dados de treino de cada
partição.

## 5. Normalização cepstral por enunciado — descartada

A CMVN (*cepstral mean and variance normalization*) por enunciado é prática padrão em
reconhecimento **de fala**, onde remover a média cepstral cancela a resposta do canal
sem prejudicar o conteúdo linguístico.

Em reconhecimento **de locutor** o efeito é oposto: a média cepstral carrega parte
substancial da identidade vocal, e zerá-la derruba a acurácia ao nível do acaso. A
técnica foi removida do pipeline.

O caso é ilustrativo do problema central do trabalho: a informação de canal e a de
locutor estão entrelaçadas no mesmo espaço de representação, e remover uma custa parte
da outra.

## 6. Limitações de escala

- **BrSD:** 400 gravações, 80 classes, **4 amostras de treino por classe**. Com tão
  poucas amostras e entradas de alta dimensão, qualquer atalho que separe as classes
  perfeitamente será encontrado antes da solução difícil. Não é falha de
  implementação; é o regime de dados.
- **VCTK:** 200 dos aproximadamente 400 enunciados por locutor, limitados pela memória
  disponível durante a montagem dos tensores.
- **Cross-microfone:** dois locutores do VCTK não possuem a trilha `mic2`, e são
  omitidos desse protocolo.

## 7. Alinhamento por índice no protocolo multi-microfone

Os enunciados são numerados por posição na listagem ordenada, e não pelo identificador
original do VCTK. Se uma trilha tiver arquivos ausentes, a posição *k* de `mic1` pode
não corresponder à mesma frase que a posição *k* de `mic2`, quebrando o pareamento por
enunciado e introduzindo um vazamento residual de conteúdo.

**Estado:** o efeito é pequeno com o número atual de ausências, mas o rigor pleno
exigiria indexar pelo identificador real da frase. Registrado como pendência.

## 8. O que não foi avaliado

- **Conjunto aberto.** Todo o trabalho é de conjunto fechado: o sistema escolhe entre
  locutores conhecidos e não sabe rejeitar um desconhecido. Verificação de locutor
  (1:1), com EER e minDCF, não é abordada.
- **Robustez a ruído aditivo** e a condições adversas de gravação.
- **Comparação com sistemas de referência** da literatura (i-vector, x-vector,
  ECAPA-TDNN). A comparação é entre as três arquiteturas propostas, sob protocolo
  idêntico.
- **Generalização a outros domínios de sinal.** A motivação declarada inclui vibração
  e bioacústica, mas nenhum experimento fora de fala foi conduzido. A transferência é
  uma hipótese, não um resultado.
