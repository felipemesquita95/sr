# Metodologia

## Cadeia de processamento

Do áudio bruto ao tensor de entrada da rede:

```
carregar → [VAD] → filtro anti-aliasing → decimação → pré-ênfase → MFCC → alinhar → normalizar
```

### Decimação e banda útil

As gravações originais estão a 48 kHz. O perfil de referência decima para 8 kHz,
reduzindo o custo computacional em um fator de seis.

A decisão tem um custo declarado: 8 kHz limita a banda a 4 kHz pelo teorema da
amostragem, e descarta a região acima disso, onde reside parte da informação de
locutor — fricativas, formantes altos, detalhes da fonte glotal. O perfil
`brsd_vad.env` mantém 16 kHz justamente para quantificar esse custo.

A filtragem anti-aliasing que precede a decimação é **de fase zero**
(`sosfiltfilt`). A filtragem direta introduz atraso de fase dependente da frequência,
que desloca os formantes uns em relação aos outros ao longo do tempo — um artefato
que a análise cepstral subsequente registraria como se fosse propriedade do sinal.

A reamostragem é **polifásica** (`resample_poly`) e não por FFT. As razões envolvidas
são racionais exatas (48/8 = 6, 48/16 = 3), e o método por FFT pressupõe periodicidade
do sinal, produzindo artefatos nas bordas de gravações que não começam e terminam no
mesmo valor.

### Janelamento

Janela de 256 amostras a 8 kHz — **32 ms** — com salto de 128 amostras, ou seja, 50%
de sobreposição. A duração é a convencional em análise de fala: longa o bastante para
conter vários períodos de pitch, curta o bastante para que o sinal seja
aproximadamente estacionário dentro dela. O perfil de 16 kHz usa janela de 512
amostras, preservando os mesmos 32 ms.

### Número de coeficientes

O perfil de referência extrai **40 coeficientes**, valor consideravelmente acima dos
13 a 20 usuais. A escolha decorre de busca empírica no trabalho original, mas tem uma
consequência que precisa ser declarada:

> Com 40 coeficientes, a transformada discreta do cosseno descarta pouca informação, e
> o vetor resultante se aproxima do log-espectro mel completo. Os coeficientes de ordem
> alta descrevem estrutura espectral fina — harmônicos, pitch e **coloração do canal** —
> além do envelope do trato vocal capturado pelos de ordem baixa.

Ou seja, aumentar o número de coeficientes aumenta simultaneamente a informação de voz
e a informação de equipamento disponíveis ao classificador. Comparar 13 e 40
coeficientes sob o protocolo cross-microfone separa os dois efeitos.

## Protocolos experimentais

### Validação cruzada dentro de um mesmo canal

Os enunciados de cada locutor são divididos em `num_folds` grupos pela posição
ordenada. Na partição *k*, o grupo *k* de cada locutor vai para teste; do restante,
um enunciado por locutor é sorteado deterministicamente para validação.

No BrSD, com cinco enunciados e cinco partições, o esquema recai no
*leave-one-utterance-out*. Como todos leram os mesmos textos, a partição *k* retém o
texto *k* de **todos** os locutores, e o conteúdo linguístico do teste nunca aparece
no treino — memorização de texto fica descartada como explicação do desempenho.

Este é o protocolo convencional, e o que produz os números mais altos. Seus resultados
não devem ser lidos isoladamente.

### Cross-microfone

Treina em uma trilha de microfone e avalia na outra. Voz, texto e instante são
idênticos entre as trilhas; apenas o transdutor muda. Um modelo apoiado na assinatura
espectral do microfone de treino não a encontra no teste.

A distância entre a acurácia dentro do mesmo microfone e a obtida aqui mede
diretamente quanto do desempenho aparente se devia ao canal. O que resta é o piso
honesto atribuível à identidade vocal.

### Multi-microfone

Treina com ambas as trilhas, particionando por **enunciado**: as duas trilhas de uma
mesma frase vão para o mesmo lado da divisão, de modo que a frase nunca aparece
simultaneamente em treino e teste. Cada locutor passa a ser visto através de dois
canais, o que descorrelaciona a assinatura do microfone do rótulo.

### Diagnóstico de canal

Independente das redes. Um classificador linear é treinado sobre estatísticas
cepstrais de três recortes do mesmo áudio: só silêncio, só fala e sinal completo.

A comparação informativa é **sinal completo contra só fala**. Contrastar o completo
com o só-silêncio isola menos do que parece, pois o sinal completo preserva as pausas
internas e portanto *contém* a condição de silêncio.

O classificador é deliberadamente simples. Uma rede profunda extrairia a assinatura de
canal de formas mais sutis, então um resultado positivo aqui é um limite **inferior**
para o tamanho do efeito, e não uma estimativa dele.

## Arquiteturas e a questão do eixo

A entrada tem forma `(num_mfccs, num_quadros)`. A `Conv1D` interpreta o primeiro eixo
como passos e o segundo como canais, de modo que **a orientação do tensor decide sobre
o que a rede opera**.

### Convolução sobre o eixo cepstral

Passos são os coeficientes; os quadros são canais. Cada quadro recebe pesos próprios,
e não há invariância temporal: a rede pode aprender que determinado instante da
gravação tem determinada energia.

Há uma objeção teórica a esta arquitetura. **O eixo cepstral não possui estrutura de
vizinhança.** Os coeficientes são projeções sobre bases distintas da transformada
discreta do cosseno; o índice que os ordena é um número de base, não uma coordenada em
um espaço métrico. Convolução pressupõe localidade — que a informação relevante esteja
em janelas de posições contíguas — e essa premissa não se sustenta aqui. Permutar a
ordem dos coeficientes alteraria o resultado sem que nada no sinal tivesse mudado.

### Convolução sobre o eixo temporal

Passos são os quadros; os coeficientes são canais. É a orientação convencional, e a
premissa de localidade é legítima: quadros adjacentes são instantes próximos.

A agregação usa `GlobalAveragePooling1D` em lugar de achatamento. A escolha é
necessária para a comparabilidade: achatar uma sequência longa produz um vetor cujo
comprimento cresce com a duração da entrada, e a camada densa seguinte concentraria
milhões de parâmetros. Em regimes de poucas amostras por classe, isso inviabiliza o
treino — e a diferença observada entre arquiteturas passaria a refletir o tamanho do
modelo, e não o eixo escolhido.

### Atenção sobre o eixo cepstral

A objeção geométrica à convolução cepstral não se aplica à atenção. O mecanismo é
**equivariante a permutação**: relações entre pares quaisquer de posições são
aprendidas diretamente, e a proximidade de índice não confere privilégio algum. Para
um eixo sem estrutura de vizinhança, é o operador adequado — a dependência entre um
coeficiente de ordem baixa e um de ordem alta é modelada tão prontamente quanto entre
dois adjacentes.

O viés posicional aprendido devolve a identidade de cada coeficiente, que a
equivariância apagaria, sem reintroduzir a suposição de que índices próximos sejam
semanticamente próximos: cada posição recebe um vetor independente, e não uma função
suave do índice.

Como subproduto, os pesos de atenção formam uma matriz `num_mfccs × num_mfccs`
diretamente inspecionável, que revela quais coeficientes o modelo relaciona entre si.

## Garantias contra vazamento

Três garantias são mantidas por construção em todos os protocolos:

1. **O teste não influencia o treino.** A parada antecipada e a redução da taxa de
   aprendizado observam exclusivamente a validação, que é separada do treino. Usar o
   teste nessa função faria da acurácia reportada o *máximo* obtido sobre o teste, e
   não uma estimativa de desempenho em dados novos.

2. **O comprimento de padding vem apenas do treino.** Derivá-lo do conjunto completo
   deixaria a duração dos enunciados de teste influenciar o formato da entrada.

3. **As estatísticas de normalização vêm apenas do treino.** Média e desvio por
   coeficiente são calculados sobre o treino e aplicados inalterados aos demais
   conjuntos, como ocorreria em uso real.

Uma quarta decisão, específica deste problema: enunciados curtos são estendidos por
**repetição** do próprio conteúdo, não por preenchimento com zeros. Zeros
introduziriam um trecho de silêncio artificial cuja duração depende do enunciado
original — e como o silêncio é o portador da assinatura de canal sob investigação,
isso criaria exatamente o artefato que o trabalho pretende medir.

## Métricas

Acurácia e F1 macro são reportados em conjunto. Em problemas de muitas classes, a
acurácia global pode conviver com classes inteiras nunca preditas; o F1 macro, por dar
peso igual a cada locutor, expõe esse desequilíbrio.

Todos os resultados são reportados ao lado do **nível do acaso** (1/*N*) e da razão
sobre ele. Com 110 locutores, uma acurácia de 20% soa baixa mas equivale a vinte e
duas vezes o acaso — a leitura absoluta induz a erro sobre a magnitude do efeito.
