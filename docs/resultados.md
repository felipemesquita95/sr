# Resultados

> **Atualização em 04/09/2026:** este documento preserva os resultados históricos.
> O cross-microfone abaixo foi comprometido pelo deslocamento de rótulos descrito
> em `EXPERIMENTOS.md`; não sustenta as conclusões originais de “voz isolada”.
> Para os artefatos da implementação atual e sua interpretação, consulte
> [`avaliacao_atual.md`](avaliacao_atual.md) ou a interface em `ui/`.

Todos os números são acurácia média sobre as partições da validação cruzada, com
desvio entre partições. O nível do acaso é 1/*N*, onde *N* é o número de locutores.

> **Procedência.** Os resultados abaixo foram obtidos com a implementação anterior
> deste sistema, já com as correções de vazamento descritas em
> [`limitacoes.md`](limitacoes.md) aplicadas. Eles são reproduzidos aqui como
> referência. A reexecução sobre a implementação atual está em curso; divergências
> serão registradas, e as causas prováveis são as revisões de processamento de sinal
> documentadas em [`metodologia.md`](metodologia.md).

## BrSD — 80 locutores, acaso 1,25%

| Perfil | CNN cepstral | Atenção |
|---|---|---|
| Referência (8 kHz, sem VAD) | **75,75%** ± 4,37 | 58,75% ± 6,75 |
| VAD + 16 kHz | **79,75%** ± 3,39 | 68,50% ± 7,72 |

Duas leituras:

- **Ampliar a banda e remover o silêncio ajuda.** O ganho de 4 pontos ao passar para
  16 kHz com VAD indica que a informação acima de 4 kHz é útil, e que o silêncio
  interno não era essencial ao desempenho.
- **A atenção perde da convolução no BrSD**, por margem larga. Com quatro amostras de
  treino por classe, o modelo de maior capacidade é penalizado. O resultado se inverte
  no VCTK, onde há dados suficientes.

## VCTK — 110 locutores, acaso 0,91%

### Dentro de um mesmo microfone

| Arquitetura | Acurácia |
|---|---|
| CNN cepstral | 97,33% ± 0,57 |
| Atenção | **97,90%** ± 0,37 |

### Diagnóstico de canal

| Condição | Acurácia |
|---|---|
| **Só silêncio** | **85,33%** ± 0,70 |
| Acaso | 0,91% |

A identidade do locutor é predita com 85% de acurácia a partir de trechos **sem fala
alguma**, por um classificador linear. Contra uma acurácia completa de 97,9%, isso
significa que a maior parte da separabilidade entre locutores está disponível fora da
fala.

O resultado surpreende porque o VCTK é um corpus controlado — mesma sala, mesmo
equipamento para todos. O confundidor não é o dispositivo, e sim a **sessão**: cada
locutor foi gravado uma única vez, e o que caracteriza aquela sessão caracteriza o
locutor.

> A condição *só fala* não foi medida na implementação anterior. É ela que quantifica
> o efeito, pois a condição completa contém o silêncio e portanto não serve de
> contraste. O instrumento atual a implementa.

### Cross-microfone: treina em `mic1`, avalia em `mic2`

| Arquitetura | 13 coeficientes | 40 coeficientes |
|---|---|---|
| CNN cepstral | 12,47% | **24,09%** |
| Atenção | 15,17% | 23,19% |
| CNN temporal | **21,40%** | 22,99% |

Três conclusões:

1. **A queda é de 97,9% para cerca de 23%.** Trocar o microfone entre treino e teste,
   mantendo voz, texto e instante idênticos, destrói três quartos do desempenho. Essa
   distância é a medida direta de quanto o modelo havia aprendido do canal.
2. **O resíduo de 23% é o piso honesto** — cerca de 25 vezes o acaso. É identidade
   vocal real, e não é desprezível; é apenas muito menor do que o número convencional
   sugeria.
3. **As três arquiteturas empatam a 40 coeficientes** (24,09 / 23,19 / 22,99). O
   resultado negativo é informativo: **trocar de arquitetura não corrige viés de
   dados.** A barreira é o deslocamento de domínio entre microfones, não a capacidade
   do modelo.

Passar de 13 para 40 coeficientes aproximadamente dobra o desempenho cruzado, o que é
consistente com a interpretação de que os coeficientes de ordem alta carregam
informação de voz adicional — mas o resultado permanece muito distante do obtido
dentro de um mesmo canal.

### Multi-microfone: treina com ambos, particiona por enunciado

| Arquitetura | Acurácia (1 partição) |
|---|---|
| CNN cepstral | 90,12% |
| Atenção | 95,44% |
| CNN temporal | **96,28%** |

Expor o treino a dois canais recupera o desempenho de 23% para mais de 90%.

**Este resultado ainda não pode ser interpretado como solução do problema.** Ele
mostra que a assinatura do *modelo de microfone* deixou de ser preditiva, mas as duas
trilhas compartilham a mesma sessão — e portanto o mesmo ganho, ambiente, postura e
respiração. O confundidor de sessão sobrevive intacto ao protocolo.

> **Experimento pendente e decisivo:** aplicar o diagnóstico de canal sob o protocolo
> multi-microfone. Se o só-silêncio continuar próximo de 85%, os 96% não representam
> aprendizado de voz, e sim a substituição de um confundidor por outro. Sem essa
> medida, o significado dos números desta seção fica em aberto.

Os resultados são de uma única partição (`MAX_FOLDS=1`), portanto sem estimativa de
desvio. A validação cruzada completa é pendência.

## Síntese

| Condição | Acurácia | O que mede |
|---|---|---|
| VCTK, mesmo microfone | 97,90% | Voz **e** canal, indistinguíveis |
| VCTK, só silêncio | 85,33% | Canal isolado |
| VCTK, cross-microfone | ~23% | Voz isolada do modelo de microfone |
| VCTK, multi-microfone | 96,28% | Voz, com o microfone descorrelacionado — sessão ainda não |
| Acaso | 0,91% | — |

A leitura conjunta é o resultado principal do trabalho: **entre 97,9% e cerca de 23%
está a diferença entre o que um protocolo convencional reporta e o que o sistema de
fato aprendeu sobre vozes.** Nenhuma das duas medidas é errada; elas respondem a
perguntas diferentes, e apenas a segunda responde à pergunta que o título de um
sistema de reconhecimento de locutor promete responder.

## Reprodução

```bash
SR_CONFIG=configs/brsd.env      python experiments/run_experiment.py
SR_CONFIG=configs/brsd.env      python experiments/channel_probe.py
SR_CONFIG=configs/vctk.env      python experiments/run_experiment.py
SR_CONFIG=configs/vctk_mic2.env python experiments/run_experiment.py --preprocess-only
SR_CONFIG=configs/vctk_cross_mic.env  python experiments/run_experiment.py
SR_CONFIG=configs/vctk_multi_mic.env  python experiments/run_experiment.py
```
