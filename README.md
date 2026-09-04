# sr — Reconhecimento Automático de Locutor

Sistema de identificação de locutor (*closed-set*) em conjunto fechado, construído
para investigar uma questão metodológica específica:

> **Quanto da acurácia reportada em reconhecimento de locutor mede a voz, e quanto
> mede o canal de gravação?**

O sistema classifica locutores a partir de coeficientes cepstrais de frequência mel
(MFCC) usando três arquiteturas de rede neural, e inclui instrumentos de diagnóstico
que quantificam o quanto do desempenho se explica por artefatos de gravação em vez
de identidade vocal.

## Motivação

O objetivo do trabalho não é maximizar acurácia. É estabelecer um **protocolo de
avaliação honesto** para classificação de sinais unidimensionais, usando
reconhecimento de locutor como caso de estudo. O mesmo protocolo se aplica a outros
domínios de sinal — diagnóstico de falhas por vibração, bioacústica, monitoramento
de máquinas — onde o mesmo confundidor aparece: **a fonte de gravação é constante
por classe, então o classificador pode identificar o equipamento em vez do fenômeno.**

## Arquitetura

Quatro subsistemas, com responsabilidades separadas:

| Subsistema | Módulo | Responsabilidade |
|---|---|---|
| Pré-processamento | `src/sr/preprocessing` | Áudio bruto → matriz de MFCCs |
| Ajuste de features | `src/sr/features` | MFCCs → tensores de treino/validação/teste |
| Aprendizado profundo | `src/sr/models`, `src/sr/training` | Definição e treinamento das redes |
| Avaliação | `src/sr/evaluation` | Métricas, figuras e relatórios |

Orquestração em `src/sr/system.py`; ponto de entrada em `experiments/run_experiment.py`.

## Arquiteturas de rede

| Nome | Eixo da convolução/atenção | Agregação |
|---|---|---|
| `cnn` | coeficientes cepstrais (quadros como canais) | `Flatten` |
| `temporal_cnn` | tempo (coeficientes como canais) | `GlobalAveragePooling1D` |
| `attention` | coeficientes cepstrais, atenção multi-cabeça | `Flatten` |

A escolha do eixo é deliberada e é objeto de estudo. O eixo cepstral **não possui
estrutura de vizinhança** — coeficientes adjacentes são projeções independentes de
bases distintas da DCT. Convolução pressupõe localidade e portanto é mal-condicionada
nesse eixo; atenção é equivariante a permutação e não faz essa suposição. Ver
[`docs/metodologia.md`](docs/metodologia.md).

## Instrumentos de diagnóstico

- **`experiments/channel_probe.py`** — classifica locutores usando **apenas trechos de
  silêncio**. Acurácia acima do acaso quantifica o confundidor de canal/dispositivo.
- **Protocolo cross-mic** — treina em um microfone e avalia em outro. Isola identidade
  vocal do canal, pois a voz e o texto são idênticos e só o transdutor muda.
- **Protocolo multi-mic** — treina com os dois microfones, particionando por enunciado.
  Descorrelaciona canal e rótulo.

## Uso

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# Treino com validação cruzada
SR_CONFIG=configs/brsd.env .venv/bin/python experiments/run_experiment.py

# Diagnóstico de canal
SR_CONFIG=configs/brsd.env .venv/bin/python experiments/channel_probe.py
```

## Testes

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest
```

A suíte cobre as funções puras — processamento de sinal, alinhamento de
comprimento, métricas e validação dos perfis — e verifica **propriedades**, não
valores numéricos: complementaridade entre fala e silêncio, ausência de zeros no
preenchimento, distinção entre classe nunca testada e classe sempre errada. São as
propriedades que, se quebradas, invalidariam os experimentos sem produzir erro.

## Documentação

- [`docs/metodologia.md`](docs/metodologia.md) — pipeline, decisões de projeto e justificativas
- [`docs/datasets.md`](docs/datasets.md) — BrSD e VCTK, e o confundidor de cada um
- [`docs/limitacoes.md`](docs/limitacoes.md) — ameaças à validade e vazamentos corrigidos
- [`docs/resultados.md`](docs/resultados.md) — resultados experimentais consolidados
