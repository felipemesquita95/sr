# Comparação pareada: 77 versus 153 quadros no VCTK

As duas durações usam as mesmas 18.067 gravações e, em cada uma das
cinco partições, os mesmos conjuntos de treino, validação e teste.
Todos os modelos foram treinados novamente com a duração correspondente.
Ambos usam os primeiros quadros de cada gravação; não há seleção de
atividade vocal neste experimento.

| Microfone | Rede | 77 quadros | 153 quadros | Diferença (p.p.) |
|---|---|---:|---:|---:|
| mic1 | cnn | 93.91% | 97.40% | +3.49 |
| mic1 | temporal_cnn | 95.96% | 98.60% | +2.64 |
| mic1 | attention | 94.66% | 98.25% | +3.59 |
| mic2 | cnn | 86.89% | 93.91% | +7.02 |
| mic2 | temporal_cnn | 92.54% | 97.08% | +4.54 |
| mic2 | attention | 89.10% | 95.25% | +6.15 |

## Quantidade de quadros de entrada

| Seleção | Gravações | Quadros por gravação | Total de quadros |
|---|---:|---:|---:|
| Corpus completo, 77 | 21.523 | 77 | 1.657.271 |
| Seleção pareada, 77 | 18.067 | 77 | 1.391.159 |
| Seleção pareada, 153 | 18.067 | 153 | 2.764.251 |

Na seleção pareada, 153 entrega 1.373.092 quadros adicionais, ou
98,7% mais quadros de entrada que 77. A unidade de treino é a
gravação, de modo que ambas as durações continuam com 18.067 exemplos.
Entre todos os cortes inteiros possíveis no corpus, 153 maximiza o
produto gravações elegíveis × quadros por gravação.
A acurácia dentro do mesmo microfone pode refletir voz e canal;
o teste entre microfones avalia essa transferência separadamente.
