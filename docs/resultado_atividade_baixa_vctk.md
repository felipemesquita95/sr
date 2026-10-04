# VCTK: quadros sem seleção, com atividade e com baixa atividade

Foram usadas as mesmas 17.272 gravações pareadas, dos 108 locutores, com 10 quadros por gravação em cada condição.
As posições de atividade e baixa atividade foram escolhidas a partir do
áudio original inteiro, após o filtro e a reamostragem para 8 kHz.
A condição sem seleção amostra uniformemente o áudio inteiro;
as outras duas amostram posições com rótulo comum aos dois microfones.
Os quadros são espalhados pela gravação e concatenados em uma entrada
curta; eles não formam necessariamente um trecho contínuo de fala.
Há uma margem de um quadro em cada transição. Treino, validação e teste
usam as mesmas gravações nas três condições e nos dois microfones.

O detector mede energia, não uma anotação humana de fala. Baixa atividade
não deve ser interpretada como silêncio garantido.

## Teste no mesmo microfone

| Microfone | Rede | Sem seleção | Atividade | Baixa atividade |
|---|---|---:|---:|---:|
| mic1 | cnn | 78,20% | 87,39% | 68,23% |
| mic1 | temporal_cnn | 74,89% | 85,06% | 65,86% |
| mic1 | attention | 83,60% | 90,95% | 69,77% |
| mic2 | cnn | 66,80% | 78,55% | 54,68% |
| mic2 | temporal_cnn | 64,34% | 77,42% | 54,49% |
| mic2 | attention | 72,60% | 82,11% | 56,58% |

## Teste ao trocar de microfone

| Treino → teste | Rede | Sem seleção | Atividade | Baixa atividade |
|---|---|---:|---:|---:|
| mic1 → mic2 | cnn | 18,30% | 26,23% | 9,65% |
| mic1 → mic2 | temporal_cnn | 15,28% | 25,84% | 7,56% |
| mic1 → mic2 | attention | 19,20% | 29,38% | 9,98% |
| mic2 → mic1 | cnn | 23,14% | 35,66% | 8,06% |
| mic2 → mic1 | temporal_cnn | 20,75% | 29,88% | 6,67% |
| mic2 → mic1 | attention | 24,35% | 36,65% | 10,62% |

Acaso em 108 classes: 0,93%. Cada célula resume cinco partições.
A detecção usa o mesmo critério de energia nas três condições;
a interpretação deve levar em conta classificação incorreta de fala fraca,
respiração e ruído. Os modelos têm inicialização estocástica; uma execução
por partição não mede toda a variabilidade de treino.

Arquivo de seleção: `docs/vctk_activity_probe_selection.json`; SHA-256 das chaves: `2bb3393d25d339e7cec8692156009f2a6ad1d39935bf294994bb78fa01eb7368`.
