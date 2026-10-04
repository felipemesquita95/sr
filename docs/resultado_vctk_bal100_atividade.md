# VCTK: atividade e baixa atividade, 100 locutores balanceados

Cada uma das 100 pessoas tem 25 gravações pareadas nos dois microfones.
Em cada uma das cinco partições, por pessoa são 15 para treino, 5 para
validação e 5 para teste. Todas as condições usam as mesmas gravações
e os mesmos papéis. Mapeamento em `vctk_bal100_selection.json`.
Baixa atividade é uma medida de energia, não silêncio anotado.

A comparação de 20 quadros confronta atividade e baixa atividade.
A comparação de 40 confronta atividade pura, mistura 20+20 e quadros
uniformemente amostrados do áudio inteiro. São quadros de posições
possivelmente não contíguas, ordenados pela posição temporal.

## 20 quadros

### Teste no mesmo microfone

| Microfone | Rede | 20 atividade | 20 baixa |
|---|---|---:|---:|
| mic1 | cnn | 91,64% | 75,40% |
| mic1 | temporal_cnn | 83,24% | 68,08% |
| mic1 | attention | 89,04% | 68,96% |
| mic2 | cnn | 82,32% | 58,08% |
| mic2 | temporal_cnn | 75,44% | 54,12% |
| mic2 | attention | 74,96% | 53,20% |

### Troca de microfone

| Treino → teste | Rede | 20 atividade | 20 baixa |
|---|---|---:|---:|
| mic1 → mic2 | cnn | 31,40% | 10,84% |
| mic1 → mic2 | temporal_cnn | 26,72% | 5,92% |
| mic1 → mic2 | attention | 30,04% | 7,48% |
| mic2 → mic1 | cnn | 40,92% | 10,76% |
| mic2 → mic1 | temporal_cnn | 27,56% | 6,60% |
| mic2 → mic1 | attention | 38,16% | 7,24% |

## 40 quadros

### Teste no mesmo microfone

| Microfone | Rede | 40 sem seleção | 40 atividade | 20 atividade + 20 baixa |
|---|---|---:|---:|---:|
| mic1 | cnn | 93,68% | 93,60% | 93,96% |
| mic1 | temporal_cnn | 86,64% | 88,08% | 87,48% |
| mic1 | attention | 92,04% | 89,84% | 91,60% |
| mic2 | cnn | 85,36% | 85,60% | 84,32% |
| mic2 | temporal_cnn | 78,84% | 83,32% | 81,28% |
| mic2 | attention | 79,00% | 77,76% | 80,60% |

### Troca de microfone

| Treino → teste | Rede | 40 sem seleção | 40 atividade | 20 atividade + 20 baixa |
|---|---|---:|---:|---:|
| mic1 → mic2 | cnn | 28,08% | 34,92% | 25,28% |
| mic1 → mic2 | temporal_cnn | 17,08% | 29,60% | 17,08% |
| mic1 → mic2 | attention | 20,64% | 32,44% | 20,92% |
| mic2 → mic1 | cnn | 35,64% | 39,96% | 33,76% |
| mic2 → mic1 | temporal_cnn | 20,80% | 27,28% | 19,56% |
| mic2 → mic1 | attention | 28,44% | 36,60% | 29,16% |

Acaso: 1% em 100 classes. Cada célula agrega cinco partições.
O experimento não testa os 108 locutores do corpus original;
compare apenas condições desta coorte de 100 locutores.
