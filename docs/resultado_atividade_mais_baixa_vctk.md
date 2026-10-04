# VCTK: efeito de combinar atividade e baixa atividade

As condições foram obtidas dos mesmos arquivos completos, após o
filtro para 8 kHz. O detector usa energia (`top_db=30`); baixa
atividade não equivale necessariamente a silêncio puro.
Os quadros escolhidos são ordenados no tempo, mas podem vir de
trechos separados da mesma gravação.

## Entradas de 10 quadros

17.272 gravações pareadas; 108 locutores; mínimo de 3 gravações por locutor em cada grupo de teste.

O misto contém **5 quadros de atividade + 5 de baixa atividade**.
A comparação com 10 de atividade testa a substituição de
metade dos quadros, mantendo a largura fixa.

### Teste no mesmo microfone

| Microfone | Rede | Sem seleção | Atividade | Baixa atividade | Atividade + baixa atividade |
|---|---|---:|---:|---:|---:|
| mic1 | cnn | 78,20% | 87,39% | 68,23% | 80,77% |
| mic1 | temporal_cnn | 74,89% | 85,06% | 65,86% | 77,69% |
| mic1 | attention | 83,60% | 90,95% | 69,77% | 85,58% |
| mic2 | cnn | 66,80% | 78,55% | 54,68% | 69,48% |
| mic2 | temporal_cnn | 64,34% | 77,42% | 54,49% | 68,17% |
| mic2 | attention | 72,60% | 82,11% | 56,58% | 75,02% |

### Teste ao trocar de microfone

| Treino → teste | Rede | Sem seleção | Atividade | Baixa atividade | Atividade + baixa atividade |
|---|---|---:|---:|---:|---:|
| mic1 → mic2 | cnn | 18,30% | 26,23% | 9,65% | 20,32% |
| mic1 → mic2 | temporal_cnn | 15,28% | 25,84% | 7,56% | 16,85% |
| mic1 → mic2 | attention | 19,20% | 29,38% | 9,98% | 20,69% |
| mic2 → mic1 | cnn | 23,14% | 35,66% | 8,06% | 25,29% |
| mic2 → mic1 | temporal_cnn | 20,75% | 29,88% | 6,67% | 21,60% |
| mic2 → mic1 | attention | 24,35% | 36,65% | 10,62% | 25,35% |

## Entradas de 20 quadros

17.270 gravações pareadas; 108 locutores; mínimo de 3 gravações por locutor em cada grupo de teste.

O misto contém **10 quadros de atividade + 10 de baixa atividade**.
O controle tem 20 quadros de atividade. Ambos usam a mesma
seleção de gravações, as mesmas partições e a mesma largura.

### Teste no mesmo microfone

| Microfone | Rede | Atividade | Atividade + baixa atividade |
|---|---|---:|---:|
| mic1 | cnn | 92,50% | 90,14% |
| mic1 | temporal_cnn | 93,51% | 90,90% |
| mic1 | attention | 95,17% | 93,35% |
| mic2 | cnn | 86,80% | 82,26% |
| mic2 | temporal_cnn | 88,37% | 83,39% |
| mic2 | attention | 89,00% | 87,04% |

### Teste ao trocar de microfone

| Treino → teste | Rede | Atividade | Atividade + baixa atividade |
|---|---|---:|---:|
| mic1 → mic2 | cnn | 30,28% | 23,97% |
| mic1 → mic2 | temporal_cnn | 27,41% | 17,52% |
| mic1 → mic2 | attention | 31,23% | 21,07% |
| mic2 → mic1 | cnn | 40,17% | 29,28% |
| mic2 → mic1 | temporal_cnn | 32,04% | 21,33% |
| mic2 → mic1 | attention | 39,64% | 27,23% |

O acaso em 108 classes é 0,93%. Cada célula resume cinco partições.
Os dois grupos de tamanho de entrada usam coortes ligeiramente
diferentes (17.272 e 17.270 gravações); compare principalmente
condições **dentro de cada grupo**. A inicialização das redes é
estocástica e não foi repetida com várias sementes.
