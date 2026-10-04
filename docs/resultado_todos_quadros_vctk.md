# Todos os quadros de atividade e baixa atividade — VCTK

Mesmas 17.270 gravações pareadas, 108 locutores e cinco partições em
todas as condições. Cada gravação vira um vetor de 80 números:
média e desvio dos 40 MFCCs calculados sobre **todos os quadros** da
condição indicada. O classificador linear é o mesmo em todas as linhas.
A normalização é ajustada apenas no treino do microfone de origem.
O detector é de energia; baixa atividade não significa silêncio puro.

| Condição | Quadros medianos por gravação | Mesmo mic1 | Mesmo mic2 | Mic1 → Mic2 | Mic2 → Mic1 |
|---|---:|---:|---:|---:|---:|
| Todos, sem seleção | 209 | 95,30% | 89,29% | 34,31% | 37,32% |
| Todos de atividade | 88 | 91,26% | 82,72% | 40,04% | 45,91% |
| Todos de baixa atividade | 55 | 74,56% | 61,80% | 13,06% | 22,13% |
| 20 de atividade | 20 | 87,01% | 75,98% | 35,99% | 42,72% |
| 10 atividade + 10 baixa | 20 | 85,71% | 72,84% | 26,44% | 32,47% |

Acaso: 0,93% em 108 classes. As condições de 20 quadros usam
os mesmos registros e o mesmo classificador das condições com
todos os quadros; diferenças de duração são parte do efeito medido.
A seleção de quadros de baixa atividade ainda pode conter fala fraca,
respiração e ruído.
