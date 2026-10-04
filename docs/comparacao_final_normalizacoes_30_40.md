# Comparação final — 30 versus 40 MFCCs

Fonte: relatórios de cinco folds em `output/`. Snapshot da rodada40 marcado completo em `output/normalization40_status.json` (24/24 etapas); `output/vctk_watchdog_status.json` registrou `validated_complete: true`. Em 30/09/2026 o HDD não estava montado nesta sessão, portanto não foi feita uma segunda validação direta dos `completed.json`.

Todas as cifras são acurácia média ± desvio amostral entre folds, em %. Diferença = média40 − média30 em pontos percentuais. CNN=convolução cepstral; CNN temporal=convolução no tempo.

VCTK: 108 locutores ×120 leituras pareadas, 111 quadros contínuos. As janelas foram escolhidas por RMS; Silero não selecionou atividade/não atividade para esses treinos. CMVN log-mel usou máscara Silero para estatísticas, sem mudar a janela. BRSD:80 locutores×5 textos; Silero restringiu a janela à fala; não houve condição separada de não atividade.

| Corpus | Rede | Normalização | Direção | 30 MFCC | 40 MFCC | Δ40−30 (pp) |
|---|---|---|---|---:|---:|---:|
| VCTK | CNN | zscore | mic1→mic1 | 95.10 ± 0.28 | 96.67 ± 0.41 | +1.57 |
| VCTK | CNN | zscore | mic1→mic2 | 43.53 ± 1.40 | 45.32 ± 1.20 | +1.79 |
| VCTK | CNN | zscore | mic2→mic1 | 48.27 ± 1.27 | 54.21 ± 1.87 | +5.94 |
| VCTK | CNN | zscore | mic2→mic2 | 92.69 ± 0.28 | 93.33 ± 0.64 | +0.64 |
| VCTK | CNN | cmn | mic1→mic1 | 12.25 ± 0.68 | 15.42 ± 1.77 | +3.17 |
| VCTK | CNN | cmn | mic1→mic2 | 10.93 ± 0.48 | 14.15 ± 1.02 | +3.22 |
| VCTK | CNN | cmn | mic2→mic1 | 11.30 ± 0.58 | 13.51 ± 0.76 | +2.21 |
| VCTK | CNN | cmn | mic2→mic2 | 12.07 ± 0.46 | 14.46 ± 0.81 | +2.39 |
| VCTK | CNN | cmvn | mic1→mic1 | 17.31 ± 0.24 | 21.19 ± 0.96 | +3.88 |
| VCTK | CNN | cmvn | mic1→mic2 | 15.79 ± 0.90 | 19.65 ± 0.87 | +3.86 |
| VCTK | CNN | cmvn | mic2→mic1 | 16.19 ± 0.47 | 19.95 ± 0.80 | +3.76 |
| VCTK | CNN | cmvn | mic2→mic2 | 17.32 ± 0.62 | 20.97 ± 0.40 | +3.65 |
| VCTK | CNN | rasta | mic1→mic1 | 52.67 ± 0.80 | 59.27 ± 0.79 | +6.60 |
| VCTK | CNN | rasta | mic1→mic2 | 39.49 ± 1.03 | 43.09 ± 1.37 | +3.60 |
| VCTK | CNN | rasta | mic2→mic1 | 39.26 ± 1.56 | 43.46 ± 0.40 | +4.20 |
| VCTK | CNN | rasta | mic2→mic2 | 50.69 ± 1.74 | 55.11 ± 1.45 | +4.42 |
| VCTK | CNN | cmvn_logmel | mic1→mic1 | 24.92 ± 0.30 | 31.81 ± 0.85 | +6.89 |
| VCTK | CNN | cmvn_logmel | mic1→mic2 | 22.29 ± 0.54 | 28.79 ± 0.65 | +6.50 |
| VCTK | CNN | cmvn_logmel | mic2→mic1 | 22.55 ± 0.68 | 28.65 ± 0.61 | +6.10 |
| VCTK | CNN | cmvn_logmel | mic2→mic2 | 25.41 ± 0.93 | 32.21 ± 1.27 | +6.80 |
| VCTK | CNN temporal | zscore | mic1→mic1 | 96.78 ± 0.57 | 97.62 ± 0.58 | +0.84 |
| VCTK | CNN temporal | zscore | mic1→mic2 | 47.80 ± 3.19 | 53.16 ± 3.49 | +5.36 |
| VCTK | CNN temporal | zscore | mic2→mic1 | 54.84 ± 1.35 | 56.44 ± 1.41 | +1.60 |
| VCTK | CNN temporal | zscore | mic2→mic2 | 96.27 ± 0.47 | 96.71 ± 0.21 | +0.44 |
| VCTK | CNN temporal | cmn | mic1→mic1 | 92.43 ± 0.23 | 92.28 ± 0.92 | -0.15 |
| VCTK | CNN temporal | cmn | mic1→mic2 | 83.00 ± 1.29 | 83.09 ± 1.15 | +0.09 |
| VCTK | CNN temporal | cmn | mic2→mic1 | 79.62 ± 1.77 | 80.73 ± 2.68 | +1.11 |
| VCTK | CNN temporal | cmn | mic2→mic2 | 93.55 ± 1.13 | 93.09 ± 1.25 | -0.46 |
| VCTK | CNN temporal | cmvn | mic1→mic1 | 85.91 ± 1.14 | 82.82 ± 1.19 | -3.09 |
| VCTK | CNN temporal | cmvn | mic1→mic2 | 74.62 ± 1.85 | 71.37 ± 1.59 | -3.25 |
| VCTK | CNN temporal | cmvn | mic2→mic1 | 75.12 ± 1.42 | 73.39 ± 2.07 | -1.73 |
| VCTK | CNN temporal | cmvn | mic2→mic2 | 85.87 ± 1.28 | 83.50 ± 2.06 | -2.37 |
| VCTK | CNN temporal | rasta | mic1→mic1 | 92.89 ± 0.87 | 92.40 ± 0.45 | -0.49 |
| VCTK | CNN temporal | rasta | mic1→mic2 | 78.00 ± 2.26 | 78.87 ± 0.77 | +0.87 |
| VCTK | CNN temporal | rasta | mic2→mic1 | 75.91 ± 1.38 | 79.14 ± 0.92 | +3.23 |
| VCTK | CNN temporal | rasta | mic2→mic2 | 93.23 ± 0.63 | 93.03 ± 0.68 | -0.20 |
| VCTK | CNN temporal | cmvn_logmel | mic1→mic1 | 85.25 ± 1.09 | 83.97 ± 0.87 | -1.28 |
| VCTK | CNN temporal | cmvn_logmel | mic1→mic2 | 76.77 ± 1.65 | 75.03 ± 1.09 | -1.74 |
| VCTK | CNN temporal | cmvn_logmel | mic2→mic1 | 72.17 ± 2.97 | 72.58 ± 1.59 | +0.41 |
| VCTK | CNN temporal | cmvn_logmel | mic2→mic2 | 86.50 ± 1.57 | 85.39 ± 0.78 | -1.11 |
| BRSD | CNN | zscore | audio→audio | 47.00 ± 4.97 | 50.25 ± 3.11 | +3.25 |
| BRSD | CNN | cmn | audio→audio | 3.25 ± 2.88 | 3.75 ± 2.34 | +0.50 |
| BRSD | CNN | cmvn | audio→audio | 2.25 ± 0.56 | 4.50 ± 2.09 | +2.25 |
| BRSD | CNN | rasta | audio→audio | 3.75 ± 2.17 | 3.75 ± 2.34 | +0.00 |
| BRSD | CNN | cmvn_logmel | audio→audio | 1.25 ± 0.88 | 4.75 ± 2.71 | +3.50 |
| BRSD | CNN temporal | zscore | audio→audio | 58.75 ± 5.80 | 58.75 ± 9.35 | +0.00 |
| BRSD | CNN temporal | cmn | audio→audio | 29.75 ± 6.34 | 31.50 ± 4.63 | +1.75 |
| BRSD | CNN temporal | cmvn | audio→audio | 27.50 ± 4.42 | 25.50 ± 5.05 | -2.00 |
| BRSD | CNN temporal | rasta | audio→audio | 29.75 ± 9.62 | 28.75 ± 6.90 | -1.00 |
| BRSD | CNN temporal | cmvn_logmel | audio→audio | 8.25 ± 4.11 | 10.50 ± 6.03 | +2.25 |

As coortes, folds, janelas e sementes são compartilhados dentro de cada corpus. VCTK e BRSD têm tarefas e estruturas de captação distintas; suas acurácias não devem ser comparadas como se fossem a mesma população. No BRSD, o aparelho é associado ao locutor; cross-dispositivo não foi medido. Folds VCTK compartilham locutores e os desvios não são intervalos de confiança.
