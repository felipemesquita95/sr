# Reexecução comparativa · BrSD e VCTK a 8 kHz

Os dados e as features estão no HDD. O BrSD contém 400 gravações de 80 locutores; o VCTK usa 21.523 pares de gravações de 108 locutores, em dois microfones.

## Parâmetros de extração

| Parâmetro | BrSD | VCTK mic1 | VCTK mic2 |
|---|---:|---:|---:|
| Taxa nominal do áudio (Hz) | 48000 | 48000 | 48000 |
| Taxa final (Hz) | 8000 | 8000 | 8000 |
| VAD | false | false | false |
| Limiar VAD (dB; desligado) | 30 | 30 | 30 |
| Pré-ênfase | 0.97 | 0.97 | 0.97 |
| Coeficientes MFCC | 40 | 40 | 40 |
| Janela (amostras) | 256 | 256 | 256 |

Cinco WAVs do BrSD (106–110) têm taxa nativa de 44,1 kHz; todos os outros áudios usados têm 48 kHz. Cada arquivo é filtrado na própria taxa nativa, antes de ser reamostrado diretamente para 8 kHz.

Filtro anti-aliasing: Butterworth de ordem 8, fase zero, corte de 3,6 kHz. Reamostragem polifásica; salto de 128 amostras (16 ms) para janelas de 256 amostras (32 ms). O limite do modelo é o menor enunciado do respectivo corpus; os dois microfones VCTK compartilham o menor valor observado entre eles.

## Comprimento das gravações após a extração

| Trilha | Gravações | Mínimo (quadros) | Média | Máximo (quadros) | Gravações extremas |
|---|---:|---:|---:|---:|---|
| brsd | 400 | 1007 | 2112.95 | 7241 | mín. L28/E4; máx. L2/E1 |
| vctk8k_mic1 | 21523 | 77 | 223.98 | 1035 | mín. L7/E51; máx. L45/E23 |
| vctk8k_mic2 | 21523 | 77 | 223.98 | 1035 | mín. L7/E51; máx. L45/E23 |
| vctk_combinado | 43046 | 77 | 223.98 | 1035 | mín. L7/E51; máx. L45/E23 |

A duração de cada gravação está no arquivo `runs/features/relatorio_comprimentos_8k.csv`. Um quadro avança 16 ms.

## Amostras usadas pelas redes

As três redes recebem as mesmas divisões. No BrSD, cada rede usa as cinco leituras por locutor. No VCTK, cada rede do experimento mic1 usa só mic1; cada rede do experimento mic2 usa só mic2. Não há mistura de microfones nestes experimentos.

| Trilha | Redes | Partições | Treino por partição | Validação | Teste |
|---|---|---:|---:|---:|---:|
| brsd | CNN, Temporal CNN, Attention | 5 | 240 | 80 | 80 |
| vctk8k_mic1 | CNN, Temporal CNN, Attention | 5 | 17110–17111 | 108 | 4304–4305 |
| vctk8k_mic2 | CNN, Temporal CNN, Attention | 5 | 17110–17111 | 108 | 4304–4305 |

O arquivo `runs/features/relatorio_particoes_8k.csv` informa o uso de cada gravação em cada partição.

Para os locutores ilustrados, a tabela abaixo mostra o uso dos cinco primeiros enunciados na partição 1:

| Trilha | E1 | E2 | E3 | E4 | E5 |
|---|---|---|---|---|---|
| brsd | teste | validacao | treino | treino | treino |
| vctk8k_mic1 | teste | treino | treino | treino | treino |
| vctk8k_mic2 | teste | treino | treino | treino | treino |

## Tamanho das redes

| Trilha | Rede | Entrada (MFCC × quadros) | Classes | Parâmetros | Modelo médio (MiB) |
|---|---|---:|---:|---:|---:|
| brsd | cnn | 40 × 1007 | 80 | 452.848 | 5.21 |
| brsd | temporal_cnn | 40 × 1007 | 80 | 107.536 | 1.27 |
| brsd | attention | 40 × 1007 | 80 | 1.730.640 | 19.93 |
| vctk8k_mic1 | cnn | 40 × 77 | 108 | 341.004 | 3.93 |
| vctk8k_mic1 | temporal_cnn | 40 × 77 | 108 | 114.732 | 1.35 |
| vctk8k_mic1 | attention | 40 × 77 | 108 | 1.618.796 | 18.65 |
| vctk8k_mic2 | cnn | 40 × 77 | 108 | 341.004 | 3.93 |
| vctk8k_mic2 | temporal_cnn | 40 × 77 | 108 | 114.732 | 1.35 |
| vctk8k_mic2 | attention | 40 × 77 | 108 | 1.618.796 | 18.65 |

## Resultados das redes

| Trilha | Rede | Acurácia média ± desvio | F1 macro médio | Cinco partições (%) |
|---|---|---:|---:|---|
| brsd | cnn | 72.25 ± 2.78% | 0.646 | 67.5, 72.5, 75.0, 71.2, 75.0 |
| brsd | temporal_cnn | 76.75 ± 4.00% | 0.703 | 71.2, 81.2, 76.2, 81.2, 73.8 |
| brsd | attention | 47.50 ± 7.03% | 0.388 | 37.5, 58.8, 43.8, 50.0, 47.5 |
| vctk8k_mic1 | cnn | 92.68 ± 0.40% | 0.927 | 92.1, 92.8, 92.7, 92.4, 93.3 |
| vctk8k_mic1 | temporal_cnn | 95.28 ± 0.30% | 0.953 | 95.2, 94.9, 95.8, 95.3, 95.3 |
| vctk8k_mic1 | attention | 94.93 ± 0.20% | 0.949 | 95.0, 95.1, 94.6, 94.8, 95.1 |
| vctk8k_mic2 | cnn | 86.34 ± 0.73% | 0.863 | 86.2, 85.0, 87.1, 86.6, 86.8 |
| vctk8k_mic2 | temporal_cnn | 92.47 ± 1.02% | 0.924 | 92.5, 92.8, 91.7, 91.1, 94.1 |
| vctk8k_mic2 | attention | 89.59 ± 0.67% | 0.896 | 88.8, 90.4, 88.8, 90.0, 90.0 |

Os corpora diferem em número de locutores, duração e conteúdo. As acurácias são apresentadas lado a lado como resultados de tarefas fechadas próprias de cada corpus, sem tratá-las como uma medição direta da mesma população.

## Cadeia de processamento lado a lado

À esquerda está o locutor 1 do BrSD; nas outras colunas, p225 do VCTK nos microfones 1 e 2. Os números de locutor dos dois corpora não identificam a mesma pessoa. Cada número abaixo é apenas um índice para organizar as figuras: as leituras 1–5 do BrSD são textos diferentes entre si, e também não correspondem aos textos do VCTK. No VCTK, mic1 e mic2 são as duas captações da mesma leitura.

As formas de onda e os MFCCs mostram cada gravação inteira; a escala de tempo varia entre as colunas. A linha vermelha tracejada marca o fim do trecho entregue à rede: os primeiros 1007 quadros (aproximadamente 16.11 s) no BrSD e 77 quadros (aproximadamente 1.23 s) no VCTK. A extração guarda o MFCC completo; o corte ocorre na entrada da rede.

| Figura | BrSD: arquivo e duração | VCTK mic1: arquivo e duração | VCTK mic2: arquivo e duração |
|---:|---|---|---|
| 1 | 1.wav · 71.54 s | p225_001_mic1.flac · 2.05 s | p225_001_mic2.flac · 2.05 s |
| 2 | 2.wav · 38.78 s | p225_002_mic1.flac · 3.94 s | p225_002_mic2.flac · 3.94 s |
| 3 | 3.wav · 32.83 s | p225_003_mic1.flac · 7.59 s | p225_003_mic2.flac · 7.59 s |
| 4 | 4.wav · 32.18 s | p225_004_mic1.flac · 4.41 s | p225_004_mic2.flac · 4.41 s |
| 5 | 5.wav · 35.76 s | p225_005_mic1.flac · 6.32 s | p225_005_mic2.flac · 6.32 s |

### Gravação ilustrativa 1

![Etapas da gravação ilustrativa 1](figuras/relatorio_8k/gravacao_1.png)

### Gravação ilustrativa 2

![Etapas da gravação ilustrativa 2](figuras/relatorio_8k/gravacao_2.png)

### Gravação ilustrativa 3

![Etapas da gravação ilustrativa 3](figuras/relatorio_8k/gravacao_3.png)

### Gravação ilustrativa 4

![Etapas da gravação ilustrativa 4](figuras/relatorio_8k/gravacao_4.png)

### Gravação ilustrativa 5

![Etapas da gravação ilustrativa 5](figuras/relatorio_8k/gravacao_5.png)

## Fontes dos corpora

- [BrSD, página dos autores](https://sites.google.com/view/brsduem).
- [VCTK 0.92, University of Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3443).
