# Atividade no áudio completo do VCTK

Foram analisados os FLACs completos de 21.523 enunciados, cada um gravado pelos
microfones 1 e 2, de 108 locutores. Nenhum corte inicial de 77 ou 153 quadros foi
aplicado. O arquivo [`vad_audio_completo_vctk.csv`](vad_audio_completo_vctk.csv)
contém uma linha por enunciado pareado; o resumo numérico está em
[`vad_audio_completo_vctk.json`](vad_audio_completo_vctk.json).

O detector `librosa.effects.split(top_db=30)` foi aplicado separadamente aos
áudios originais de 48 kHz. Cada posição de quadro MFCC, com salto de 16 ms, foi
classificada pelo instante central correspondente. `voz_comum` conta posições
ativas nos dois microfones; `baixa_comum` conta posições de baixa atividade nos
dois. Posições em que os microfones discordam ficam fora desses dois grupos.

| Medida por gravação pareada | Mínimo | Q1 | Mediana | Q3 | Máximo |
|---|---:|---:|---:|---:|---:|
| Quadros MFCC do áudio completo | 77 | 166 | 203 | 254 | 1.035 |
| Atividade comum aos dois microfones | 24 | 83 | 110 | 150 | 736 |
| Baixa atividade comum aos dois microfones | 0 | 26 | 55 | 82 | 723 |

Em 296 gravações pareadas não há quadros de baixa atividade comuns aos dois
microfones. As 21.523 gravações têm quadros de atividade comum.

Para criar versões com exatamente **K quadros ativos e K quadros de baixa
atividade por gravação**, nos mesmos instantes nos dois microfones, é necessário
reter só as gravações com ambas as contagens maiores ou iguais a K:

| K em cada versão | Gravações elegíveis | Fração do corpus | Locutores | Menor número de gravações de um locutor |
|---:|---:|---:|---:|---:|
| 10 | 19.360 | 90,0% | 108 | 59 |
| 15 | 18.369 | 85,3% | 108 | 37 |
| 20 | 17.412 | 80,9% | 108 | 24 |
| 25 | 16.486 | 76,6% | 108 | 14 |
| 30 | 15.537 | 72,2% | 108 | 10 |
| 35 | 14.599 | 67,8% | 108 | 7 |
| 40 | 13.629 | 63,3% | 108 | 5 |
| 50 | 11.562 | 53,7% | 108 | 1 |
| 60 | 9.271 | 43,1% | 107 | 0 |

**Escolha prudente para um teste com cinco partições:** K=20 deixa pelo menos
24 gravações em cada locutor e preserva 80,9% das gravações. K=30 dá sequências
mais longas, mas um locutor fica com apenas dez gravações. K=40 já deixa três
locutores com menos de dez gravações. O protocolo deve formar as partições por
*gravação original* antes de criar as três versões (áudio completo, atividade e
baixa atividade), usando as mesmas gravações em cada condição. A escolha final
de K precisa ser fixada antes de avaliar os modelos.

Este é um detector de energia, não uma anotação de fala humana. “Baixa atividade”
pode conter fala fraca, respiração e ruído; “atividade” pode conter ruído alto.
O diagnóstico estima a viabilidade do teste, mas não demonstra sozinho que os
quadros representam voz ou silêncio puros.
