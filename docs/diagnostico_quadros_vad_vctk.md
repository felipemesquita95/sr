# Quantidade de quadros com atividade no início das gravações VCTK

Foram lidos os **21.523 pares** de FLACs originais. O detector de atividade do
projeto (`librosa.effects.split`, limiar de 30 dB) foi aplicado ao começo do áudio,
até o instante correspondente aos primeiros 153 quadros MFCC. Cada quadro de
16 ms foi classificado pela presença de atividade no seu instante central.
“Baixa atividade” significa apenas que o detector não marcou fala; pode conter
respiração, fala fraca ou ruído da sessão.

O [CSV por gravação](vad_quadros_vctk.csv) registra as contagens separadas por
microfone nos primeiros 77 e 153 quadros, além das contagens **comuns**: instantes
que os dois microfones classificaram da mesma forma. Uma gravação com menos de
153 quadros conserva sua duração real no CSV.

No subconjunto de **18.067 pares com pelo menos 153 quadros**, a mediana é de
**90 quadros de atividade comum** e **41 de baixa atividade comum**.
Há **312 pares sem nenhum quadro de baixa atividade comum** nesse trecho.

| Quantidade exigida de cada condição, nos mesmos instantes dos dois mics | Pares restantes | Locutores | Menor número de gravações por locutor |
|---:|---:|---:|---:|
| 5 quadros | 16.809 | 108 | 77 |
| 10 quadros | 15.655 | 108 | 39 |
| 20 quadros | 13.591 | 108 | 13 |
| 30 quadros | 11.521 | 108 | 5 |
| 40 quadros | 9.356 | 108 | 1 |

Com a divisão existente em cinco partições, **10 quadros** ainda deixam pelo
menos **4 gravações de validação e 4 de teste por locutor em cada partição**.
Com 20 quadros, algumas partições ficam sem validação ou teste para certos
locutores. Portanto um contraste rigorosamente pareado **dentro dos primeiros
153 quadros** só suporta um recorte muito curto, cerca de 160 ms por condição.
Para um diagnóstico de fala/baixa atividade com duração maior, será necessário
buscar os trechos no áudio completo e repetir a contagem de elegibilidade.

O [resumo numérico](vad_quadros_vctk.json) e o
[script](../experiments/count_vad_frames.py) permitem conferir o cálculo.
