# Seleção das gravações e janelas no VCTK a 16 kHz

Inventário completo dos 108 locutores com pares mic1/mic2 disponíveis:
**43.873 leituras pareadas**. O inventário está em
`output/vctk16_trim_lengths.csv` e contém limites de trim e número de quadros
após a decimação 48→16 kHz. Os originais já pertencem a
`wav48_silence_trimmed`; o trim adicional foi medido nos dois sinais do par.

O menor arquivo do inventário todo é `p279_227`, com **36 quadros** após trim.
Usar todos os arquivos e impor um tamanho único a todos reduziria cada gravação
a esse valor. Como o protocolo exige o mesmo número de gravações para os 108
locutores e divisão exata 60/20/20 em cinco grupos, usamos **120 leituras por
locutor**: o locutor com menos pares, p362, possui 123. Em cada locutor, foram
selecionadas as 120 gravações mais longas após trim, com desempate pelo código
do enunciado. Esta escolha favorece gravações longas e deve ser declarada ao
interpretar o resultado; o teste estima desempenho nessa coorte, não em todas
as gravações do VCTK.

O menor trecho da coorte é `p362_347`, com **111 quadros**. Logo, todas as
12.960 leituras da coorte terão uma janela **contínua de 111 quadros**,
equivalente a 1,792 s de sinal coberto por janelas de 32 ms com hop de 16 ms.
Em cada leitura, a janela é escolhida pela maior soma de RMS ao longo dos 111
quadros, após normalizar a curva de cada microfone pelo seu próprio pico. Os
dois microfones usam o mesmo intervalo da leitura, de modo que o conteúdo
temporal correspondente é mantido. A seleção usa o áudio de ambos os canais
para definir o intervalo, sem usar rótulos de locutor ou resultados do teste.
Essa informação pareada na etapa de pré-processamento deve ser explicitada ao
interpretar a transferência entre microfones.

O manifesto `output/vctk16_cohort_120.json` guarda todos os identificadores,
comprimentos, limites de trim e o grupo do fold. Há exatamente 2.592 leituras
em cada grupo: em um fold, três grupos formam o treino (7.776), um a validação
(2.592) e um o teste (2.592). A divisão é por leitura; os dois canais da mesma
leitura recebem o mesmo grupo. A regra de seleção foi fixada antes de qualquer
treinamento ou consulta aos resultados de teste.

O inventário `output/vctk16_windows.csv` registra, para cada leitura, o número
de quadros após trim, os limites do corte e o início/fim da janela máxima de
RMS. O áudio original e esses limites permitem reconstruir o sinal antes do
recorte de 111 quadros. Foram geradas 18 páginas em
`output/vctk16_auditoria_locutores/`, mostrando ao menos uma gravação de cada
locutor e sua janela selecionada.

![Distribuição de quadros após trim](../output/vctk16_distribuicao_quadros.png)

O processador grava, para cada leitura, os MFCCs, deltas e delta-deltas da janela
escolhida e um manifesto com os índices de início/fim. Delta e delta-delta são
calculados **após** selecionar a janela, para não incorporar quadros externos
nas bordas. São hipóteses de representação para comparação na validação, não
uma alteração da quantidade de quadros.
