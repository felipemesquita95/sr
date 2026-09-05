# Avaliação do projeto e dos resultados — 4 de setembro de 2026

Nota: **7,5/10 para o projeto experimental**, como avaliação qualitativa da
implementação, dos controles e da força das conclusões. Os resultados atuais são
coerentes com sensibilidade às condições de gravação, mas não demonstram separação
causal entre voz e canal.

Esta leitura usa os `metricas.json` e `resumo.json` presentes em `runs/models/`,
o diagnóstico de travessia e a implementação atual. Não foram repetidos treinos
para esta avaliação. Os artefatos antigos não incluem um snapshot completo do
perfil, sementes de inicialização e versão do código; sua reprodução exata não é
verificável somente pelas métricas.

## Resultados efetivamente disponíveis

| Experimento | CNN cepstral | CNN temporal | Atenção |
|---|---:|---:|---:|
| BrSD, 5 partições | 73,75% ± 4,81 | 83,00% ± 3,22 | 61,25% ± 5,65 |
| VCTK intra mic1, 5 partições | 97,34% ± 0,24 | 97,56% ± 0,77 | 97,54% ± 0,36 |
| VCTK cross, com VAD, 1 execução | 38,12% | 36,65% | 37,54% |
| VCTK cross, sem VAD, 1 execução | 38,20% | 18,50% | 30,29% |

Desvios são populacionais entre partições, seguindo o relatório do sistema, e não
intervalos de confiança. Uma execução cross não permite estimar variabilidade.
O acaso uniforme é 1,25% no BrSD e 0,93% no VCTK atual de 108 locutores.

Fonte: `runs/models/{brsd,vctk_mic1,vctk_cross_mic,vctk_cross_mic_novad}/<arquitetura>/`.

O diagnóstico linear de canal registra:

| Recorte | Dentro de mic1 | mic1 → mic2 |
|---|---:|---:|
| Baixa energia (`silence`) | 85,40% | 4,33% |
| Maior energia (`speech`) | 99,64% | 55,38% |
| Completo (`full`) | 99,81% | 51,44% |

Fonte: `runs/models/vctk_cross_mic/diagnostico_travessia/travessia_canal.json`.
O recorte de silêncio dispõe de 21.034 pares, e os demais de 21.523: a população
também não é exatamente igual entre todas as condições.

## O que provavelmente explica os números

1. **Condições de captura carregam pistas associadas ao locutor.** A queda da CNN
   de 97,34% para 38,12%, junto ao diagnóstico de baixa energia, sustenta essa
   hipótese. Subtrair essas acurácias não mede a porcentagem de informação de canal:
   os protocolos diferem também em tamanho do treino, enunciados compartilhados e
   seleção da validação.

2. **Cross-microfone ainda preserva sessão e conteúdo.**
   `prepare_cross_microphone` usa os enunciados de mic1 no treino/validação e todos
   os de mic2 no teste. Isso é válido para transferência entre transdutores de
   gravações pareadas, mas não avalia simultaneamente enunciados inéditos ou sessões
   inéditas. Os 4,33% no recorte de baixa energia são aproximadamente 4,7 vezes o
   acaso e reforçam que não se pode chamar o resultado restante de “voz isolada”.

3. **Desligar o VAD não recupera uniformemente o desempenho.** A CNN praticamente
   não muda, enquanto a temporal cai cerca de 18,15 pontos e a atenção, 7,25. Pausas
   adicionais, truncamento em 300 quadros e agregação temporal podem contribuir.
   São hipóteses, pois esta comparação muda o conteúdo retido e não possui
   repetições com diferentes sementes. A divergência de duração medida com VAD
   (11,5% mediana em 6.200 pares) não basta para atribuir causalmente a queda ao VAD.

4. **O controle de permutação é compatível com ausência de sinal útil.** A temporal
   registra 0,70% ± 0,02 intra e 0,81% cross. O F1 macro próximo de 0,0001 é
   compatível com concentração das predições em poucas classes; não há aqui uma
   demonstração de predições aleatórias uniformes. O controle reduz a suspeita de
   alguns vazamentos, sem provar que todos foram eliminados.

5. **Há um baseline simples competitivo.** Os 55,38% da regressão logística em
   `speech` superam as redes cross disponíveis. Porém, a assinatura é extraída na
   taxa original, usa estatísticas do recorte inteiro e recebe outra representação;
   não é uma comparação controlada de classificadores com os mesmos MFCCs.

## Pontos que reduzem a nota

- `docs/resultados.md` é histórico. Seus 24,09% cross foram comprometidos pelo
  deslocamento de rótulos registrado em `EXPERIMENTOS.md`. O runbook também contém
  pendências que já foram executadas; a UI usa os artefatos atuais.
- “Silêncio” é o complemento de um detector de energia, não uma anotação de
  ausência de fala. Pode incluir respiração e fala fraca.
- No BrSD atual há **3 gravações de treino, 1 de validação e 1 de teste por locutor**,
  não 4 de treino, conforme a divisão implementada.
- O código fixa sementes para partição/permutação, mas não fixa uma semente global
  de inicialização e treinamento das redes.
- Redução de taxa de aprendizado e parada antecipada têm a mesma paciência e
  observam a mesma métrica; quando a estagnação atinge o limite, a redução pode
  ocorrer junto com a parada, sem épocas para explorar a taxa menor.
- Os pares de arquiteturas não isolam necessariamente uma única variável: a CNN
  cepstral e a temporal também diferem em agregação e número de parâmetros. Atenção
  com viés posicional e `Flatten` não torna a rede completa invariante a permutação.

## Melhoria prioritária

Adicionar um protocolo **cross-microfone com enunciados separados**: reservar um
grupo de enunciados por locutor, treinar no mic1 somente sobre os demais e testar no
mic2 somente no grupo reservado. A validação sai apenas dos grupos de treino.
Executar também no sentido inverso e repetir com sementes de treino registradas.

Manter essa comparação ao lado do cross pareado atual torna explícitas duas
perguntas: transferência de transdutor sobre a mesma gravação e transferência
sobre outra gravação. Isso pode ser implementado com os MFCCs já existentes.
Separar o efeito de sessão exigirá dados de sessões independentes por locutor.

Esta melhoria experimental é uma proposta; a implementação desta entrega é a
interface de inspeção e os ajustes necessários para acioná-la corretamente.
