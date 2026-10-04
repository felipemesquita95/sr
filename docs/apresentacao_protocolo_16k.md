Reconhecimento de locutor

Protocolo pareado de 16 kHz e normalizações separadas

Atualização de 28/09/2026 às 21:46. Resultados de 30 MFCCs concluídos; a rodada de 40 permanece em execução nesta fotografia dos dados.

**Objetivo:** identificar o locutor em uma gravação reservada para teste e medir quanto a classificação muda quando o microfone de treino difere do microfone de teste.

| Corpus | Coorte desta rodada | Captação |
| --- | --- | --- |

| VCTK | 108 locutores × 120 leituras = 12.960 pares | Dois microfones por leitura |
| --- | --- | --- |

| BRSD | 80 locutores × 5 textos = 400 arquivos | Um aparelho por locutor |
| --- | --- | --- |



Processamento do áudio - exemplo real

VCTK p225_241_mic1: recorte de 0.828 a 3.698 s do arquivo de entrada. A janela de 111 quadros é comum ao par mic1/mic2, contínua e escolhida por RMS. O tempo do segundo painel começa no recorte.

Esta versão substitui o relato de 8 kHz, Hann, filtro Butterworth separado e janelas de 77/153 quadros. Esses experimentos anteriores ficam arquivados; seus números não entram nas tabelas atuais.

# 1. Cadeia de processamento vigente

A frequência de saída é 16 kHz nos dois corpora. O pré-processamento e a janela são mantidos entre condições; a intervenção muda apenas a normalização ou a representação indicada.

| Etapa | Configuração atual |
| --- | --- |

| Leitura | VCTK: FLAC mono nativo de 48 kHz. BRSD: WAV; arquivos multicanais convertidos em mono pela média dos canais. |
| --- | --- |

| Recorte de bordas | RMS com limiar relativo de -30 dB, cinco quadros consecutivos; preservar pausas internas. VCTK: união da atividade dos dois microfones. |
| --- | --- |

| Margens e suavização | 100 ms no início, 250 ms no fim; fade de 8 ms nas bordas recortadas. |
| --- | --- |

| 48 → 16 kHz | decimate(q=3, n=8, ftype=iir, zero_phase=True). Antialiasing Chebyshev I interno; nenhum Butterworth separado. |
| --- | --- |

| Exceção BRSD | Arquivos 106-110 são nativos de 44,1 kHz: resample_poly diretamente para 16 kHz, com seu filtro antialiasing interno. |
| --- | --- |

| Pré-ênfase | Coeficiente 0,97 depois da redução da taxa. |
| --- | --- |

| Análise espectral | Hamming de 32 ms (512 amostras); salto de 16 ms (256 amostras); center=False; 128 bandas mel. |
| --- | --- |

| Representação | Log-mel e DCT-II ortonormal; reter 30 ou 40 MFCCs. Calcular Δ e ΔΔ apenas na janela selecionada, largura de nove quadros. |
| --- | --- |

| Janela VCTK | 111 quadros contínuos: maximizar a soma de RMS normalizado dos dois canais; mesmos índices nas duas trilhas e em todas as condições. |
| --- | --- |

| Janela BRSD | 111 quadros contínuos dentro de uma região de fala Silero; escolher por RMS. Mesma janela e máscara em todas as condições. |
| --- | --- |



Silero 6.2.3: limiar 0,5; fala mínima de 250 ms; silêncio mínimo de 100 ms; margem de fala de 30 ms. Um quadro conta como fala quando seu centro pertence ao intervalo detectado. No BRSD, ausência de um trecho contínuo de 111 quadros interrompe a extração para inspeção.

# 2. De 30 para 40 coeficientes

Uma gravação é uma amostra. Um quadro é uma janela temporal sobreposta de 32 ms. A janela final contém 111 quadros; sua cobertura no áudio é de 1,792 s (110 saltos de 16 ms mais a última janela de 32 ms).

| Representação | Estáticos | Δ | ΔΔ | Entrada por gravação |
| --- | --- | --- | --- | --- |

| 30 MFCCs | 30 | 30 | 30 | 90 × 111 |
| --- | --- | --- | --- | --- |

| 40 MFCCs | 40 | 40 | 40 | 120 × 111 |
| --- | --- | --- | --- | --- |



Os dez coeficientes adicionais são reextraídos do mesmo áudio; não são obtidos por preenchimento. A coorte, os recortes, a janela, as partições e as sementes permanecem iguais. Os primeiros 30 coeficientes foram conferidos contra a referência em amostras dos dois corpora.

Redes desta comparação

**CNN:** convoluções sobre o eixo das características na entrada original. **CNN temporal:** permuta a entrada para aplicar as convoluções ao longo dos quadros. A normalização pode afetar cada arquitetura de forma diferente; a comparação mantém a mesma rede entre 30 e 40.

A rodada ativa contém somente CNN e CNN temporal. X-vector, attention e combinações de normalizações não fazem parte desta rodada.

# 3. Cinco tratamentos separados

Z-score é a referência. CMN, CMVN, RASTA e CMVN log-mel são testados individualmente, sem acrescentar o z-score global e sem combinar os tratamentos entre si.

| Condição | Domínio e estatísticas | Transformação |
| --- | --- | --- |

| Z-score | 90/120 características; média e desvio estimados apenas nas gravações do treino de origem. | (x - média_treino) / (desvio_treino + 1e-8). Mesmas estatísticas no treino, validação e ambos os testes. |
| --- | --- | --- |

| CMN | MFCCs estáticos; média temporal da própria janela de 111 quadros. | Subtrair a média de cada coeficiente. Δ e ΔΔ mantidos. |
| --- | --- | --- |

| CMVN MFCC | MFCCs estáticos; média e desvio da própria janela de 111 quadros. | Centrar e dividir pelo desvio (piso 1e-8). Escalar Δ e ΔΔ pelo mesmo desvio estático. |
| --- | --- | --- |

| RASTA | Trajetórias das 128 bandas log-mel do áudio recortado completo, antes da DCT. | Filtragem temporal causal; depois DCT, janela fixa e deltas. Sem CMN/CMVN extra. |
| --- | --- | --- |

| CMVN log-mel | Cada banda log-mel; estatísticas nos quadros de fala Silero do próprio áudio recortado, antes da DCT. | Centrar e escalar as bandas; depois DCT, janela fixa e deltas. Sem CMN extra nem RASTA. |
| --- | --- | --- |



RASTA implementado

Numerador [0,2; 0,1; 0; -0,1; -0,2]; denominador [1; -0,94]. Inicialização estacionária a partir do primeiro quadro; sem compensação de atraso. Os coeficientes canônicos são mantidos com salto de 16 ms, portanto a resposta em Hz difere da implementação habitual com salto de 10 ms.

No CMVN log-mel VCTK, se Silero não encontrar fala, as estatísticas usam todos os quadros do recorte e o fallback é avisado. Isso não muda a janela de avaliação. No BRSD, a seleção exige fala contínua suficiente e interrompe quando ela falta.

As tabelas anteriores de CMN/RASTA acompanhados de z-score descrevem outra condição experimental. Elas são preservadas no histórico, mas não são misturadas aos resultados isolados abaixo.

# 4. Como os testes são separados do treino

| Aspecto | VCTK | BRSD |
| --- | --- | --- |

| Classes | 108 locutores | 80 locutores |
| --- | --- | --- |

| Cinco folds | Divisão por leitura: 60% treino, 20% validação, 20% teste. | Leave-one-text-out: um texto reservado para teste em cada fold. |
| --- | --- | --- |

| Por fold | 7.776 pares de treino; 2.592 de validação; 2.592 de teste. | 240 arquivos de treino; 80 de validação; 80 de teste. |
| --- | --- | --- |

| Validação | 24 leituras por locutor, distintas das 72 de treino e 24 de teste. | Uma leitura dos quatro textos restantes sorteada por locutor; semente 42. |
| --- | --- | --- |

| Direções | Treinar no mic1 ou mic2; testar o mesmo checkpoint em mic1 e mic2. | Uma direção audio → audio. |
| --- | --- | --- |

| Treino | GPU obrigatória; lote 128; máximo 150 épocas, paciência 15. | GPU obrigatória; lote 128; máximo 1.000 épocas, paciência 30. |
| --- | --- | --- |



Escolha do modelo e retomada

O checkpoint é escolhido pela acurácia da validação de origem. O teste não decide a época. Em uma interrupção, o treino retoma o melhor checkpoint salvo; resultados concluídos e protocolos compatíveis são reaproveitados.

Estatísticas e comparabilidade

A média e o desvio das tabelas são calculados entre cinco folds. Os folds compartilham locutores e conjuntos de treino; o desvio não é um intervalo de confiança. Diferenças pequenas de médias não demonstram significância estatística.

Limites da interpretação

No VCTK, a transferência compara capturas pareadas da mesma leitura e sessão. Uma melhora cruzada não prova isolamento da identidade vocal dos efeitos de sessão. No BRSD há um aparelho por locutor, portanto voz e dispositivo continuam confundidos.

A identificação é fechada: os locutores das classes aparecem no treino e no teste, em gravações distintas. Esta rodada não testa identificação de locutores nunca vistos nem autenticação em ambiente aberto.

# 5. VCTK: normalizações isoladas com 30 MFCCs

Rodada concluída. Acurácia média ± desvio nos cinco folds. As quatro colunas representam treino → teste no microfone indicado.

| Rede / condição | 1 → 1 | 1 → 2 | 2 → 1 | 2 → 2 |
| --- | --- | --- | --- | --- |

| CNN / Z-score | 95,10 ± 0,28% | 43,53 ± 1,40% | 48,27 ± 1,27% | 92,69 ± 0,28% |
| --- | --- | --- | --- | --- |

| CNN / CMN | 12,25 ± 0,68% | 10,93 ± 0,48% | 11,30 ± 0,58% | 12,07 ± 0,46% |
| --- | --- | --- | --- | --- |

| CNN / CMVN MFCC | 17,31 ± 0,24% | 15,79 ± 0,90% | 16,19 ± 0,47% | 17,32 ± 0,62% |
| --- | --- | --- | --- | --- |

| CNN / RASTA | 52,67 ± 0,80% | 39,49 ± 1,03% | 39,26 ± 1,56% | 50,69 ± 1,74% |
| --- | --- | --- | --- | --- |

| CNN / CMVN log-mel | 24,92 ± 0,30% | 22,29 ± 0,54% | 22,55 ± 0,68% | 25,41 ± 0,93% |
| --- | --- | --- | --- | --- |

| CNN temporal / Z-score | 96,78 ± 0,57% | 47,80 ± 3,19% | 54,84 ± 1,35% | 96,27 ± 0,47% |
| --- | --- | --- | --- | --- |

| CNN temporal / CMN | 92,43 ± 0,23% | 83,00 ± 1,29% | 79,62 ± 1,77% | 93,55 ± 1,13% |
| --- | --- | --- | --- | --- |

| CNN temporal / CMVN MFCC | 85,91 ± 1,14% | 74,62 ± 1,85% | 75,12 ± 1,42% | 85,87 ± 1,28% |
| --- | --- | --- | --- | --- |

| CNN temporal / RASTA | 92,89 ± 0,87% | 78,00 ± 2,26% | 75,91 ± 1,38% | 93,23 ± 0,63% |
| --- | --- | --- | --- | --- |

| CNN temporal / CMVN log-mel | 85,25 ± 1,09% | 76,77 ± 1,65% | 72,17 ± 2,97% | 86,50 ± 1,57% |
| --- | --- | --- | --- | --- |



Gráfico: CNN temporal, média e desvio entre folds. CMN teve as maiores médias cruzadas nesta rodada (83,00% e 79,62%). CMVN log-mel alcançou 76,77% e 72,17%. A CNN cepstral apresentou queda forte nas intervenções; isso não sustenta uma conclusão universal de que a normalização elimina a informação de locutor.

# 6. BRSD: normalizações isoladas com 30 MFCCs

Rodada concluída: 80 locutores, 400 arquivos, Silero e janela contínua de 111 quadros. Acurácia média ± desvio nos cinco folds leave-one-text-out.

| Condição | CNN | CNN temporal |
| --- | --- | --- |

| Z-score | 47,00 ± 4,97% | 58,75 ± 5,80% |
| --- | --- | --- |

| CMN | 3,25 ± 2,88% | 29,75 ± 6,34% |
| --- | --- | --- |

| CMVN MFCC | 2,25 ± 0,56% | 27,50 ± 4,42% |
| --- | --- | --- |

| RASTA | 3,75 ± 2,17% | 29,75 ± 9,62% |
| --- | --- | --- |

| CMVN log-mel | 1,25 ± 0,88% | 8,25 ± 4,11% |
| --- | --- | --- |



Leitura do resultado

Z-score teve a maior média nas duas redes: 47,00% na CNN e 58,75% na CNN temporal. As quatro intervenções reduziram a acurácia neste corpus e protocolo. A CNN com CMVN log-mel ficou em 1,25%, o nível médio do acaso para 80 classes.

A redução pode envolver perda de pistas vocais, de canal, diferenças de escala ou dificuldade de otimização. Os testes medem o efeito da intervenção; não identificam sozinhos a causa da queda. O resultado do BRSD também não contradiz automaticamente os ganhos de transferência observados no VCTK.

# 7. Rodada de 40 MFCCs e próximos resultados

Fotografia de 28/09/2026 às 21:46: status **running**; etapa **vctk/cnn/cmvn**. Só são exibidas médias quando os cinco folds e seus testes estão completos.

| Rede / condição | 1 → 1 | 1 → 2 | 2 → 1 | 2 → 2 |
| --- | --- | --- | --- | --- |

| CNN / Z-score | 96,67 ± 0,41% | 45,32 ± 1,20% | 54,21 ± 1,87% | 93,33 ± 0,64% |
| --- | --- | --- | --- | --- |

| CNN / CMN | 15,42 ± 1,77% | 14,15 ± 1,02% | 13,51 ± 0,76% | 14,46 ± 0,81% |
| --- | --- | --- | --- | --- |

| CNN temporal / Z-score | 97,62 ± 0,58% | 53,16 ± 3,49% | 56,44 ± 1,41% | 96,71 ± 0,21% |
| --- | --- | --- | --- | --- |

| CNN temporal / CMN | 92,28 ± 0,92% | 83,09 ± 1,15% | 80,73 ± 2,68% | 93,09 ± 1,25% |
| --- | --- | --- | --- | --- |



Comparação direta disponível: z-score, 30 → 40

| Rede / direção | 30 MFCCs | 40 MFCCs | Diferença da média |
| --- | --- | --- | --- |

| CNN / mic1 → mic1 | 95,10 ± 0,28% | 96,67 ± 0,41% | +1,57 pp |
| --- | --- | --- | --- |

| CNN / mic1 → mic2 | 43,53 ± 1,40% | 45,32 ± 1,20% | +1,80 pp |
| --- | --- | --- | --- |

| CNN / mic2 → mic1 | 48,27 ± 1,27% | 54,21 ± 1,87% | +5,94 pp |
| --- | --- | --- | --- |

| CNN / mic2 → mic2 | 92,69 ± 0,28% | 93,33 ± 0,64% | +0,65 pp |
| --- | --- | --- | --- |

| CNN temporal / mic1 → mic1 | 96,78 ± 0,57% | 97,62 ± 0,58% | +0,84 pp |
| --- | --- | --- | --- |

| CNN temporal / mic1 → mic2 | 47,80 ± 3,19% | 53,16 ± 3,49% | +5,36 pp |
| --- | --- | --- | --- |

| CNN temporal / mic2 → mic1 | 54,84 ± 1,35% | 56,44 ± 1,41% | +1,60 pp |
| --- | --- | --- | --- |

| CNN temporal / mic2 → mic2 | 96,27 ± 0,47% | 96,71 ± 0,21% | +0,45 pp |
| --- | --- | --- | --- |



As oito médias VCTK com z-score melhoraram ao passar para 40 coeficientes. Esta é uma descrição das médias, sem teste de significância. As demais condições e o BRSD com 40 ainda devem ser lidos na tabela atualizada quando concluírem.

Execução e rastreabilidade

As duas CNNs e as cinco condições são executadas sequencialmente, com uma única fila de GPU. O monitor local verifica a cada 30 minutos e retoma interrupções recuperáveis pelos checkpoints. O relatório é uma fotografia; os arquivos de status e resultados continuam sendo atualizados pelo treino.

Fontes locais: manifestos em output/vctk16_corrected_features; tabelas output/vctk16_isolated_suite_comparison.md, output/normalization_remaining_comparison.md e output/normalization40_comparison.md; resumos summary.json e completed.json em cada diretório de resultado no HDD. Gerador: experiments/build_protocol_presentation_pdf.py.

Referência metodológica RASTA: Hermansky e Morgan, “RASTA processing of speech”, IEEE Transactions on Speech and Audio Processing, 1994. Implementação e parâmetros atuais registrados nos manifestos; os números apresentados vêm dos artefatos locais, não da literatura.
