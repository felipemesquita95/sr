# Retomar a correção dos experimentos BrSD × VCTK (8 kHz)

## Atualização de 24/09/2026, 11h22 (São Paulo)

O usuário pediu dois controles adicionais. O primeiro, **todos os quadros de
atividade versus todos os de baixa atividade**, terminou: cada gravação foi
resumida pela média e pelo desvio dos 40 MFCCs em todos os quadros da condição,
e classificada com a mesma regressão ridge linear. São 17.270 gravações pareadas,
108 locutores e cinco partições. Resultados em
[`resultado_todos_quadros_vctk.md`](resultado_todos_quadros_vctk.md).
O segundo controle está em `sr-vctk-mixed-probe.service`: treina as mesmas três
redes nos dois microfones para 5+5 quadros, 20 só de atividade e 10+10 quadros,
depois mede transferência entre microfones e gera
[`resultado_atividade_mais_baixa_vctk.md`](resultado_atividade_mais_baixa_vctk.md).
O par de 20 quadros usa as mesmas 17.270 gravações nas duas condições;
o controle 5+5 usa as 17.272 gravações dos testes anteriores de dez quadros.
Para acompanhar:
`systemctl --user status sr-vctk-mixed-probe.service --no-pager` e
`journalctl --user -u sr-vctk-mixed-probe.service -n 30 --no-pager`.

## Atualização de 24/09/2026, 10h34 (São Paulo)

O controle sem seleção / atividade / baixa atividade **terminou sem erros**:
90 partições de treino e 90 avaliações no microfone oposto foram concluídas.
O relatório final está em
[`resultado_atividade_baixa_vctk.md`](resultado_atividade_baixa_vctk.md).
Atividade venceu baixa atividade nas seis combinações de rede e microfone.
Baixa atividade ficou acima do acaso no mesmo microfone (54,49%–69,77%),
mas caiu para 6,67%–10,62% ao trocar de microfone. A atividade obteve
25,84%–36,65% ao trocar de microfone. O acaso é 0,93% em 108 classes.
O detector é de energia; esses números não demonstram silêncio sem fala.
O serviço transitório `sr-vctk-activity-probe-v2.service` concluiu e desapareceu
da lista de unidades ativas.

## Atualização de 24/09/2026, 07h42 (São Paulo)

Os treinos corrigidos de 77 e 153 quadros e suas avaliações cross-mic terminaram.
A comparação justa entre durações, com as mesmas 18.067 gravações e partições,
está em [`comparacao_pareada_77_153.md`](comparacao_pareada_77_153.md).

O usuário pediu que o controle **sem seleção / atividade / baixa atividade**
rode durante a noite e gere uma tabela. A unidade ativa é
`sr-vctk-activity-probe-v2.service`, com
`experiments/run_vctk_activity_probe.sh`. A varredura do áudio completo terminou;
o resumo está em [`vctk_activity_probe_selection.json`](vctk_activity_probe_selection.json).
O detector opera após o filtro e a conversão para 8 kHz, com `top_db=30`, janela
de 256 amostras, salto de 128 e uma margem de quadro nas transições. Foram
selecionados **17.272 pares de gravações dos 108 locutores**, com **10 quadros
por condição**, no mínimo 25 gravações por locutor e no mínimo três em cada
grupo de teste. O corte preliminar de 20 quadros não preservava cobertura
suficiente dos cinco grupos para todos os locutores neste detector mais rígido.

As seis bases de MFCC foram criadas sob `runs/features/vctk_activity10_*`.
O serviço treina três redes em cada condição e microfone, depois executa cross-mic
nas duas direções. Ao completar, gera
[`resultado_atividade_baixa_vctk.md`](resultado_atividade_baixa_vctk.md).
Para acompanhar: `systemctl --user status sr-vctk-activity-probe-v2.service --no-pager`
e `journalctl --user -u sr-vctk-activity-probe-v2.service -n 30 --no-pager`.
Uma tentativa anterior, `sr-vctk-activity-probe.service`, falhou antes do treino
por exigir 20 gravações por locutor com 20 quadros; foi substituída pela unidade
v2. Os 10 quadros são posições espalhadas pelo áudio completo, não um trecho
contínuo. "Baixa atividade" é um rótulo de energia, não silêncio anotado por
uma pessoa.

## Atualização de 23/09/2026, 22h00 (São Paulo)

O usuário também pediu os testes cross mic na versão corrigida. A unidade
`sr-vctk-cross-queued.service` aguarda a conclusão da execução de 153 quadros e
confere as 60 partições intra dos dois comprimentos. Depois executa
`experiments/evaluate_cross_8k.py` para 77 e 153 quadros. O script carrega cada
checkpoint já treinado e avalia as **mesmas gravações inéditas** tanto no
microfone de origem quanto no outro, mantendo a normalização da origem. Faz as
duas direções e as cinco partições para CNN, Temporal CNN e Attention. Não há
novo treino nessa fase; o resultado é uma matriz de transferência pareada,
gravada em `runs/models/vctk8k_cross_pareado/` e
`runs/models/vctk153_cross_pareado/`. Para acompanhar:
`systemctl --user status sr-vctk-cross-queued.service --no-pager` e
`journalctl --user -u sr-vctk-cross-queued.service -f`.

## Atualização de 23/09/2026, 21h43 (São Paulo)

O usuário pediu uma segunda execução após terminar a de 77 quadros. A unidade
`sr-vctk-153-queued.service` está aguardando a conclusão das 30 partições atuais;
então verifica esses artefatos e treina `configs/vctk153_mic1.env` e
`configs/vctk153_mic2.env`, nessa ordem. Para acompanhar:
`systemctl --user status sr-vctk-153-queued.service --no-pager` e
`journalctl --user -u sr-vctk-153-queued.service -f`.

O novo perfil usa os MFCCs já extraídos, mantém as gravações com ao menos 153
quadros e usa os primeiros 153 quadros de cada uma. Restam 18.067 gravações por
microfone, dos 108 locutores (mínimo de 100 por locutor). Cada gravação preserva,
em cada uma das cinco partições, o papel de treino, validação ou teste que tinha
na execução de 77 quadros. Os modelos vão para `runs/models/vctk153_mic1` e
`runs/models/vctk153_mic2`, sem sobrescrever os atuais.

## Atualização de 23/09/2026, 21h24 (São Paulo)

O HDD foi montado em `/media/lsmsqt/HDD` e o treinamento foi retomado a pedido do
usuário. A unidade transitória `sr-vctk-602020.service` foi iniciada com
`experiments/retrain_vctk_602020.sh`. O roteiro agora passa `--resume` para os dois
microfones: valida modelo, `divisao.json` e `metricas.json`, reaproveita partições
completas, refaz as incompletas e recompõe os resumos. Antes do início, foram
reconhecidas exatamente 7 partições completas; `vctk8k_mic1/temporal_cnn/particao3`
continuava incompleta. Ao terminar os dois microfones, o roteiro gera o relatório.

Para acompanhar: `systemctl --user status sr-vctk-602020.service --no-pager` e
`journalctl --user -u sr-vctk-602020.service -f`. O estado abaixo é o registro
histórico da pausa anterior.

Atualizado em **23/09/2026, 19h45 (São Paulo)**. Este arquivo registra o estado para retomar o trabalho sem confundir resultados antigos com os novos.

## O que foi corrigido

- **“Enunciado”** significa o conteúdo falado em um arquivo de áudio; cada arquivo é uma **gravação/amostra** e não tem duração fixa. O gráfico antigo mostrava só os primeiros 5 s por erro de apresentação. As figuras atuais mostram o arquivo inteiro, nome, duração e uma linha vermelha marcando o trecho usado pela rede.
- **BrSD:** 80 locutores × 5 gravações = 400. Em cada uma das 5 partições, por locutor: **3 treino, 1 validação, 1 teste**. Totais: **240/80/80**. O teste gira pelas leituras E1–E5; a validação é sorteada das quatro restantes com semente 42. Os treinos BrSD anteriores já seguiam essa regra e permanecem válidos.
- **VCTK:** o resultado anterior estava errado para a intenção 60/20/20: usava apenas **uma** gravação de validação por locutor. A regra corrigida usa 5 grupos intercalados por locutor: um para teste, o seguinte para validação, três para treino. Em cada partição, mic1 e mic2 usam os mesmos papéis para cada gravação pareada.
- No VCTK, 107 locutores têm 200 gravações por microfone: **120/40/40**. O locutor **p362** tem 123: **73–75 treino, 24–25 validação, 24–25 teste**. Totais por microfone e partição: treino **12.913–12.915**, validação **4.304–4.305**, teste **4.304–4.305**.
- As features no HDD **não foram reextraídas**; só a divisão e os modelos VCTK mudaram. A entrada continua com 40 MFCCs e comprimento mínimo: BrSD **1.007 quadros**, VCTK **77 quadros**. O áudio e o MFCC completos continuam armazenados.

## Estado do trabalho

- **Treino parado a pedido do usuário em 23/09/2026 às 19h45.** A unidade `sr-vctk-602020.service` está **inativa** e não há processo `run_experiment.py` em execução. **7 das 30 combinações** (2 microfones × 3 redes × 5 partições) foram concluídas: as 5 partições de `vctk8k_mic1/cnn` e as partições 1–2 de `vctk8k_mic1/temporal_cnn`. A partição 3 da Temporal CNN foi interrompida durante o treino e **não** conta como concluída.
- O [PDF atual](relatorio_8k.pdf) é **provisório**: explica a divisão e os intervalos, mas não apresenta acurácias VCTK antigas. O PDF comparativo completo só será gerado depois que o treino for retomado e terminar.
- Os modelos e o PDF do protocolo antigo (validação de um arquivo por locutor) foram arquivados em `runs/models/arquivo_protocolo_anterior/` e `docs/arquivo_protocolo_anterior/` para auditoria. **Não use as acurácias deles como resultados 60/20/20.**
- O [CSV de partições](../runs/features/relatorio_particoes_8k.csv) dá o papel de cada gravação em cada partição. Foram conferidos **107.615 pares de papéis idênticos** entre mic1 e mic2.
- O código e os testes estão atualizados; **175 testes passaram**.

## Ao retomar

1. Confirmar que o treino permanece parado e examinar as últimas mensagens, se necessário:

   ```bash
   systemctl --user status sr-vctk-602020.service --no-pager
   journalctl --user -u sr-vctk-602020.service -n 50 --no-pager
   ```

2. Contar os treinos concluídos (meta: **30 arquivos `divisao.json`**):

   ```bash
   find runs/models/vctk8k_mic1 runs/models/vctk8k_mic2 -name divisao.json | wc -l
   ```

3. Ao retomar, implementar um modo de continuar a validação cruzada que **salte apenas partições com `divisao.json` e `metricas.json` válidos**, leia suas métricas para recompor `resumo.json` e refaça desde o início a partição 3 interrompida da Temporal CNN. Isso preserva os sete treinos concluídos. O roteiro atual `experiments/retrain_vctk_602020.sh` **não** oferece esse salto: rodá-lo como está refaz os treinos completos.

4. Depois de concluir as 30 combinações e gerar o relatório, abrir `docs/relatorio_8k.pdf`. Confirmar as contagens 60/20/20, os resultados das 3 redes nos 2 microfones, 5 partições por resultado e as cinco figuras de processamento. Conferir que a tabela de divisão veio do CSV atualizado, não dos modelos arquivados.

Arquivos centrais: `src/sr/features/adjustment.py` (divisão), `src/sr/config/settings.py` (opção de divisão), `configs/vctk8k.env` e `configs/vctk8k_mic2.env` (60/20/20), `experiments/final_report_8k.py` (relatório) e `experiments/retrain_vctk_602020.sh` (execução).
