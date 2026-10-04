# Retomar o experimento VCTK balanceado de atividade

Em 24/09/2026, o usuário autorizou o recorte de **100 locutores × 25
gravações originais pareadas**. São 2.500 gravações no total. Cada uma das cinco
partições contém exatamente 1.500 para treino, 500 para validação e 500 para
teste (15/5/5 por locutor), com os mesmos papéis em mic1 e mic2. O mapeamento
reversível para os IDs originais está em `docs/vctk_bal100_selection.json`.

O recorte exige pelo menos 20 quadros de baixa atividade e 40 de atividade
comuns aos dois microfones. O detector é de energia e baixa atividade não
equivale a silêncio puro. Oito locutores com poucas gravações elegíveis foram
retirados; a tabela de viabilidade está em
`docs/fronteira_locutores_atividade_vctk.md`.

Condições sobre as **mesmas** 2.500 gravações: 20 atividade, 20 baixa,
40 sem seleção, 40 atividade, 20 atividade + 20 baixa. Três arquiteturas e
cinco partições para cada condição e microfone; depois transferência em ambas
as direções. O roteiro `experiments/run_vctk_balanced100.sh` foi iniciado na
unidade transitória `sr-vctk-balanced100.service` às 16:32 locais.
O roteiro gerou `docs/resultado_vctk_bal100_atividade.md`.

Para acompanhar: `systemctl --user status sr-vctk-balanced100.service --no-pager`.
Em 17:04 locais, 66 dos 150 treinos intramicrofone estavam concluídos e o
serviço estava ativo, treinando `unfiltered40_mic1`. As duas condições de
20 quadros já tinham 30/30 treinos concluídos cada uma.

**Estado final:** os 150 treinos intramicrofone e os 150 testes pareados de
troca de microfone terminaram. Os 150 arquivos `divisao.json` foram conferidos:
todos registram 1.500/500/500 e a largura esperada para sua condição.
O relatório final foi gerado. A unidade transitória encerrou e foi coletada
automaticamente pelo systemd (`--collect`).

Se a unidade falhar, inspecionar `journalctl --user -u
sr-vctk-balanced100.service -n 100 --no-pager`. O roteiro é retomável por
`SpeakerRecognitionSystem(..., resume=True)` e a preparação ignora features
existentes com a forma correta. Não repetir treinos concluídos sem necessidade.
