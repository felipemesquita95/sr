# Retomar x-vector VCTK 100 após reiniciar

Em 30/09/2026, o HDD foi montado com sucesso em **somente leitura** usando:

```bash
udisksctl mount -b /dev/sda3 -o ro
findmnt /dev/sda3 -o TARGET,FSTYPE,OPTIONS
```

O resultado confirmado foi `/media/lsmsqt/HDD`, `ntfs3`, opção `ro`.
A montagem padrão (`udisksctl mount -b /dev/sda3`) falhou com erro de NTFS.
Não usar `ntfsfix` nem forçar escrita sem diagnosticar o volume. Depois do reinício,
repetir o comando acima e conferir os dados em
`/media/lsmsqt/HDD/sr_project/vctk100_activity_20260929`.

Rodada a retomar: `experiments/run_vctk_activity_xvector_grid.py`. Resultado antigo:
`/media/lsmsqt/HDD/sr_project/vctk100_activity_xvector40_zscore_cmn_silero_20260929`.
O estado conferido em 30/09 tinha **7 de 20 modelos completos**; o modelo
`100_zscore_fold4_mic2` foi interrompido, mas possui `best.keras` e `history.csv`.

Como o HDD está somente leitura, copiar o diretório de resultados (~145 MB) para
uma saída gravável em `output/` antes de retomar. O script aceita o caminho da saída
como argumento e aproveita os modelos concluídos. Ao final, atualizar
`output/pdf/resultados_vctk_40_80_100_quadros.pdf` incluindo a x-vector de 100
quadros. Não apresentar números parciais como resultado final.

A cópia gravável já foi feita em
`output/vctk100_xvector_silero_20260929/results`. A retomada começou em
30/09/2026 com o comando:

```bash
MPLCONFIGDIR=/home/lsmsqt/Documents/sr/tmp/matplotlib \
  .venv/bin/python experiments/run_vctk_activity_xvector_grid.py \
  output/vctk100_xvector_silero_20260929/results --launch
```

Após reiniciar, conferir o estado com:

```bash
cat output/vctk100_xvector_silero_20260929/results/status.json
tail -20 output/vctk100_xvector_silero_20260929/results/suite_execution.log
```

Se o estado não for `complete` e não houver processo da rodada ativo, repetir o
comando de retomada. A execução pula modelos com `completed.json` e retoma o
checkpoint do modelo interrompido. O PDF só deve ser gerado com 20/20 modelos.

Para acompanhar em uma janela: `bash experiments/watch_vctk100_xvector.sh`.
Quando o estado for `complete`, gerar o relatório com
`.venv/bin/python experiments/build_vctk_results_pdf.py` e verificar visualmente
as 26 páginas. O gerador exige todos os 20 modelos e os oito resumos x-vector
com cinco folds cada.

## Conclusão

A rodada terminou em 01/10/2026: 20 modelos, 40 avaliações e oito combinações
com cinco folds completos. Os resultados finais estão na cópia gravável em
`output/vctk100_xvector_silero_20260929/results`. Os PDFs para apresentação e
detalhamento foram regenerados em `output/pdf/` e verificados visualmente.
