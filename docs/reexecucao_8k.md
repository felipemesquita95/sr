# Reexecução BrSD × VCTK a 8 kHz

## Arquivos e estado

- Áudios BrSD: `/media/lsmsqt/HDD/datasets/brsd/utterances/` (400 WAVs) e `utterances.zip`.
- Áudios VCTK: `/media/lsmsqt/HDD/datasets/vctk/VCTK-Corpus-0.92/wav48_silence_trimmed/` e ZIP original.
- Features e modelos: `runs/features` e `runs/models`, links para `/media/lsmsqt/HDD/sr_project/`.
- Relatório final, após todos os treinos: `docs/relatorio_8k.md` e `docs/relatorio_8k.pdf`.
- Duração de cada gravação: `runs/features/relatorio_comprimentos_8k.csv`.
- Uso de cada gravação em cada partição: `runs/features/relatorio_particoes_8k.csv` (gerado no relatório final).

Os arquivos do BrSD 106–110 têm taxa nativa de 44,1 kHz; os outros 395 são 48 kHz.
Todos os áudios VCTK selecionados são 48 kHz. O carregamento preserva a taxa
nativa, de modo que o filtro ocorre antes da reamostragem em todos os arquivos.

## Cadeia de extração, igual nos dois corpora

1. Carregar mono na taxa nativa, sem VAD.
2. Filtrar com Butterworth de ordem 8, fase zero, corte de 3,6 kHz.
3. Reamostrar com `resample_poly` para 8 kHz.
4. Aplicar pré-ênfase de 0,97.
5. Extrair 40 MFCCs por quadro de 256 amostras (32 ms), salto 128 (16 ms).
6. Salvar as matrizes completas em `float32` no HDD.

As três redes (CNN, Temporal CNN e Attention) usam as mesmas cinco partições,
normalização calculada apenas no treino, lote 64, Adam com taxa inicial de 0,001
e parada antecipada após 30 épocas sem melhora na validação. Para a largura de
entrada, usa-se o menor número de quadros observado no corpus. O BrSD tem 1.007
quadros no menor arquivo; o VCTK tem 77 quadros previstos pelos metadados dos
FLACs, a confirmar pelos MFCCs. Os dois microfones VCTK usarão o mesmo mínimo.

## Como acompanhar

```bash
systemctl --user status sr-brsd-nativo.service sr-rebuild-8k-minimo.service sr-final-report-8k-nativo.service
tail -f runs/brsd_nativo_2026-09-23.log
tail -f runs/rebuild_8k_minimo_2026-09-23.log
```

O serviço `sr-final-report-8k-nativo` espera os treinos terminarem e gera o PDF.
Se o HDD for desmontado, os serviços não conseguem continuar; mantenha-o montado
durante a execução.
