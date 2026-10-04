# Retomada — realinhamento do VCTK ao perfil do BrSD

Estado em **17 de setembro de 2026, 16h41**. Prazo da apresentação: **20h30**.

Este arquivo existe para que a sessão seguinte continue sem reconstruir contexto.
Leia-o inteiro antes de rodar qualquer coisa: há uma ordem, e ela existe por motivo.

---

## A decisão que originou tudo

O BrSD foi feito primeiro, a 8 kHz, e o perfil de referência dele reproduz o
experimento original. O VCTK entrou depois, a 16 kHz, com VAD ligado e janela de 512.
Comparar os dois corpora sob parâmetros diferentes **confunde corpus com
pré-processamento** — os 83,00% do BrSD contra os 97,56% do VCTK não são uma
comparação de corpora.

A decisão do autor: **o VCTK se alinha ao BrSD**, não o contrário. Reextrair o VCTK a
8 kHz, janela 256, VAD desligado, pré-ênfase 0,97.

Sub-decisões, já tomadas, não reabrir:

| questão | decisão |
|---|---|
| Pré-ênfase | **0,97**, igual ao BrSD (não desligar) |
| VAD | **desligado**, igual ao perfil de referência do BrSD |
| Features antigas | podem ser apagadas |
| Resultados antigos (`runs/models/`) | **nunca apagar** — são a apresentação |

---

## O que já foi feito

1. **`configs/vctk8k.env` e `configs/vctk8k_mic2.env`** criados. Os nove parâmetros de
   DSP batem exatamente com `configs/brsd.env`. Verificar a qualquer momento com:

   ```bash
   for k in SOURCE_SAMPLING_RATE TARGET_SAMPLING_RATE ENABLE_VAD VAD_TOP_DB \
            PRE_EMPHASIS_COEF NUM_MFCCS FRAME_SIZE NUM_FOLDS VALIDATION_SEED; do
     printf "%-24s brsd=%-8s vctk8k=%s\n" "$k" \
       "$(grep -oP "(?<=^$k=).*" configs/brsd.env)" \
       "$(grep -oP "(?<=^$k=).*" configs/vctk8k.env)"
   done
   ```

2. **`runs/features/vctknovad_mic{1,2}` apagados** (2,2 GB). Não tinham assinaturas de
   canal, e a variante sem VAD a 16 kHz perdeu o sentido: a nova extração já é sem VAD.
   Os resultados correspondentes seguem em `runs/models/vctk_cross_mic_novad/`.

3. **Download do zip do VCTK em andamento.** O corpus tinha sido apagado do disco pela
   ingestão original (`/home/lsmsqt/datasets/vctk/` estava vazio).

4. **Apresentação** — `docs/apresentacao.md` → `docs/apresentacao.pdf` (6 páginas), com
   as seções *Datasets* e *Pré-processamento* prontas, e 7 figuras geradas pelo pipeline
   real. Ferramentas novas: `gerar_pdf.py` e `figuras_cadeia.py`.

---

## O que está rodando agora

```
curl  PID 11146   →  /home/lsmsqt/datasets/vctk/VCTK-Corpus-0.92.zip
log:  runs/download_vctk_8k.log
```

Às 16h41 estava em **5,08 GB de 10,9 GB (45%)**, a ~4,5 MB/s, terminando por volta das
**17h05**. O download é retomável: se cair, repetir o mesmo comando com `-C -`.

```bash
cd /home/lsmsqt/datasets/vctk && curl -L -C - --retry 5 --retry-delay 10 \
  -o VCTK-Corpus-0.92.zip \
  "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
```

Conferir progresso:

```bash
ls -l --block-size=M /home/lsmsqt/datasets/vctk/VCTK-Corpus-0.92.zip
tail -c 200 runs/download_vctk_8k.log | tr '\r' '\n' | tail -1
```

---

## O que falta, na ordem

### Passo 1 — verificar o zip

Só seguir se isto passar. O zip completo tem **10,9 GB**.

```bash
python3 -c "
import zipfile
z = zipfile.ZipFile('/home/lsmsqt/datasets/vctk/VCTK-Corpus-0.92.zip')
print('entradas:', len(z.namelist()))
print('corrompido em:', z.testzip())"
```

### Passo 2 — ingerir as duas trilhas a 8 kHz

Um locutor por vez: extrai do zip, converte em MFCCs, apaga o áudio, segue. As
assinaturas de canal são gravadas na mesma passada — **não** usar `--no-signatures`.

```bash
KERAS_BACKEND=torch .venv/bin/python experiments/ingest_vctk.py \
  --config configs/vctk8k.env --mics mic1,mic2 --prefix vctk8k \
  2>&1 | tee runs/vctk8k_ingestao.log
```

Saída esperada: `runs/features/vctk8k_mic1` e `runs/features/vctk8k_mic2`, **21.523
gravações cada**, ~550 MB por trilha.

### Passo 3 — apagar as features de 16 kHz e o zip

**Só depois que o passo 2 terminar**, porque `vctk_mic{1,2}` são as únicas cópias das
43.046 assinaturas de canal e o áudio de origem some de novo na ingestão.

```bash
# conferir que as novas assinaturas existem antes de apagar as velhas
find runs/features/vctk8k_mic1 -name assinaturas.npz | wc -l   # esperado: 21523
find runs/features/vctk8k_mic2 -name assinaturas.npz | wc -l   # esperado: 21523

rm -rf runs/features/vctk_mic1 runs/features/vctk_mic2
rm -f /home/lsmsqt/datasets/vctk/VCTK-Corpus-0.92.zip           # devolve 11 GB
```

**Nunca apagar** `runs/features/vctk_manifesto.json` nem
`runs/features/vctk_speaker_info.txt`: ficam na raiz de `runs/features/`, não são
features, e são usados por `transfer_matrix.py`, `error_structure.py`,
`verify_alignment.py` e pela UI.

### Passo 4 — reavaliar o `MAX_FRAMES_CAP` antes de treinar

**Este passo não é opcional.** O valor atual, 300, foi calibrado para 16 kHz *com* VAD,
onde o p95 era 299 quadros. A 8 kHz e sem VAD a distribuição é outra: o salto dobra de
duração relativa e as pausas voltam para o sinal. Truncar no valor errado enviesa a
comparação inteira, e é exatamente o tipo de viés que `EXPERIMENTOS.md` já denuncia em
"Truncamento assimétrico".

A ingestão grava o resumo no log. Extrair o p95 e ajustar `MAX_FRAMES_CAP` nos dois
perfis se divergir muito de 300:

```bash
grep "Resumo do conjunto" runs/vctk8k_ingestao.log
```

### Passo 5 — treinar, na ordem de valor decrescente

Se o tempo acabar, o que estiver pronto já serve; o que não rodou não deixa o autor sem
apresentação, porque os resultados a 16 kHz continuam todos em `runs/models/`.

```bash
# 1. intra-microfone (o mais rápido, e o que compara direto com o BrSD)
KERAS_BACKEND=torch SR_CONFIG=configs/vctk8k.env \
  .venv/bin/python experiments/run_experiment.py 2>&1 | tee runs/vctk8k_intra.log

# 2. cross-microfone — exige criar configs/vctk8k_cross_mic.env a partir de
#    configs/vctk_cross_mic.env, trocando taxa/janela/VAD e os caminhos para vctk8k_*
```

---

## Armadilhas registradas

- **O áudio do VCTK não sobrevive à ingestão.** Ela apaga cada arquivo após converter.
  Qualquer medida que dependa da forma de onda tem de ser tomada naquela passada — por
  isso as assinaturas são calculadas ali.
- **As assinaturas não dependem do VAD nem da taxa de destino.** `signatures_of` roda o
  próprio detector a 48 kHz, com `VAD_TOP_DB=30`. Desligar o VAD do perfil não afeta o
  diagnóstico de silêncio; as novas assinaturas saem idênticas às antigas.
- **A numeração sai da interseção das trilhas.** `p280` e `p315` não têm `mic2`; são
  108 locutores, não 110. Não mexer em `VCTK_MICS`.
- **Disco.** 11 GB livres às 16h41, com o zip ainda crescendo. O aperto se desfaz
  sozinho no passo 3.

---

## Prompt para a sessão seguinte

Colar isto:

> Estou retomando o realinhamento do VCTK ao perfil do BrSD no projeto em
> `/home/lsmsqt/Documents/sr`. Leia `RETOMAR.md` na raiz do repositório: ele tem o
> estado, as decisões já tomadas e os passos numerados. Continue do ponto em que a
> ingestão estiver.
>
> Confira primeiro se o download do zip terminou e se a ingestão a 8 kHz
> (`runs/vctk8k_ingestao.log`) rodou ou não, em vez de presumir. Não reabra as decisões
> de parâmetro já registradas: pré-ênfase 0,97, VAD desligado, 8 kHz, janela 256.
> Nunca apague `runs/models/` — é a apresentação. Apagar features é permitido, mas só na
> ordem descrita no passo 3.
>
> Antes de treinar qualquer coisa, execute o passo 4: o `MAX_FRAMES_CAP=300` foi
> calibrado para 16 kHz com VAD e precisa ser reavaliado.
>
> Tenho até as 20h30. Se algo não couber no tempo, me diga o que ficou de fora em vez de
> reduzir o escopo silenciosamente.
