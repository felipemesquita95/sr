# Sequência experimental — VCTK

Runbook de trabalho. Registra o que já está pronto, o que roda em seguida e o que
cada passo decide. Não é documentação do sistema: para isso, ver `docs/`.

Última atualização: 2026-09-04.

---

## Estado atual

### Pronto e verificado

| item | estado |
|---|---|
| Features VCTK **sem** silêncio (VAD ligado) | `runs/features/vctk_mic{1,2}` — 21.523 cada, 1,9 GB |
| Features VCTK **com** silêncio (VAD desligado) | `runs/features/vctknovad_mic{1,2}` — 21.523 cada, 2,2 GB |
| Assinaturas de canal (3 condições, áudio bruto) | `runs/features/vctk_mic{1,2}/*/*/assinaturas.npz` |
| Numeração locutor↔trilha | verificada: recuperação top-1 **100/100** |
| GPU | RX 7800 XT ativa via torch-ROCm, Keras no backend torch |
| Suíte de testes | 104 passando |

### O que ainda não existe

**Nenhum experimento de classificação rodou no VCTK.** Todos os números de VCTK em
`docs/resultados.md` são da implementação anterior, de junho, e a seção
cross-microfone daquele documento está comprometida (ver abaixo).

---

## Ambiente

O TensorFlow foi removido: o repositório usa Keras 3, que roda sobre PyTorch, e é o
PyTorch que tem suporte ROCm para a placa AMD.

```bash
export KERAS_BACKEND=torch
```

Sem essa variável o Keras não encontra backend algum e falha na importação.
`HSA_OVERRIDE_GFX_VERSION` **não** é necessário — a `gfx1101` é reconhecida direto.

Conferir a GPU:

```bash
KERAS_BACKEND=torch .venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Avisos do `MIOpen(HIP)` sobre *workspace* durante o treino são benignos.

### Disco

**3,3 GB livres.** O zip do VCTK foi apagado para caber o torch (14 GB). Se algum
experimento precisar do áudio bruto de novo — outra taxa de amostragem, outro limiar
de VAD, aumento de canal —, são 16 minutos de download:

```bash
curl -L -C - -o /home/lsmsqt/datasets/vctk/VCTK-Corpus-0.92.zip \
  https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip
```

Só que hoje **não há espaço** para ele. Montar o HDD de 1 TB (`/dev/sda3`, NTFS, não
montado) resolveria de vez:

```bash
sudo mkdir -p /mnt/hdd
sudo mount -t ntfs-3g -o uid=$(id -u),gid=$(id -g) /dev/sda3 /mnt/hdd
```

Atenção: se houver dual boot com *Fast Startup* ligado, o NTFS pode estar sujo e
montar para escrita pode corromper. O `ntfs-3g` avisa; não force.

---

## O que foi corrigido, e por que importa

O VCTK 0.92 não tem trilha `mic2` para `p280` e `p315`, que ocupam as posições **54 e
84** de 110 na ordenação. A indexação numerava cada trilha isoladamente, então a
partir da posição 54 as duas numerações deslocavam.

Consequência medida: **57 dos 110 locutores recebiam rótulo trocado** no protocolo
cross-microfone, e o teto de acurácia possível era **48,2%**, não 100%. A
implementação anterior tem o mesmo defeito, verificado no código.

Os 24,09% reportados em `docs/resultados.md` foram medidos contra esse teto. A
correção está em `sr/datasets/index.py`: a numeração passa a sair da interseção das
trilhas declaradas em `VCTK_MICS`, e sobram 108 locutores (acaso 0,93%).

Sobrevivem intactos os números intra-microfone e o do só-silêncio — usam uma trilha
só, onde a numeração é consistente consigo mesma.

---

## Sequência

Ordem deliberada: os controles baratos vêm antes dos treinos caros, porque podem
mudar o que vale treinar.

### 1. Diagnóstico de canal atravessando o microfone

**Pergunta:** as pistas que identificam o locutor **fora da fala** sobrevivem à troca
de transdutor?

**Por que é o primeiro:** o cross-microfone troca o microfone mas mantém a sessão —
as duas trilhas são simultâneas. Se as pistas de sessão atravessarem a troca, a
acurácia que sobra no cross-microfone **não é voz isolada**, e o número deixa de ser
um piso honesto para virar um teto. Isso muda a conclusão central antes de qualquer
treino.

Roda em CPU, em minutos, sobre as assinaturas já extraídas.

> Instrumento ainda não escrito.

### 2. Controle de permutação

Embaralhar os rótulos e confirmar que a acurácia cai ao acaso. Verifica que o
arcabouço não vaza. Barato, e é o que um parecerista pede primeiro.

> Ainda não escrito.

### 3. Cross-microfone corrigido

O número que substitui os 24,09%. Rodar nas **duas** variantes:

```bash
KERAS_BACKEND=torch SR_CONFIG=configs/vctk_cross_mic.env .venv/bin/python experiments/run_experiment.py
```

Comparar com silêncio contra sem silêncio separa duas causas hoje misturadas: quanto
da queda é o canal, e quanto é o VAD tendo cortado trechos diferentes em cada trilha
(medido: só 212 de 6.200 gravações mantêm a mesma duração entre as trilhas,
divergência mediana de 11,5%).

> `configs/vctk_cross_mic.env` ainda aponta para os diretórios com VAD. Falta o perfil
> equivalente sobre `vctknovad_*`.

### 4. Intra-microfone e multi-microfone

Completam a tabela, e só são comparáveis entre si porque agora compartilham a
numeração de 108 locutores.

### 5. Busca de arquiteturas

Seis arquiteturas registradas, organizadas como três pares de ablação:

| par | isola |
|---|---|
| `cnn` ↔ `temporal_cnn` | o eixo da convolução |
| `temporal_cnn` ↔ `temporal_cnn_stats` | a agregação (média contra média+desvio) |
| `xvector` ↔ `xvector_attentive` | a ponderação dos quadros |

Contagem de parâmetros com entrada `(40, 300)` e 108 classes:

| arquitetura | parâmetros |
|---|---|
| `temporal_cnn` | 114.732 |
| `temporal_cnn_stats` | 147.500 |
| `cnn` | 369.548 |
| `xvector` | 1.203.564 |
| `xvector_attentive` | 1.401.068 |
| `attention` | 1.647.340 |

---

## Decisões em aberto

### Seleção sem vazamento

Escolher a melhor arquitetura pela acurácia de **teste** é a mesma classe de
vazamento que `docs/limitacoes.md` denuncia, um nível acima: o número reportado vira
o pico sobre o teste. A seleção tem de ser pela **validação**, com o teste lido uma
vez só, para a arquitetura escolhida.

### "Melhor" segundo qual protocolo

Sob o protocolo intra-microfone, a vencedora é a que melhor explora o canal — que é
justamente o que o trabalho critica. Sob o cross-microfone, é a que transfere entre
transdutores. Se as duas discordarem, a discordância é resultado, não problema: *a
arquitetura que vence o benchmark convencional não é a que melhor identifica voz.*

### Capacidade desbalanceada

14× entre a menor e a maior. Os pares de ablação são limpos; a tabela inteira lida
como ranking, não é. No mínimo, publicar a contagem de parâmetros ao lado de cada
resultado.

### Truncamento assimétrico

`MAX_FRAMES_CAP=300` nas duas variantes. Com VAD o p95 é 299, então quase nada é
truncado; sem VAD o p95 é 406, e perde-se bastante. A comparação entre variantes
carrega esse viés e precisa dizê-lo.

### Uma ressalva sobre a agregação atenta

`xvector_attentive` pode aprender a **privilegiar** os quadros de silêncio, se for de
lá que a identidade se prediz melhor. Não é defesa contra o confundidor de canal — é
um mecanismo mais capaz de explorá-lo. Se vencer, interpretar com cuidado.

---

## Depois de rodar

`docs/resultados.md` precisa ser reescrito. A seção cross-microfone e a síntese estão
construídas sobre números que a correção de numeração invalidou. A conclusão
qualitativa — parte grande da acurácia é canal, não voz — não depende disso; a
magnitude, que é a manchete, depende inteiramente.
