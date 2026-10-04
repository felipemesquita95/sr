# Retomar CNN, temporal e RASTA — 28/09/2026

O usuário autorizou explicitamente toda a sequência, na GPU e com lote 128.
Não pedir nova confirmação para executar estas etapas. A x-vector com referência
e CMN já tem 20/20 treinos concluídos e deve ser preservada.

## Fila automática

1. CNN com CMN: cinco folds × dois microfones de origem (10 treinos).
2. CNN referência, lote 128: 10 treinos.
3. CNN temporal com CMN: 10 treinos.
4. CNN temporal referência, lote 128: 10 treinos.
5. Reextrair RASTA-MFCC na mesma coorte/janelas.
6. X-vector, CNN e CNN temporal com RASTA: 10 treinos por rede.

Total: 70 treinos novos. Cada modelo recebe os dois testes pareados. Não mudar
coorte, folds, lote ou rede entre referência e intervenção. RASTA é uma condição
separada, sem combinação com CMN. O roteiro segue para as etapas independentes
após uma falha, registra o erro e bloqueia as etapas RASTA se sua extração falhar.

Para iniciar no computador com acesso à GPU e ao serviço do usuário:

```bash
cd /home/lsmsqt/Documents/sr
bash experiments/start_vctk16_channel_suite.sh
```

Unidade: `sr-vctk16-channel-suite.service`. Log ao vivo é aberto pelo lançador.
Status: `output/vctk16_channel_suite_status.json`.
Relatório atualizado após cada etapa: `output/vctk16_channel_suite_comparison.md`.
Resultados e features RASTA ficam em `/media/lsmsqt/HDD/sr_project`.
Treinos concluídos são saltados; os incompletos retomam do melhor checkpoint.

## Bloqueio da sessão atual

O perfil de permissões mudou para restrito. O processo de execução não tem
`/dev/kfd` ou `/dev/dri/renderD128`, o PyTorch retorna GPU indisponível, e o
acesso ao barramento do systemd é negado. Escrita no HDD também não está liberada.
Em 28/09 a fila **não foi iniciada**. O status `blocked_no_gpu` reflete esse
diagnóstico. Restaurar o acesso ao computador e executar o lançador acima;
não treinar na CPU nem tentar contornar a restrição.

## RASTA verificado

Filtro causal no log-mel completo antes da DCT:
`b=[0.2,0.1,0,-0.1,-0.2]`, `a=[1,-0.94]`.
Inicialização estacionária a partir do primeiro quadro para não criar transiente
por preenchimento com zero. 128 bandas mel, 30 MFCCs, Hamming 32 ms, hop 16 ms,
center=False, pré-ênfase 0,97, mesma decimação 48→16 kHz e limites de trim.
Os coeficientes canônicos são mantidos no hop de 16 ms: a resposta de modulação
física difere da versão usual com hop de 10 ms; isso fica no manifesto.

Mantêm-se os índices de janela da referência, sem recalcular seleção RMS após
filtragem. O contexto anterior do próprio áudio influencia o filtro causal.
Deltas e delta-deltas são recalculados apenas nos 111 quadros retidos. Não é
RASTA-PLP. Fonte: [Hermansky e Morgan (1994)](https://labrosa.ee.columbia.edu/~dpwe/papers/HermM94-rasta.pdf).

Verificações realizadas: sintaxe Python/shell; cancelamento de offset espectral
constante; reprodução dos MFCCs originais antes de RASTA nos dois canais de
`p225_241`; formas 30×111 e ausência de valores inválidos; deltas corretos;
retomada da extração sem reescrever features válidas; recusa de treino sem GPU.
