# Prioridade 5 — referência linear sobre os mesmos MFCCs

**Objetivo.** Implementar e executar a prioridade 5 de `docs/ideias.md`: uma
referência estática linear sobre exatamente os mesmos tensores usados pela
matriz de transferência, para testar se os ~38% do cross são um teto do
protocolo ou uma propriedade das redes e do regime de ajuste.

**Prompt entregue ao Codex** (`gpt-5.6-terra`, esforço medium, `codex exec`,
sandbox `workspace-write`): ver `runs/codex_p5.log`. O prompt fixava os
controles que o documento exige — não substituir os tensores pelas assinaturas
de 48 kHz, grade de regularização definida antes de olhar resultado, artefatos
guardando divisão, sementes, normalização e predições por gravação — além das
convenções do repositório.

**Estado no momento do despacho.** Prioridades 1, 2 e 3 implementadas e
executadas (`error_structure.py`, `transfer_matrix.py`,
`session_or_transducer.py`). Prioridade 4 pendente. Árvore limpa em `186ca43`.

## O que foi incorporado

Tudo, no commit `a333240`, com duas intervenções minhas:

- Correção de formatação: a seção 2.5 havia sido inserida depois do separador
  que fecha a seção 2, e foi movida para dentro dela.
- O commit foi feito por mim. O Codex não conseguiu criar o seu: o sandbox
  montou `.git` como somente leitura e `git add` falhou no `index.lock`.

Verifiquei a suíte de forma independente: **169 testes passam**. Conferi que a
comparação é pareada — mesma divisão (16.138 / 1.080 / 4.305), mesma semente de
divisão 42, mesmas gravações de teste da matriz.

## O achado

A regressão logística supera as três sementes da CNN em **todas as quatro
células**, e a margem é pequena no intra (+2,5 pp) e grande no cross
(+15 a +19 pp).

Isso confirma sob condições pareadas algo que a seção 3 do `docs/resultados.md`
já registrava sem controle equivalente: o mesmo classificador de 80 números
atingia 55,38% no cross enquanto as redes ficavam em 38%. A prioridade 5 removeu
a diferença de taxa de amostragem e de recorte que impedia a comparação.

## Decisão do modelo que o documento não especificava

A grade de regularização. O Codex fixou `C = (0,1; 1; 10)` antes de executar,
com desempate preservando o primeiro valor. A validação da origem selecionou
`C = 10` nas duas direções.

## Alcance

O experimento não atribui a vantagem a coeficiente, banda ou ausência de ordem
temporal, e não estabelece que a ordem temporal seja irrelevante para outros
modelos. A penalidade de canal permanece grande também para o modelo linear
(46,69 e 40,67 pontos). O padrão de ganhar pouco no intra e muito no cross é
compatível com as redes aprendendo detalhe fino específico da trilha, mas isso
é **hipótese**, não resultado desta medição.
