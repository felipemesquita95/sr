# Corpora

Dois corpora são usados, e a diferença entre eles é o eixo central do trabalho:
cada um carrega um confundidor distinto entre identidade do locutor e condição de
gravação.

## BrSD — Brazilian Speech Database

Publicado por Paulino et al. (ICTAI, 2018), Universidade Estadual de Maringá.

| Propriedade | Valor |
|---|---|
| Locutores | 80 |
| Enunciados por locutor | 5 |
| Total de gravações | 400 |
| Duração | 2 a 3 minutos por gravação |
| Taxa de amostragem | 48 kHz |
| Texto | **Dependente** — os mesmos 5 trechos para todos |
| Idioma | Português brasileiro |

Os cinco textos são trechos literários: *História do Brasil* (Frei Vicente do
Salvador), *Dom Quixote* (Cervantes), *O Pequeno Príncipe* (Saint-Exupéry), *Meu Pé
de Laranja Lima* (José Mauro de Vasconcelos) e *Marcelo, Marmelo, Martelo* (Ruth
Rocha).

### O confundidor: um dispositivo por locutor

O ponto decisivo está na coleta. **Cada contribuinte gravou as suas cinco leituras
no próprio aparelho** — tipicamente um telefone celular, em ambiente não controlado.
A consequência é estrutural:

> A assinatura de canal — resposta em frequência do microfone, ruído de fundo do
> ambiente, ganho, codec de compressão — é **constante dentro de cada locutor e
> distinta entre locutores**. Ela é, portanto, um identificador perfeito do locutor,
> independentemente da voz.

Nenhuma escolha de partição corrige isso. Identificação de locutor em conjunto
fechado exige que cada locutor apareça tanto em treino quanto em teste; logo, o seu
dispositivo aparece nos dois lados por construção. O artigo original evita o problema
nas suas próprias tarefas — classificação de idade, gênero e sotaque — mantendo todas
as gravações de uma pessoa na mesma partição, estratégia impossível aqui.

O índice do enunciado, de 1 a 5, identifica **o texto**, não uma sessão independente
de gravação. Reter o texto *k* de todos os locutores na partição *k*, como faz o
protocolo adotado, elimina a memorização de conteúdo linguístico como explicação —
mas não toca no confundidor de dispositivo.

## VCTK — Voice Cloning Toolkit, versão 0.92

Centre for Speech Technology Research, Universidade de Edimburgo.

| Propriedade | Valor |
|---|---|
| Locutores | 110 |
| Enunciados por locutor | cerca de 400 (200 utilizados) |
| Duração | poucos segundos por enunciado |
| Taxa de amostragem | 48 kHz |
| Texto | **Independente** — textos distintos por locutor |
| Microfones | **Dois, simultâneos** (`mic1` omnidirecional, `mic2` condensador) |

### O que o VCTK resolve

Todos os locutores foram gravados na mesma sala e com o mesmo equipamento. O modelo
do microfone é, portanto, constante entre classes, e deixa de ser preditivo do
rótulo. O corpus também é texto-independente, removendo a memorização de conteúdo.

### O que o VCTK não resolve

Cada locutor foi gravado em **uma única sessão**. Tudo o que for particular daquela
sessão — nível de ganho ajustado no dia, distância e postura em relação ao microfone,
ruído de fundo do momento, respiração — permanece constante dentro do locutor e
variável entre locutores. A forma do confundidor muda de *dispositivo* para *sessão*,
mas a estrutura é a mesma.

### O que o VCTK permite medir

A gravação simultânea por dois microfones é a propriedade que torna este corpus
valioso para o trabalho. Para uma mesma frase, as duas trilhas têm voz, texto e
instante idênticos, diferindo apenas no transdutor. Isso viabiliza dois protocolos
que nenhum corpus de trilha única permite:

- **Cross-microfone** — treinar em uma trilha e avaliar na outra. Um modelo apoiado
  na assinatura do microfone de treino não a encontra no teste.
- **Multi-microfone** — treinar com ambas, particionando por enunciado. Cada locutor
  passa a ser visto através de dois canais, descorrelacionando canal e rótulo.

## Obtenção

O BrSD é distribuído pelos autores mediante solicitação. O VCTK 0.92 está disponível
publicamente no repositório da Universidade de Edimburgo (DOI 10.7488/ds/2645).

Nenhum dos dois é versionado neste repositório. Os perfis em `configs/` apontam para
os caminhos locais através de `AUDIO_PATH` e `VCTK_ROOT`.
