"""Preparação de conteúdo fora da thread da janela.

Os carregadores entregam arrays, metadados e QImage, nunca widgets.
Separar leitura e apresentação permite descartar uma resposta antiga
antes de desenhá-la quando a seleção muda durante uma tarefa.
"""
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtGui import QImage

from sr.features import FeatureAdjustmentSubsystem
from ui import dados


@dataclass(frozen=True)
class Selection:
    """Captura o contexto de uma solicitação sem consultar os seletores depois.

    A dataclass congelada impede reatribuir seus campos; os dicionários e a
    configuração são compartilhados e devem ser tratados como somente leitura.

    Attributes:
        track: Diretório da trilha.
        speaker: Locutor em base um.
        utterance: Enunciado em base um.
        inventory: Cobertura de gravações e figuras.
        manifest: Manifesto do corpus, quando disponível.
        names: Nomes dos locutores por índice.
        profile: Perfil associado à trilha, se encontrado.
        settings: Configuração desse perfil, ou ausência.
    """
    track: Path
    speaker: int
    utterance: int
    inventory: dict
    manifest: dict
    names: dict
    profile: Path | None
    settings: object

    @property
    def sample(self):
        return self.track / str(self.speaker) / str(self.utterance)

    @property
    def key(self):
        """Identifica a gravação ao conferir respostas assíncronas.

        A janela acrescenta revisão e parâmetros da página à chave de carregamento.

        Returns:
            Tupla com caminho da trilha, locutor e enunciado.
        """
        return str(self.track), self.speaker, self.utterance

    @property
    def caption(self):
        return f'{self.track.name} · {self.names.get(self.speaker, f"Locutor {self.speaker}")} · enunciado {self.utterance:03d}'


STAGES = [
    ('corpus', 'Visão geral', 'Conheça a distribuição das gravações e as condições que definem o experimento.'),
    ('sinal', 'Sinal bruto', 'A forma de onda e o espectro mostram a gravação antes das transformações.'),
    ('vad', 'Atividade vocal', 'A detecção por energia remove trechos abaixo de um limiar relativo ao pico de cada trilha.'),
    ('filtragem', 'Filtragem e reamostragem', 'Filtrar antes de reduzir a taxa evita que altas frequências contaminem a banda útil.'),
    ('preenfase', 'Pré-ênfase', 'Um filtro de primeira ordem realça as altas frequências antes da extração de características.'),
    ('mfcc', 'Coeficientes MFCC', 'Explore a representação que as redes recebem e compare a mesma frase entre microfones.'),
    ('assinatura', 'Assinatura de canal', 'Estatísticas cepstrais revelam pistas associadas ao locutor nos diferentes recortes do sinal.'),
    ('tensores', 'Montagem dos tensores', 'Inspecione a divisão real dos conjuntos, o alinhamento e as garantias contra vazamento.'),
    ('protocolos', 'Protocolos experimentais', 'Cada protocolo responde a uma pergunta diferente sobre o que o sistema aprendeu.'),
    ('treino', 'Arquiteturas e treino', 'Configure um experimento e acompanhe sua evolução enquanto continua explorando os dados.'),
    ('resultados', 'Resultados', 'Compare métricas persistidas, curvas de aprendizado e erros por locutor.'),
    ('defesa', 'Roteiro da defesa', 'Siga o argumento do trabalho e abra cada evidência diretamente.'),
    ('evidencias', 'Evidências diagnósticas', 'Experimentos, controles e ressalvas lidos dos artefatos locais.'),
]


def image(path, title):
    """Decodifica o PNG em QImage para transferi-lo sem criar recursos de widget.

    A conversão para QPixmap fica na thread da janela. Uma imagem nula mantém
    o caminho e o título para que a página explique a ausência do artefato.

    Args:
        path: Caminho esperado do PNG.
        title: Legenda da figura.

    Returns:
        Tupla de título, QImage e caminho de origem.
    """
    value = QImage(str(path)) if path.is_file() else QImage()
    return title, value, path


def load_stage(stage, ctx, settings=None, fold=1):
    """Reúne somente os artefatos necessários à etapa solicitada.

    Nenhum estágio é reconstruído a partir de áudio ausente. A comparação
    entre microfones exige manifesto e o MFCC correspondente na outra trilha.

    Args:
        stage: Identificador do catálogo de etapas.
        ctx: Seleção capturada antes de iniciar a tarefa.
        settings: Perfil escolhido para reconstruir tensores.
        fold: Partição a inspecionar, em base um.

    Returns:
        Dicionário de dados para a página, com imagens decodificadas e campos por etapa.

    Raises:
        ValueError: Se faltar perfil para os tensores ou os artefatos forem inválidos.
        OSError: Se uma matriz necessária não puder ser lida.
    """
    payload = {'images': []}
    files = {
        'sinal': [('sinal_original.png', 'Forma de onda original'), ('espectro_original.png', 'Espectro original')],
        'vad': [('sinal_original.png', 'Antes do VAD'), ('sinal_vad.png', 'Depois do VAD')],
        'filtragem': [('espectro_filtrado.png', 'Após filtro anti-aliasing'), ('espectro_reamostrado.png', 'Após reamostragem')],
        'preenfase': [('espectro_reamostrado.png', 'Antes da pré-ênfase'), ('espectro_preenfase.png', 'Depois da pré-ênfase')],
    }
    if stage in files:
        selected = files[stage]
        if stage == 'vad' and ctx.settings and not ctx.settings.enable_vad:
            selected = [('sinal_original.png', 'Sinal preservado • VAD desativado')]
        payload['images'] = [image(ctx.sample / name, title) for name, title in selected]
        payload['available_figures'] = [u for u in ctx.inventory[ctx.speaker]
            if all((ctx.track / str(ctx.speaker) / str(u) / name).is_file() for name, _ in selected)]
        if stage == 'sinal' and any(value.isNull() for _, value, _ in payload['images']):
            source = dados.audio_source(ctx.settings, ctx.speaker, ctx.utterance, ctx.manifest, ctx.track.name)
            if source:
                payload['raw'] = dados.raw_signal(source)
                payload['images'] = []
        if stage == 'vad':
            payload['alignment'] = dados.read_json(dados.RUNS / 'models/verificacao_alinhamento/alinhamento.json')
    elif stage == 'corpus':
        payload['images'] = [image(ctx.track / '_resumo' / name, title) for name, title in
                             [('duracao.png', 'Duração das gravações'), ('enunciados_por_locutor.png', 'Gravações por locutor')]]
    elif stage == 'mfcc':
        payload['matrices'] = [dados.mfcc(ctx.sample / 'mfccs.npy')]
        payload['titles'] = [ctx.track.name]
        if ctx.track.name.endswith(('_mic1', '_mic2')) and ctx.manifest:
            partner = ctx.track.with_name(ctx.track.name[:-1] + ('2' if ctx.track.name.endswith('1') else '1'))
            file = partner / str(ctx.speaker) / str(ctx.utterance) / 'mfccs.npy'
            if file.is_file():
                payload['matrices'].append(dados.mfcc(file))
                payload['titles'].append(partner.name)
    elif stage == 'assinatura':
        payload['signatures'] = dados.signatures(ctx.sample / 'assinaturas.npz')
        payload['transfer'] = dados.read_json(dados.RUNS / 'models/vctk_cross_mic/diagnostico_travessia/travessia_canal.json')
    elif stage == 'tensores':
        if settings is None:
            raise ValueError('Escolha um perfil disponível para inspecionar a divisão.')
        split, frames = dados.inspect_split(settings, fold)
        matrix = dados.mfcc(ctx.sample / 'mfccs.npy')
        payload.update(split=split, frames=frames, settings=settings, fold=fold,
                       matrices=[matrix, FeatureAdjustmentSubsystem.pad_or_truncate(matrix, frames)])
    elif stage == 'resultados':
        payload['metrics'], payload['errors'] = dados.metrics(dados.RUNS / 'models')
    elif stage == 'defesa':
        payload['resultados'] = dados.texto_documento('resultados.md')
    elif stage == 'evidencias':
        payload['diagnosticos'] = dados.diagnosticos(dados.RUNS / 'models')
        payload['resultados'] = dados.texto_documento('resultados.md')
        payload['limitacoes'] = dados.texto_documento('limitacoes.md')
    return payload


def model_parameters(shape, classes, rate):
    # Importar/compilar as redes ocorre apenas após solicitação e fora da janela.
    """Calcula contagens para a forma escolhida sem ocupar a GPU do treino.

    Keras e as redes só são importados sob demanda. Cada modelo é descartado
    antes do próximo, evitando acumular as seis arquiteturas na memória.

    Args:
        shape: Forma de entrada sem o eixo de lote.
        classes: Número de locutores de saída.
        rate: Taxa de aprendizado usada na construção do modelo.

    Returns:
        Pares de nome de arquitetura e número de parâmetros.
    """
    import torch
    from keras import backend
    from sr.models import ARCHITECTURES, build_model
    result = []
    with torch.device('cpu'):
        for architecture in ARCHITECTURES:
            model = build_model(architecture, shape, classes, rate)
            result.append((architecture, model.count_params()))
            del model
            backend.clear_session()
    return result
