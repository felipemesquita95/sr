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


ROTEIRO = [
    ('passo_sinal', 'Sinal', 'Corpus, sinal original e pré-processamento da gravação escolhida.'),
    ('passo_mfcc', 'MFCC', 'Representação persistida, par entre trilhas e assinatura de canal.'),
    ('rede', 'Rede', 'Arquiteturas salvas, parâmetros e divisão dos tensores.'),
    ('resultado', 'Resultado', 'Resultados, comparação de experimentos, evidências e ressalvas.'),
]
STAGES = ROTEIRO
SELECTION_STAGES = {stage for stage, _, _ in ROTEIRO[:-1]}
SECOES = {
    'passo_sinal': (('corpus', 'Corpus'), ('sinal', 'Sinal original'),
                    ('preprocessamento', 'Pré-processamento')),
    'passo_mfcc': (('mfcc', 'Matriz e derivadas'), ('assinatura', 'Assinatura de canal')),
    'rede': (('modelo', 'Arquiteturas'), ('tensores', 'Montagem dos tensores')),
    'resultado': (('resultados', 'Resultado principal'), ('comparacao', 'Comparar experimentos'),
                  ('evidencias', 'Evidências'), ('limites', 'Ressalvas')),
}
CAMPOS = {
    'passo_sinal': ('source_sampling_rate', 'target_sampling_rate', 'enable_vad',
                    'vad_top_db', 'pre_emphasis_coef'),
    'passo_mfcc': ('num_mfccs', 'frame_size'),
    'rede': ('num_folds', 'num_speakers', 'num_utterances', 'batch_size',
             'learning_rate', 'early_stopping_patience', 'cross_mic',
             'cross_mic_disjoint_utterances', 'both_mics', 'max_frames_cap'),
}


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
    if stage in SECOES:
        secoes = {}
        metricas = None
        evidencia = None
        for nome, _ in SECOES[stage]:
            try:
                if nome in ('resultados', 'comparacao'):
                    if metricas is None:
                        metricas = load_stage('resultados', ctx)
                    secoes[nome] = metricas
                elif nome in ('evidencias', 'limites'):
                    if evidencia is None:
                        evidencia = load_stage('evidencias', ctx)
                    secoes[nome] = evidencia
                else:
                    secoes[nome] = load_stage(nome, ctx, ctx.settings, fold)
            except (OSError, ValueError) as erro:
                secoes[nome] = {'erro': str(erro)}
        return secoes
    payload = {'images': []}
    files = {
        'sinal': [('sinal_original.png', 'Forma de onda original'), ('espectro_original.png', 'Espectro original')],
        'preprocessamento': [('sinal_original.png', 'Antes do VAD'), ('sinal_vad.png', 'Depois do VAD'),
                              ('espectro_filtrado.png', 'Após filtro anti-aliasing'),
                              ('espectro_reamostrado.png', 'Após reamostragem'),
                              ('espectro_preenfase.png', 'Depois da pré-ênfase')],
    }
    if stage in files:
        selected = files[stage]
        if stage == 'preprocessamento' and ctx.settings and not ctx.settings.enable_vad:
            selected = [('sinal_original.png', 'Sinal preservado • VAD desativado')] + selected[2:]
        payload['images'] = [image(ctx.sample / name, title) for name, title in selected]
        payload['available_figures'] = [u for u in ctx.inventory[ctx.speaker]
            if all((ctx.track / str(ctx.speaker) / str(u) / name).is_file() for name, _ in selected)]
        if stage == 'sinal' and any(value.isNull() for _, value, _ in payload['images']):
            source = dados.audio_source(ctx.settings, ctx.speaker, ctx.utterance, ctx.manifest, ctx.track.name)
            if source:
                payload['raw'] = dados.raw_signal(source)
                payload['images'] = []
        if stage == 'preprocessamento':
            payload['alignment'] = dados.read_json(dados.RUNS / 'models/verificacao_alinhamento/alinhamento.json')
    elif stage == 'corpus':
        payload['images'] = [image(ctx.track / '_resumo' / name, title) for name, title in
                             [('duracao.png', 'Duração das gravações'), ('enunciados_por_locutor.png', 'Gravações por locutor')]]
    elif stage == 'modelo':
        payload['modelos'] = dados.model_summaries(ctx.settings)
    elif stage == 'mfcc':
        payload['matrices'] = [dados.mfcc(ctx.sample / 'mfccs.npy')]
        payload['titles'] = [ctx.track.name]
        payload['origens_mfcc'] = [ctx.sample / 'mfccs.npy']
        payload['derivadas'] = [(ctx.sample / nome, dados.mfcc(ctx.sample / nome)
                                 if (ctx.sample / nome).is_file() else None)
                                for nome in ('delta.npy', 'delta_delta.npy')]
        if ctx.track.name.endswith(('_mic1', '_mic2')) and ctx.manifest:
            partner = ctx.track.with_name(ctx.track.name[:-1] + ('2' if ctx.track.name.endswith('1') else '1'))
            file = partner / str(ctx.speaker) / str(ctx.utterance) / 'mfccs.npy'
            if file.is_file():
                payload['matrices'].append(dados.mfcc(file))
                payload['titles'].append(partner.name)
                payload['origens_mfcc'].append(file)
    elif stage == 'assinatura':
        payload['signatures'] = dados.signatures(ctx.sample / 'assinaturas.npz')
        payload['transfer'] = dados.read_json(dados.RUNS / 'models/vctk_cross_mic/diagnostico_travessia/travessia_canal.json')
    elif stage == 'tensores':
        if settings is None:
            raise ValueError(f'Sem perfil associado a {ctx.track}. Esperado: perfil em {dados.ROOT / "configs"} que referencie esta trilha; divisão indisponível.')
        split, frames = dados.inspect_split(settings, fold)
        matrix = dados.mfcc(ctx.sample / 'mfccs.npy')
        payload.update(split=split, frames=frames, settings=settings, fold=fold,
                       matrices=[matrix, FeatureAdjustmentSubsystem.pad_or_truncate(matrix, frames)])
    elif stage == 'resultados':
        payload['metrics'], payload['errors'] = dados.metrics(dados.RUNS / 'models')
    elif stage in ('evidencias', 'limites'):
        payload['diagnosticos'] = dados.diagnosticos(dados.RUNS / 'models')
        payload['resultados'] = dados.texto_documento('resultados.md')
        payload['limitacoes'] = dados.texto_documento('limitacoes.md')
    return payload
