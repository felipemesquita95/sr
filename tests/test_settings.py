"""Testes da configuração tipada e das invariantes que ela protege.

As validações verificadas aqui existem para falhar cedo. Um perfil mal
especificado que passe despercebido não produz erro: produz um experimento que
roda por horas e devolve um número sem significado. São essas as falhas caras, e
são as que estes testes cobrem.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sr.config.settings import Settings, _as_bool, _as_path_list, _validate, load_settings

CONFIGS = Path(__file__).resolve().parent.parent / 'configs'


# ----------------------------------------------------------------------
# Propriedades derivadas
# ----------------------------------------------------------------------

def test_hop_size_is_half_the_frame():
    """O salto deve ser metade da janela, ou seja, 50% de sobreposição.

    O valor é citado na documentação e assumido no cálculo do número de quadros;
    divergir dele deslocaria silenciosamente a escala temporal das features.
    """
    assert Settings(frame_size=256).hop_size == 128
    assert Settings(frame_size=512).hop_size == 256


def test_frame_duration_follows_the_target_rate():
    """A duração da janela deve ser lida na taxa final, não na taxa de origem.

    Os perfis de 8 kHz e 16 kHz usam janelas de 256 e 512 amostras justamente para
    manter os mesmos 32 ms; se a duração fosse calculada na taxa de origem, essa
    equivalência — que é o que torna os dois perfis comparáveis — seria falsa.
    """
    assert Settings(frame_size=256, target_sampling_rate=8_000).frame_duration_ms == 32.0
    assert Settings(frame_size=512, target_sampling_rate=16_000).frame_duration_ms == 32.0


def test_chance_level_matches_the_number_of_speakers():
    """O nível do acaso deve ser o inverso do número de locutores, em porcentagem."""
    assert Settings(num_speakers=80).chance_level == pytest.approx(1.25)
    assert Settings(num_speakers=110).chance_level == pytest.approx(100 / 110)


def test_effective_folds_respects_the_cap():
    """``max_folds`` limita quantas partições rodam; zero significa todas."""
    assert Settings(num_folds=5, max_folds=0).effective_folds == 5
    assert Settings(num_folds=5, max_folds=2).effective_folds == 2
    # Um teto acima do total não inventa partições que não existem.
    assert Settings(num_folds=5, max_folds=9).effective_folds == 5


# ----------------------------------------------------------------------
# Invariantes de validação
# ----------------------------------------------------------------------

def test_accepts_the_reference_configuration():
    """A configuração padrão deve ser válida, senão o restante não se sustenta."""
    _validate(Settings())


def test_rejects_frame_size_that_is_not_a_power_of_two():
    """A FFT exige potência de dois; valores como 300 seriam reamostrados em silêncio."""
    with pytest.raises(ValueError, match='potência de dois'):
        _validate(Settings(frame_size=300))


def test_rejects_upsampling():
    """Reamostrar para cima não cria informação e mascararia um perfil errado.

    Um perfil que peça 48 kHz a partir de material de 16 kHz produziria features
    com banda aparentemente maior, mas vazia acima do Nyquist original.
    """
    with pytest.raises(ValueError, match='não cria informação'):
        _validate(Settings(source_sampling_rate=16_000, target_sampling_rate=48_000))


def test_rejects_degenerate_classification():
    """Menos de dois locutores, ou menos de duas partições, não é experimento."""
    with pytest.raises(ValueError, match='num_speakers'):
        _validate(Settings(num_speakers=1))
    with pytest.raises(ValueError, match='num_folds'):
        _validate(Settings(num_folds=1))


def test_rejects_unknown_dataset_format():
    """Um formato desconhecido deve falhar aqui, não na indexação do corpus."""
    with pytest.raises(ValueError, match='dataset_format'):
        _validate(Settings(dataset_format='timit'))


def test_vctk_requires_its_root():
    """O formato VCTK depende de VCTK_ROOT para localizar as trilhas."""
    with pytest.raises(ValueError, match='VCTK_ROOT'):
        _validate(Settings(dataset_format='vctk'))


def test_cross_mic_requires_both_feature_paths():
    """Sem os dois diretórios, o protocolo cross-mic testaria no próprio microfone.

    Essa é a falha silenciosa mais perigosa do repositório: o experimento rodaria
    até o fim e reportaria a acurácia intra-microfone como se fosse cruzada,
    invertendo exatamente a conclusão que o protocolo existe para sustentar.
    """
    with pytest.raises(ValueError, match='FEATURES_PATH_TRAIN'):
        _validate(Settings(cross_mic=True))
    with pytest.raises(ValueError, match='FEATURES_PATH_TRAIN'):
        _validate(Settings(cross_mic=True, features_path_train=Path('a')))


def test_multi_mic_requires_at_least_two_sources():
    """Com um só diretório, o protocolo multi-mic não descorrelaciona canal e rótulo."""
    with pytest.raises(ValueError, match='both_mics'):
        _validate(Settings(both_mics=True, features_paths=[Path('a')]))


# ----------------------------------------------------------------------
# Conversão dos valores textuais do perfil
# ----------------------------------------------------------------------

@pytest.mark.parametrize('raw', ['true', 'True', '1', 'yes', 'on', ' TRUE '])
def test_truthy_values(raw):
    assert _as_bool(raw) is True


@pytest.mark.parametrize('raw', ['false', 'False', '0', 'no', 'off', '', 'talvez'])
def test_falsy_values(raw):
    """Qualquer coisa que não seja afirmativa desativa a opção.

    Importa que ``ENABLE_VAD=talvez`` desligue o VAD em vez de ligá-lo: o padrão
    silencioso deve ser a condição menos processada do sinal.
    """
    assert _as_bool(raw) is False


def test_path_list_splits_and_strips():
    """FEATURES_PATHS é uma lista separada por vírgulas, tolerante a espaços."""
    assert _as_path_list('/a, /b ,/c') == [Path('/a'), Path('/b'), Path('/c')]
    assert _as_path_list('') == []


def test_profile_values_reach_the_declared_types(tmp_path):
    """Os valores do arquivo devem chegar convertidos, não como strings.

    ``NUM_MFCCS`` chegando como ``'40'`` só falharia lá adiante, dentro do
    librosa, com uma mensagem que não aponta para o perfil.
    """
    profile = tmp_path / 'perfil.env'
    profile.write_text(
        'EXPERIMENT_NAME=teste\n'
        'NUM_MFCCS=13\n'
        'PRE_EMPHASIS_COEF=0.0\n'
        'ENABLE_VAD=true\n'
        'ARCHITECTURES=cnn, attention\n'
        'AUDIO_PATH=/tmp/audios\n',
        encoding='utf-8',
    )

    settings = load_settings(profile)

    assert settings.experiment_name == 'teste'
    assert settings.num_mfccs == 13
    assert settings.pre_emphasis_coef == 0.0
    assert settings.enable_vad is True
    assert settings.architectures == ('cnn', 'attention')
    assert settings.audio_path == Path('/tmp/audios')


def test_environment_overrides_the_profile(tmp_path, monkeypatch):
    """O ambiente do processo tem precedência, permitindo variações pontuais."""
    profile = tmp_path / 'perfil.env'
    profile.write_text('NUM_MFCCS=40\n', encoding='utf-8')
    monkeypatch.setenv('NUM_MFCCS', '13')

    assert load_settings(profile).num_mfccs == 13


def test_invalid_value_names_the_offending_key(tmp_path):
    """A mensagem de erro deve identificar a chave e o arquivo."""
    profile = tmp_path / 'perfil.env'
    profile.write_text('NUM_MFCCS=quarenta\n', encoding='utf-8')

    with pytest.raises(ValueError, match='NUM_MFCCS'):
        load_settings(profile)


def test_missing_profile_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match='SR_CONFIG'):
        load_settings(tmp_path / 'inexistente.env')


# ----------------------------------------------------------------------
# Os perfis versionados
# ----------------------------------------------------------------------

@pytest.mark.parametrize('profile', sorted(CONFIGS.glob('*.env')), ids=lambda p: p.name)
def test_versioned_profiles_load_and_validate(profile):
    """Todo perfil em ``configs/`` deve carregar e passar na validação.

    Os perfis são o registro reprodutível de cada experimento. Um perfil quebrado
    só apareceria na próxima vez que alguém tentasse repetir o experimento — que é
    exatamente quando não se quer descobrir isso.
    """
    load_settings(profile)
