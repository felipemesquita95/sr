"""Controle do treino independente do ciclo de vida das páginas.

O subprocesso isola o treinamento da janela. Sua saída precisa ser drenada
continuamente para não bloquear no pipe, mesmo quando outra página está
visível; a thread leitora só compartilha um log limitado e protegido por lock.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from sr.config import Settings
from ui.dados import ROOT


def training_command(profile: Path, architecture: str, folds: int, epochs: int,
                     environ: dict | None = None) -> tuple[list[str], dict[str, str]]:
    """Reproduz o formulário no ambiente do mesmo executável de experimentos.

    Argumentos separados preservam caminhos com espaços sem passar por shell.
    Somente os controles do formulário e o ambiente de execução são impostos;
    ``PREPROCESS_ONLY`` continua sujeito ao perfil e às sobrescritas herdadas.

    Args:
        profile: Arquivo de configuração escolhido.
        architecture: Arquitetura escolhida no formulário.
        folds: Limite de partições a executar.
        epochs: Limite de épocas por treino.
        environ: Ambiente base; se omitido, copia o ambiente do processo atual.

    Returns:
        Lista de argumentos e ambiente próprios do subprocesso, sem alterar o ambiente base.

    Raises:
        ValueError: Se partições ou épocas não forem positivas.
    """
    if folds < 1 or epochs < 1:
        raise ValueError('Partições e épocas devem ser positivas.')
    env = dict(os.environ if environ is None else environ)
    env.update(KERAS_BACKEND='torch', SR_CONFIG=str(profile.resolve()),
               ARCHITECTURES=architecture, MAX_FOLDS=str(folds), EPOCHS=str(epochs),
               PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
    python = ROOT / '.venv/bin/python'
    command = [str(python) if python.exists() else sys.executable,
               str(ROOT / 'experiments/run_experiment.py')]
    return command, env


def feature_paths(settings: Settings) -> list[Path]:
    """Permite validar as entradas exigidas pelo protocolo antes de iniciar.

    Args:
        settings: Perfil do experimento.

    Returns:
        Diretórios de features de uma ou das várias trilhas do protocolo.
    """
    if settings.cross_mic:
        return [settings.features_path_train, settings.features_path_test]
    if settings.both_mics:
        return list(settings.features_paths)
    return [settings.features_path]


@dataclass
class Job:
    """Mantém o estado necessário para acompanhar e interromper um treino local.

    O log retém até 800 linhas para limitar memória durante treinos longos.
    Toda leitura ou escrita nele usa o mesmo lock; nenhuma referência a widget
    é entregue à thread que consome o subprocesso.

    Attributes:
        process: Subprocesso criado pelo gerenciador.
        output: Destino dos artefatos.
        architecture: Rede escolhida.
        started: Instante de início para filtrar progresso antigo.
        log: Últimas linhas recebidas.
        lock: Proteção do log compartilhado.
        stopping: Indica que o encerramento já foi solicitado.
    """
    process: subprocess.Popen
    output: Path
    architecture: str
    started: float
    log: deque = field(default_factory=lambda: deque(maxlen=800))
    lock: threading.Lock = field(default_factory=threading.Lock)
    stopping: bool = False

    def lines(self) -> str:
        """Obtém uma cópia consistente enquanto a leitora pode receber novas linhas.

        Returns:
            Trecho retido do log concatenado sob lock.
        """
        with self.lock:
            return ''.join(self.log)

    def stop(self):
        """Interrompe somente o subprocesso criado aqui sem bloquear a janela.

        A espera de até oito segundos e o eventual encerramento forçado ocorrem
        em outra thread. Chamadas repetidas não iniciam novas esperas.
        """
        if self.process.poll() is not None or self.stopping:
            return
        self.stopping = True
        self.process.terminate()

        def finish():
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        threading.Thread(target=finish, daemon=True).start()


class TrainingManager:
    """Limita cada janela a um treino, independentemente da navegação.

    O lock impede dois inícios concorrentes no mesmo gerenciador. Processos
    iniciados externamente não fazem parte desse controle.
    """
    def __init__(self):
        self.job: Job | None = None
        self.lock = threading.Lock()

    def start(self, command, env, output, architecture) -> Job:
        """Inicia o experimento e drena sua saída sem envolver widgets.

        A thread leitora só altera ``job.log`` sob lock; o timer da janela consulta
        esse estado. Isso evita tanto o bloqueio do pipe quanto acessos a widgets
        fora da thread gráfica.

        Args:
            command: Argumentos do executável, sem shell.
            env: Ambiente do subprocesso.
            output: Diretório acompanhado para métricas e progresso.
            architecture: Rede acompanhada nesse diretório.

        Returns:
            Estado do novo treino, disponível imediatamente após iniciar a leitora.

        Raises:
            RuntimeError: Se já houver um treino ativo neste gerenciador.
            OSError: Se o subprocesso não puder ser iniciado.
        """
        with self.lock:
            if self.job and self.job.process.poll() is None:
                raise RuntimeError('Já existe um treino ativo nesta interface.')
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True,
                                       encoding='utf-8', errors='replace', bufsize=1)
            job = self.job = Job(process, output, architecture, time.time())

            def consume():
                try:
                    for line in process.stdout:
                        with job.lock:
                            job.log.append(line)
                finally:
                    process.stdout.close()
                    process.wait()
            threading.Thread(target=consume, daemon=True).start()
            return job
