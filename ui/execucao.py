"""Treino em subprocesso, log limitado em memória e encerramento sem bloquear a UI."""
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
    if settings.cross_mic:
        return [settings.features_path_train, settings.features_path_test]
    if settings.both_mics:
        return list(settings.features_paths)
    return [settings.features_path]


@dataclass
class Job:
    process: subprocess.Popen
    output: Path
    architecture: str
    started: float
    log: deque = field(default_factory=lambda: deque(maxlen=800))
    lock: threading.Lock = field(default_factory=threading.Lock)
    stopping: bool = False

    def lines(self) -> str:
        with self.lock:
            return ''.join(self.log)

    def stop(self):
        """Só sinaliza o subprocesso criado aqui; nunca um PID arbitrário."""
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
    """Um treino por janela; o subprocesso continua independente da navegação."""
    def __init__(self):
        self.job: Job | None = None
        self.lock = threading.Lock()

    def start(self, command, env, output, architecture) -> Job:
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
