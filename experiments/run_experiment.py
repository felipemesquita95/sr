#!/usr/bin/env python3
"""Ponto de entrada dos experimentos de reconhecimento de locutor.

O experimento é descrito por um perfil em ``configs/``. O protocolo executado
decorre do próprio perfil, de modo que reproduzir um resultado exige apenas o
arquivo correspondente::

    SR_CONFIG=configs/brsd.env python experiments/run_experiment.py

Parâmetros isolados podem ser sobrescritos pelo ambiente, sem editar o perfil::

    NUM_MFCCS=13 SR_CONFIG=configs/vctk.env python experiments/run_experiment.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from sr.config import load_settings  # noqa: E402
from sr.system import SpeakerRecognitionSystem  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando.

    Returns:
        Argumentos com o perfil a carregar e o nível de detalhe do log.
    """
    parser = argparse.ArgumentParser(
        description='Executa um experimento de reconhecimento de locutor.')
    parser.add_argument(
        '--config', '-c', default=None,
        help='Perfil do experimento. Se omitido, usa a variável SR_CONFIG.')
    parser.add_argument(
        '--preprocess-only', action='store_true',
        help='Executa apenas o pré-processamento e encerra.')
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Reduz o log a avisos e erros.')
    return parser.parse_args()


def main() -> int:
    """Carrega o perfil e executa o experimento.

    Returns:
        Código de saída: ``0`` em caso de sucesso, ``1`` em caso de erro de
        configuração.
    """
    arguments = parse_arguments()

    logging.basicConfig(
        level=logging.WARNING if arguments.quiet else logging.INFO,
        format='%(asctime)s  %(levelname)-7s %(name)s  %(message)s',
        datefmt='%H:%M:%S',
    )

    try:
        settings = load_settings(arguments.config)
    except (FileNotFoundError, ValueError) as error:
        logging.error('%s', error)
        return 1

    if arguments.preprocess_only:
        from dataclasses import replace
        settings = replace(settings, preprocess_only=True)

    SpeakerRecognitionSystem(settings).run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
