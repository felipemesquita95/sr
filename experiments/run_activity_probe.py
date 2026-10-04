#!/usr/bin/env python3
"""Treina uma condição com MFCCs previamente separados do áudio completo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from sr.config import load_settings  # noqa: E402
from sr.system import SpeakerRecognitionSystem  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    config = Path(parser.parse_args().config)
    settings = load_settings(config)
    if not (settings.features_path / '1').is_dir():
        raise FileNotFoundError(f'Features do controle indisponíveis: {settings.features_path}')
    system = SpeakerRecognitionSystem(settings, resume=True)
    # As features deste diagnóstico são subconjuntos de quadros dos MFCCs já
    # extraídos. Processar o corpus novamente misturaria outras gravações.
    system.preprocessing.run = lambda: None
    system.run_cross_validation()


if __name__ == '__main__':
    main()
