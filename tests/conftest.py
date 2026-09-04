"""Configuração comum à suíte de testes.

Torna ``src/`` importável a partir do repositório, sem exigir que o pacote esteja
instalado. O pytest carrega este arquivo antes de coletar os módulos de teste, de
modo que todos podem importar ``sr`` diretamente.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / 'src'

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
