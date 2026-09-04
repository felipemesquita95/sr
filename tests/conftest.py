"""Configuração comum à suíte de testes.

Torna ``src/`` importável a partir do repositório, sem exigir que o pacote esteja
instalado. O pytest carrega este arquivo antes de coletar os módulos de teste, de
modo que todos podem importar ``sr`` diretamente.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ``experiments/`` não é um pacote: são scripts de linha de comando. Entra no path
# para que os testes possam exercitar a lógica que vive neles.
for source in (ROOT / 'src', ROOT / 'experiments'):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
