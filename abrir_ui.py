#!/usr/bin/env python3
"""Atalho local: usa automaticamente o ambiente virtual do projeto."""
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
python = root / '.venv/bin/python'
if not python.is_file():
    raise SystemExit('Ambiente não encontrado. Crie .venv e instale as dependências antes de abrir a interface.')
os.execv(str(python), [str(python), str(root / 'ui/app.py'), *sys.argv[1:]])
