#!/usr/bin/env python3
"""Converte um documento Markdown em PDF, sem depender de rede nem de LaTeX.

A cadeia é ``markdown`` para HTML e Chromium em modo headless para o PDF. Ambos já
existem na máquina: o pandoc, o wkhtmltopdf e as distribuições TeX, não.

Uso::

    python3 gerar_pdf.py docs/apresentacao.md
    python3 gerar_pdf.py docs/apresentacao.md --saida /tmp/apresentacao.pdf
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

#: Folha de estilo de impressão. Serifa no corpo do texto porque o documento é lido
#: no papel; monoespaçada nas tabelas de números, para que as colunas alinhem.
CSS = """
@page { size: A4; margin: 1.6cm 1.8cm; }
body { font-family: 'DejaVu Serif', Georgia, serif; font-size: 9.8pt;
       line-height: 1.4; color: #1a1a1a; }
h1 { font-size: 15pt; margin: 0 0 .2em; padding-bottom: .25em;
     border-bottom: 2px solid #1a1a1a; }
h2 { font-size: 13pt; margin: .9em 0 .4em; page-break-after: avoid;
     border-bottom: 1px solid #bbb; padding-bottom: .2em; }
h3 { font-size: 11.5pt; margin: 1.2em 0 .4em; page-break-after: avoid; }
p, ul, ol { margin: .5em 0; }
li { margin: .25em 0; }
table { border-collapse: collapse; width: 100%; margin: .6em 0;
        font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: .22em .55em; text-align: left;
         vertical-align: top; }
th { background: #f0f0f0; font-weight: bold; }
td:not(:first-child) { font-variant-numeric: tabular-nums; }
blockquote { margin: .5em 0; padding: .45em 1em; border-left: 3px solid #666;
             background: #f7f7f7; page-break-inside: avoid; }
blockquote p { margin: .25em 0; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt;
       background: #f0f0f0; padding: .1em .3em; border-radius: 2px; }
pre { background: #f7f7f7; border: 1px solid #ddd; padding: .7em;
      page-break-inside: avoid; }
pre code { background: none; padding: 0; }
hr { border: none; border-top: 1px solid #ccc; margin: 1.5em 0; }
a { color: #1a1a1a; }
img { width: 100%; height: auto; display: block; margin: .5em 0 .2em;
      page-break-inside: avoid; }
/* Legenda: o parágrafo em itálico logo abaixo da figura. */
p > em:only-child { display: block; font-size: 8.6pt; color: #444;
                    margin: -.2em 0 .7em; }
"""

MOLDE = """<!DOCTYPE html>
<html lang="pt-BR"><head><meta charset="utf-8"><base href="{base}">
<title>{titulo}</title><style>{css}</style></head>
<body>{corpo}</body></html>
"""


def html_de(origem: Path) -> str:
    """Renderiza o Markdown com as extensões necessárias às tabelas e ao código.

    O ``<base>`` aponta para o diretório do documento: o HTML intermediário vive em
    um diretório temporário, e sem isso as figuras referenciadas por caminho relativo
    não seriam encontradas pelo navegador.
    """
    corpo = markdown.markdown(
        origem.read_text(encoding='utf-8'),
        extensions=['tables', 'fenced_code', 'sane_lists', 'attr_list'],
    )
    return MOLDE.format(base=origem.resolve().parent.as_uri() + '/',
                        titulo=origem.stem, css=CSS, corpo=corpo)


def navegador() -> str:
    """Localiza um Chromium utilizável.

    Raises:
        SystemExit: Se nenhum executável conhecido estiver instalado.
    """
    for nome in ('chromium', 'chromium-browser', 'google-chrome', 'chrome'):
        caminho = shutil.which(nome)
        if caminho:
            return caminho
    sys.exit('Nenhum Chromium encontrado; instale um para gerar o PDF.')


def gerar(origem: Path, saida: Path) -> None:
    """Grava o PDF de ``origem`` em ``saida``.

    O HTML intermediário vai para um diretório temporário, junto do perfil do
    navegador: sem ``--user-data-dir`` próprio, uma instância já aberta do Chromium
    faria o processo headless encerrar sem produzir arquivo algum.
    """
    with tempfile.TemporaryDirectory() as temporario:
        pasta = Path(temporario)
        pagina = pasta / 'documento.html'
        pagina.write_text(html_de(origem), encoding='utf-8')

        saida.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [navegador(), '--headless', '--disable-gpu', '--no-sandbox',
             f'--user-data-dir={pasta / "perfil"}',
             '--no-pdf-header-footer', '--print-to-pdf-no-header',
             f'--print-to-pdf={saida}', pagina.as_uri()],
            check=True, capture_output=True, timeout=180,
        )

    if not saida.exists():
        sys.exit(f'O Chromium terminou sem gravar {saida}.')


def main() -> None:
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument('origem', type=Path, help='documento Markdown de entrada')
    analisador.add_argument('--saida', type=Path, default=None,
                            help='PDF de saída (padrão: mesmo nome, extensão .pdf)')
    argumentos = analisador.parse_args()

    origem = argumentos.origem
    if not origem.is_file():
        sys.exit(f'Documento não encontrado: {origem}')

    saida = argumentos.saida or origem.with_suffix('.pdf')
    gerar(origem, saida)
    print(f'{saida}  ({saida.stat().st_size / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
