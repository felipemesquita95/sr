"""Captura uma tela por seção da interface, com uma gravação fixa.

Percorre os quatro passos e cada aba interna, espera o carregamento em segundo
plano terminar e grava um PNG por tela. Serve para revisar a apresentação sem
abrir a janela.

    QT_QPA_PLATFORM=offscreen .venv/bin/python capturar_ui.py vctk_mic2 99 3
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-captura')

ROOT = Path(__file__).resolve().parent
for source in (ROOT, ROOT / 'src'):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from PySide6.QtCore import QCoreApplication, QEventLoop
from PySide6.QtWidgets import QApplication

from ui.app import MainWindow
from ui.conteudo import ROTEIRO


def aguardar(condicao, limite=90):
    """Roda o laço de eventos até a condição valer ou o limite estourar.

    Args:
        condicao: Chamável sem argumentos avaliado a cada iteração.
        limite: Tempo máximo de espera em segundos.

    Returns:
        Verdadeiro se a condição passou a valer dentro do limite.
    """
    fim = time.monotonic() + limite
    while not condicao() and time.monotonic() < fim:
        QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        time.sleep(.01)
    QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
    return condicao()


def assentar(voltas=14):
    """Deixa o laço girar para que gráficos e imagens terminem de desenhar."""
    for _ in range(voltas):
        QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 60)
        time.sleep(.06)


def main(trilha, locutor, enunciado):
    destino = ROOT / 'capturas'
    destino.mkdir(exist_ok=True)
    for antigo in destino.glob('*.png'):
        antigo.unlink()

    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion')
    janela = MainWindow(persist=False)
    janela.resize(1600, 1040)
    janela.show()

    indice = janela.track.findText(trilha)
    if indice < 0:
        raise SystemExit(f'Trilha {trilha} não encontrada.')
    janela.track.setCurrentIndex(indice)
    if not aguardar(lambda: janela.speaker.count() > 0):
        raise SystemExit('O índice da trilha não terminou de carregar.')

    alvo = janela.speaker.findData(locutor)
    if alvo < 0:
        raise SystemExit(f'Locutor {locutor} ausente em {trilha}.')
    janela.speaker.setCurrentIndex(alvo)
    aguardar(lambda: janela.utterance.count() > 0)
    alvo = janela.utterance.findData(enunciado)
    if alvo < 0:
        raise SystemExit(f'Enunciado {enunciado} ausente para o locutor {locutor}.')
    janela.utterance.setCurrentIndex(alvo)
    aguardar(lambda: janela.selection is not None
             and janela.selection.key == (str(janela.track.currentData()), locutor, enunciado))

    print(f'Seleção: {janela.selection.caption}')
    gravadas = []
    for passo, (stage, titulo, _) in enumerate(ROTEIRO):
        janela.navigation.setCurrentRow(passo)
        pagina = janela.page_by_stage[stage]
        aguardar(lambda p=pagina: p.loaded_key is not None)
        abas = getattr(pagina, 'sections', None)
        nomes = list(getattr(pagina, 'section_names', ()) or ())
        total = abas.count() if abas is not None else 1
        for aba in range(total):
            if abas is not None:
                abas.setCurrentIndex(aba)
            assentar()
            nome = nomes[aba] if aba < len(nomes) else f'aba{aba + 1}'
            limpo = ''.join(c if c.isalnum() else '_' for c in nome).strip('_').lower()
            arquivo = destino / f'{passo + 1}{chr(97 + aba)}_{stage}_{limpo}.png'
            janela.grab().save(str(arquivo))
            gravadas.append(arquivo)
            barra = 'com barra' if janela.toolbar.isVisible() else 'sem barra'
            print(f'  {arquivo.name}  ({titulo} → {nome}, {barra})')

    janela.close()
    print(f'\n{len(gravadas)} telas em {destino}')


if __name__ == '__main__':
    trilha = sys.argv[1] if len(sys.argv) > 1 else 'vctk_mic2'
    locutor = int(sys.argv[2]) if len(sys.argv) > 2 else 99
    enunciado = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    main(trilha, locutor, enunciado)
