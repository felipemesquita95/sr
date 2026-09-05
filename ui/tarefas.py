"""Entrega de resultados de disco e CPU pelo laço de eventos do Qt.

QRunnable executa o trabalho no pool, mas apenas o slot do objeto Tasks,
criado na thread da janela, chama os callbacks de apresentação. As funções
enviadas ao pool não devem criar nem modificar widgets.
"""
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class Signals(QObject):
    """Fornece um emissor QObject ao QRunnable para atravessar a fronteira de threads.

    O sinal carrega o identificador da solicitação, seu resultado e uma mensagem
    de erro; o callback permanece guardado na thread da janela.
    """
    finished = Signal(int, object, str)


class Worker(QRunnable):
    """Converte o término da tarefa em uma mensagem para a janela.

    Args:
        number: Identificador que associa o resultado ao callback.
        function: Função sem argumentos que não acessa widgets.
    """
    def __init__(self, number, function):
        super().__init__()
        self.number, self.function = number, function
        self.signals = Signals()

    @Slot()
    def run(self):
        """Encaminha sucesso ou falha sem executar o callback na thread do pool.

        Exceções do trabalho viram mensagens para a página. Se a aplicação já
        destruiu o emissor ao fechar, a entrega é abandonada.
        """
        try:
            result, message = self.function(), ''
        except Exception as error:
            result, message = None, str(error)
        try:
            self.signals.finished.emit(self.number, result, message)
        except RuntimeError:
            # QApplication pode já ter destruído os sinais ao fechar a janela.
            pass


class Tasks(QObject):
    """Mantém tarefas vivas até a entrega de seus resultados à janela.

    Deve ser criado na thread gráfica. O pool global fica limitado a duas
    threads para conter leituras e cálculos concorrentes; referências aos
    workers e callbacks permanecem em ``pending`` até a conclusão.

    Args:
        parent: Objeto Qt que controla a vida do gerenciador.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(2)
        self.pending = {}
        self.counter = 0
        self.closed = False

    def submit(self, function, callback):
        """Associa uma tarefa a uma entrega posterior na thread da janela.

        O chamador decide se a resposta ainda corresponde à seleção atual.
        O gerenciador garante a thread de entrega, não a atualidade do conteúdo.

        Args:
            function: Função sem argumentos, executada no pool sem acessar widgets.
            callback: Recebe ``(valor, erro)``; erro vazio indica sucesso.

        Returns:
            Identificador crescente da solicitação.
        """
        self.counter += 1
        worker = Worker(self.counter, function)
        self.pending[self.counter] = worker, callback
        worker.signals.finished.connect(self._finish)
        self.pool.start(worker)
        return self.counter

    @Slot(int, object, str)
    def _finish(self, number, value, error):
        entry = self.pending.pop(number, None)
        if entry and not self.closed:
            entry[1](value, error)

    def close(self):
        """Impede callbacks de apresentação depois de iniciar o fechamento.

        Não espera nem cancela trabalhos em execução; eles ainda podem concluir,
        mas seus resultados deixam de ser entregues aos widgets.
        """
        self.closed = True
