"""Sistema de reconhecimento automático de locutor.

Fixa o backend do Keras antes que qualquer submódulo o importe.

O motivo é a placa de vídeo. O TensorFlow distribuído no PyPI é compilado para CUDA
e não enxerga uma GPU AMD; o PyTorch tem distribuição oficial com ROCm, que a
reconhece. Como o Keras 3 é agnóstico de backend e o código deste repositório usa
apenas a sua API, trocar o backend não exige alterar modelo algum — mas exige que a
variável esteja definida **antes** do primeiro ``import keras``, porque é nesse
momento que o Keras resolve qual backend carregar.

Definir isso aqui, e não no ambiente do usuário, evita uma falha de importação cuja
mensagem não sugere a causa. ``setdefault`` preserva a escolha de quem já tiver
definido a variável — trocar de backend para comparar continua possível.
"""

import os

os.environ.setdefault('KERAS_BACKEND', 'torch')
