"""Subsistema de pré-processamento: áudio bruto para matrizes de MFCC."""

from sr.preprocessing.pipeline import PreprocessingSubsystem
from sr.preprocessing import signal

__all__ = ['PreprocessingSubsystem', 'signal']
