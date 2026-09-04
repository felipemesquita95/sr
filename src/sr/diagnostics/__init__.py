"""Instrumentos de diagnóstico: o que o sinal revela fora da fala."""

from sr.diagnostics.signature import (
    CONDITIONS,
    MIN_SAMPLES,
    channel_signature,
    extract_condition,
    signatures_of,
)

__all__ = [
    'CONDITIONS',
    'MIN_SAMPLES',
    'channel_signature',
    'extract_condition',
    'signatures_of',
]
