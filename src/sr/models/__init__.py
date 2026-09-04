"""Arquiteturas de rede neural para identificação de locutor."""

from sr.models.registry import ARCHITECTURES, build_model

__all__ = ['ARCHITECTURES', 'build_model']
