"""Infraestrutura compartilhada pelo projeto educacional de classificação."""

from .auxiliares import (
    ALVO,
    COLUNA_ID,
    COLUNAS_NOMINAIS,
    COLUNAS_STATUS,
    MAPEAMENTO_COLUNAS,
    carregar_base_preparada,
    encontrar_raiz,
)

__all__ = [
    "ALVO",
    "COLUNA_ID",
    "COLUNAS_NOMINAIS",
    "COLUNAS_STATUS",
    "MAPEAMENTO_COLUNAS",
    "encontrar_raiz",
    "carregar_base_preparada",
]
