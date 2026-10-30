"""Funciones de evaluación adicionales para el Ejercicio 05.

Complementan a ``metrics.py``: NDCG, aciertos por usuario, concentración en
ítems populares y frontera de Pareto.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def per_user_hits(recommendations: dict[int, list[int]], truth: pd.DataFrame, k: int = 10) -> dict[int, int]:
    """1 si el ítem retenido de cada usuario aparece en su Top-K; 0 en caso contrario."""
    hits: dict[int, int] = {}
    for row in truth.itertuples():
        if row.userId in recommendations:
            hits[int(row.userId)] = int(row.movieId in recommendations[row.userId][:k])
    return hits


def ndcg_at_k(recommendations: dict[int, list[int]], truth: pd.DataFrame, k: int = 10) -> float:
    """NDCG@K con un único ítem relevante por usuario: 1/log2(posición+1) si aparece, 0 si no."""
    gains: list[float] = []
    for row in truth.itertuples():
        if row.userId not in recommendations:
            continue
        top = list(recommendations[row.userId][:k])
        gains.append(1.0 / math.log2(top.index(row.movieId) + 2) if row.movieId in top else 0.0)
    return sum(gains) / len(gains) if gains else 0.0


def popularity_share(recommendations: dict[int, list[int]], popular_items: set[int]) -> float:
    """Proporción de posiciones recomendadas ocupadas por ítems del conjunto popular."""
    total = sum(len(values) for values in recommendations.values())
    if total == 0:
        return 0.0
    inside = sum(1 for values in recommendations.values() for item in values if item in popular_items)
    return inside / total


def pareto_mask(frame: pd.DataFrame, maximize: list[str], minimize: list[str]) -> np.ndarray:
    """Marca con True las filas no dominadas.

    Una fila está dominada si otra es igual o mejor en todos los objetivos y
    estrictamente mejor en al menos uno.
    """
    parts = [frame[maximize].to_numpy(dtype=float)]
    if minimize:
        parts.append(-frame[minimize].to_numpy(dtype=float))
    values = np.hstack(parts)
    keep = np.ones(len(values), dtype=bool)
    for i in range(len(values)):
        better_or_equal = np.all(values >= values[i], axis=1)
        strictly_better = np.any(values > values[i], axis=1)
        keep[i] = not np.any(better_or_equal & strictly_better)
    return keep
