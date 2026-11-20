"""Normalización de latidos.

Se normaliza cada latido de forma independiente (z-score), lo que elimina la
dependencia de la ganancia absoluta del front-end analógico. Esto es clave
mientras no conozcamos la resolución ni la ganancia reales del ADC: el modelo
entrenado con MIT-BIH (mV) podrá recibir cuentas de ADC sin calibrar.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def zscore(beats: np.ndarray, eps: float = EPS) -> np.ndarray:
    """(x - media) / desviación por fila. Acepta un latido 1D o una matriz 2D."""
    x = np.asarray(beats, dtype=np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / np.maximum(std, eps)


def minmax(beats: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Escala cada latido a [-1, 1]."""
    x = np.asarray(beats, dtype=np.float32)
    lo = x.min(axis=-1, keepdims=True)
    hi = x.max(axis=-1, keepdims=True)
    return 2.0 * (x - lo) / np.maximum(hi - lo, eps) - 1.0


NORMALIZERS = {"zscore": zscore, "minmax": minmax}


def normalize(beats: np.ndarray, method: str = "zscore") -> np.ndarray:
    try:
        return NORMALIZERS[method](beats)
    except KeyError:
        raise ValueError(f"normalización desconocida: {method!r} (opciones: {sorted(NORMALIZERS)})") from None
