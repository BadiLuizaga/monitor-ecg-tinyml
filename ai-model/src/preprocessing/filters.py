"""Filtrado y remuestreo de señales ECG.

Decisión de diseño: por defecto el filtrado es **causal** (``sosfilt``), igual
que el que podrá ejecutar el ESP32 muestra a muestra. Así la forma de los
latidos que ve el modelo durante el entrenamiento es la misma que verá en el
dispositivo. El filtrado de fase cero (``sosfiltfilt``) queda disponible para
análisis offline, pero no debe usarse para entrenar el modelo que se despliega.

Los coeficientes SOS (secciones de segundo orden / biquads) se exportan a C++
con ``src/export/export_model.py`` para que el firmware use exactamente los
mismos filtros.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal as sps

DEFAULT_BAND_HZ = (0.5, 40.0)
DEFAULT_ORDER = 2


def design_bandpass(fs: float, low_hz: float = DEFAULT_BAND_HZ[0], high_hz: float = DEFAULT_BAND_HZ[1],
                    order: int = DEFAULT_ORDER) -> np.ndarray:
    """Butterworth paso banda en formato SOS.

    El paso alto (0.5 Hz) elimina la deriva de línea base; el paso bajo
    (40 Hz) reduce ruido muscular y de alta frecuencia.
    """
    nyq = fs / 2.0
    if not 0 < low_hz < high_hz < nyq:
        raise ValueError(f"banda inválida ({low_hz}, {high_hz}) Hz para fs={fs} Hz")
    return sps.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")


def design_notch(fs: float, f0_hz: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Filtro de banda eliminada para la interferencia de red (50 o 60 Hz)."""
    if not 0 < f0_hz < fs / 2.0:
        raise ValueError(f"frecuencia de notch {f0_hz} Hz inválida para fs={fs} Hz")
    b, a = sps.iirnotch(f0_hz, q, fs=fs)
    return sps.tf2sos(b, a)


def design_preprocessing_sos(fs: float, band_hz: tuple[float, float] = DEFAULT_BAND_HZ,
                             order: int = DEFAULT_ORDER, notch_hz: float | None = None) -> np.ndarray:
    """Cascada completa (paso banda + notch opcional) como una sola matriz SOS."""
    sos = design_bandpass(fs, band_hz[0], band_hz[1], order)
    # Si el notch cae por encima del paso bajo ya está muy atenuado; aun así se
    # permite añadirlo cuando la interferencia es fuerte.
    if notch_hz is not None:
        sos = np.vstack([sos, design_notch(fs, notch_hz)])
    return sos


def apply_sos(x: np.ndarray, sos: np.ndarray, causal: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return sps.sosfilt(sos, x) if causal else sps.sosfiltfilt(sos, x)


def bandpass_filter(x: np.ndarray, fs: float, band_hz: tuple[float, float] = DEFAULT_BAND_HZ,
                    order: int = DEFAULT_ORDER, notch_hz: float | None = None,
                    causal: bool = True) -> np.ndarray:
    return apply_sos(x, design_preprocessing_sos(fs, band_hz, order, notch_hz), causal)


def resample_signal(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Remuestreo polifásico (p. ej. MIT-BIH 360 Hz -> frecuencia del sensor real)."""
    if fs_in == fs_out:
        return np.asarray(x, dtype=float)
    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    return sps.resample_poly(np.asarray(x, dtype=float), ratio.numerator, ratio.denominator)


def resample_indices(indices: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Convierte posiciones de muestra (p. ej. picos R anotados) a la nueva frecuencia."""
    return np.round(np.asarray(indices) * (fs_out / fs_in)).astype(np.int64)
