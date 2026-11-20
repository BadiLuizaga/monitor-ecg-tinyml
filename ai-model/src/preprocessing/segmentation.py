"""Detección de picos R y segmentación en latidos.

* ``detect_r_peaks``: detector tipo Pan-Tompkins simplificado, escrito como un
  bucle muestra a muestra para que sea trasladable 1:1 al firmware
  (``device-firmware/src/beat_detector.cpp``). Para entrenar se usan las
  anotaciones de MIT-BIH; el detector se usa en el dispositivo y para medir
  su propia sensibilidad.
* ``refine_peaks``: alinea cada anotación con el máximo de la señal filtrada,
  que es lo que hará el detector en el dispositivo (consistencia
  entrenamiento / despliegue).
* ``extract_beats``: ventanas de longitud fija centradas en el pico R.
"""

from __future__ import annotations

from collections import deque

import numpy as np

WINDOW_BEFORE_S = 0.25
WINDOW_AFTER_S = 0.45


def window_lengths(fs: float, before_s: float = WINDOW_BEFORE_S, after_s: float = WINDOW_AFTER_S) -> tuple[int, int]:
    return int(round(before_s * fs)), int(round(after_s * fs))


def extract_beats(x: np.ndarray, r_peaks: np.ndarray, fs: float, before_s: float = WINDOW_BEFORE_S,
                  after_s: float = WINDOW_AFTER_S) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve ``(beats[n, L], indices_validos)``; descarta latidos en los bordes."""
    nb, na = window_lengths(fs, before_s, after_s)
    r = np.asarray(r_peaks, dtype=np.int64)
    valid = np.flatnonzero((r - nb >= 0) & (r + na <= len(x)))
    if valid.size == 0:
        return np.empty((0, nb + na), dtype=np.float32), valid
    idx = r[valid, None] + np.arange(-nb, na)[None, :]
    return np.asarray(x, dtype=np.float32)[idx], valid


def refine_peaks(x: np.ndarray, peaks: np.ndarray, fs: float, search_s: float = 0.05) -> np.ndarray:
    """Mueve cada pico al máximo de ``x`` dentro de ±``search_s``."""
    half = max(1, int(round(search_s * fs)))
    out = np.empty(len(peaks), dtype=np.int64)
    for k, p in enumerate(np.asarray(peaks, dtype=np.int64)):
        lo, hi = max(0, p - half), min(len(x), p + half + 1)
        out[k] = lo + int(np.argmax(x[lo:hi])) if lo < hi else p
    return out


class StreamingRDetector:
    """Detector de picos R en tiempo real (una muestra por llamada).

    Pasos: derivada -> cuadrado -> integración en ventana móvil (150 ms) ->
    umbral adaptativo (niveles de señal/ruido) -> periodo refractario (200 ms)
    -> pico R = máximo de la señal filtrada en la ventana del complejo QRS.

    Los parámetros son los del algoritmo clásico y deben revalidarse con la
    señal del sensor real. Supone complejos QRS de polaridad positiva
    (derivación tipo MLII); otras derivaciones pueden requerir usar |x|.
    """

    def __init__(self, fs: float, integration_s: float = 0.150, refractory_s: float = 0.200,
                 learning_s: float = 2.0, threshold_ratio: float = 0.25) -> None:
        self.fs = fs
        self.w = max(1, int(round(integration_s * fs)))
        self.refractory = int(round(refractory_s * fs))
        self.learning = int(round(learning_s * fs))
        self.ratio = threshold_ratio
        self._sq = deque([0.0] * self.w, maxlen=self.w)
        self._hist = deque(maxlen=2 * self.w)  # historial de x para localizar el pico R
        self._integ_sum = 0.0
        self._prev_x = 0.0
        self._prev_i = [0.0, 0.0]
        self._n = 0
        self._spk = 0.0
        self._npk = 0.0
        self._learn_max = 0.0
        self._last_r = -10 ** 9
        self._in_qrs = False
        self._qrs_start = 0
        self._qrs_max = 0.0

    def threshold(self) -> float:
        return self._npk + self.ratio * (self._spk - self._npk)

    def process(self, x: float) -> int | None:
        """Procesa una muestra; devuelve el índice absoluto de un pico R o ``None``."""
        n = self._n
        self._n += 1
        d = x - self._prev_x if n > 0 else 0.0
        self._prev_x = x
        s = d * d
        self._integ_sum += s - self._sq[0]
        self._sq.append(s)
        integ = self._integ_sum / self.w
        self._hist.append(x)

        if n < self.learning:
            self._learn_max = max(self._learn_max, integ)
            if n == self.learning - 1:
                self._spk = 0.5 * self._learn_max
                self._npk = 0.1 * self._spk
            self._prev_i = [self._prev_i[1], integ]
            return None

        result = None
        thr = self.threshold()
        if not self._in_qrs:
            if integ > thr and n - self._last_r > self.refractory:
                self._in_qrs, self._qrs_start, self._qrs_max = True, n, integ
            elif self._prev_i[1] > self._prev_i[0] and self._prev_i[1] >= integ:
                # máximo local por debajo del umbral -> nivel de ruido
                self._npk = 0.125 * self._prev_i[1] + 0.875 * self._npk
        else:
            self._qrs_max = max(self._qrs_max, integ)
            if integ < 0.5 * thr or n - self._qrs_start >= self.w:
                # Fin del complejo: el R está en el historial reciente.
                span = min(len(self._hist), n - self._qrs_start + self.w)
                recent = list(self._hist)[-span:]
                r = n - span + 1 + int(np.argmax(recent))
                if r - self._last_r > self.refractory:
                    result = r
                    self._last_r = r
                self._spk = 0.125 * self._qrs_max + 0.875 * self._spk
                self._in_qrs = False
        self._prev_i = [self._prev_i[1], integ]
        return result


def detect_r_peaks(x: np.ndarray, fs: float, **kwargs) -> np.ndarray:
    """Aplica ``StreamingRDetector`` a una señal ya filtrada."""
    det = StreamingRDetector(fs, **kwargs)
    peaks = [p for p in map(det.process, np.asarray(x, dtype=float)) if p is not None]
    return np.asarray(peaks, dtype=np.int64)


def match_peaks(detected: np.ndarray, reference: np.ndarray, tolerance: int) -> tuple[int, int, int]:
    """Empareja detecciones con referencias (±tolerance muestras). Devuelve (TP, FP, FN)."""
    det = np.sort(np.asarray(detected))
    ref = np.sort(np.asarray(reference))
    used = np.zeros(det.size, dtype=bool)
    tp = 0
    for r in ref:
        j = np.searchsorted(det, r - tolerance)
        while j < det.size and det[j] <= r + tolerance:
            if not used[j]:
                used[j] = True
                tp += 1
                break
            j += 1
    return tp, int(det.size - tp), int(ref.size - tp)
