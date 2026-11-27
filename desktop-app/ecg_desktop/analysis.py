"""Análisis de señal en el PC: detección de picos R y frecuencia cardíaca.

La frecuencia cardíaca NO la envía el dispositivo (el protocolo v1 no tiene ese
campo): se calcula aquí a partir de la señal ECG recibida. Detector en
streaming, inspirado en el del firmware (derivada al cuadrado, integración
móvil de 150 ms, umbral adaptativo, periodo refractario de 200 ms).
"""
from __future__ import annotations

from collections import deque
from statistics import median


class RPeakDetector:
    """Detecta picos R muestra a muestra. ``process`` devuelve el índice absoluto
    del pico cuando el complejo QRS termina (retraso ≈ ancho del QRS)."""

    def __init__(self, fs: float, integration_s: float = 0.150, refractory_s: float = 0.200,
                 learning_s: float = 2.0, threshold_ratio: float = 0.25) -> None:
        self.fs = float(fs)
        self.w = max(1, round(integration_s * fs))
        self.refractory = round(refractory_s * fs)
        self.learning = round(learning_s * fs)
        self.ratio = threshold_ratio
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self._sq: deque[float] = deque([0.0] * self.w, maxlen=self.w)
        self._sum = 0.0
        self._hist: deque[float] = deque(maxlen=3 * self.w)
        self._prev = 0.0
        self._learn_max = 0.0
        self._spk = 0.0
        self._in_qrs = False
        self._qrs_max = 0.0
        self._qrs_start = 0
        self._last_r = -10**9

    def process(self, x: float) -> int | None:
        idx = self.n
        self.n += 1
        d = x - self._prev
        self._prev = x
        sq = d * d
        self._sum += sq - self._sq[0]
        self._sq.append(sq)
        integ = max(self._sum, 0.0) / self.w
        self._hist.append(x)

        if idx < self.learning:  # aprendizaje inicial del nivel de QRS
            self._learn_max = max(self._learn_max, integ)
            if idx == self.learning - 1:
                self._spk = self._learn_max
            return None

        thr = self.ratio * self._spk
        if not self._in_qrs:
            if integ > thr and thr > 0 and idx - self._last_r > self.refractory:
                self._in_qrs = True
                self._qrs_max = integ
                self._qrs_start = idx
            return None

        self._qrs_max = max(self._qrs_max, integ)
        if integ < thr or idx - self._qrs_start > 2 * self.w:
            self._in_qrs = False
            # R = máximo de |x| (respecto a la media local) en la ventana reciente
            hist = list(self._hist)
            base = sum(hist) / len(hist)
            k = max(range(len(hist)), key=lambda i: abs(hist[i] - base))
            r = idx - (len(hist) - 1 - k)
            self._spk = 0.125 * self._qrs_max + 0.875 * self._spk
            self._last_r = r
            return r
        return None


class HeartRateEstimator:
    """BPM = 60 / mediana de los últimos RR plausibles (30–220 lpm)."""

    def __init__(self, fs: float, window: int = 8) -> None:
        self.fs = float(fs)
        self._rr: deque[float] = deque(maxlen=window)
        self._last_r: int | None = None

    def reset(self) -> None:
        self._rr.clear()
        self._last_r = None

    def add_peak(self, r_index: int) -> None:
        if self._last_r is not None:
            rr = (r_index - self._last_r) / self.fs
            if 60.0 / 220.0 <= rr <= 60.0 / 30.0:
                self._rr.append(rr)
            else:  # pausa o latido perdido: no se mezcla con el ritmo previo
                self._rr.clear()
        self._last_r = r_index

    @property
    def bpm(self) -> float | None:
        if len(self._rr) < 2:
            return None
        return 60.0 / median(self._rr)

    @property
    def beats_used(self) -> int:
        return len(self._rr)
