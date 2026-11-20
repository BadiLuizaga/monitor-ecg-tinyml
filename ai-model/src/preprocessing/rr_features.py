"""Características de intervalos RR (segunda entrada del modelo).

Definición para el latido *k* (índices de muestra enteros; tiempos = muestras / fs):

* ``pre``   = t_k − t_{k−1}                         (causal)
* ``post``  = min(t_{k+1} − t_k, max_rr_s)          (requiere esperar 1 latido)
* ``local`` = media de los ``local_beats`` RR anteriores, SIN incluir ``pre``
* ``long``  = media exponencial (tau = ``long_tau_s``) de los RR anteriores,
              valor previo a incorporar ``pre``

Características (transformaciones FIJAS, sin parámetros ajustados a datos, así
que no pueden filtrar información entre pacientes), recortadas a ±``clip``:

    log2(pre/local), log2(post/local), log2(post/pre), log2(pre/long),
    log2(local/long), log2(pre/ref), log2(local/ref), log2(long/ref)

Reglas de artefactos y calentamiento:

* Un RR > ``max_rr_s`` (pausa, latidos no detectados) no entra en el historial,
  reinicia el historial local y el reloj de calentamiento; ``pre`` se recorta.
* ``reset()`` (p. ej. electrodo suelto) olvida el último latido, el historial
  local y el latido pendiente; conserva la media de largo plazo.
* Un latido está "caliente" (apto para entrenar, evaluar y alertar) si su
  ``pre`` es válido, hay al menos ``warmup_min_rr`` RR en el historial y han
  pasado ``warmup_s`` desde el inicio o el último reinicio.

Hay dos implementaciones independientes que deben coincidir (prueba P1):
``RrFeatureTracker`` (latido a latido, espejo exacto del firmware C++
``ecg::RrFeatureTracker``) y ``compute_rr_features`` (por lotes, para entrenar).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

RR_FEATURE_NAMES = (
    "log2_pre_local", "log2_post_local", "log2_post_pre", "log2_pre_long",
    "log2_local_long", "log2_pre_ref", "log2_local_ref", "log2_long_ref",
)
N_RR_FEATURES = len(RR_FEATURE_NAMES)


@dataclass(frozen=True)
class RrConfig:
    local_beats: int = 10
    long_tau_s: float = 300.0
    max_rr_s: float = 3.0
    ref_rr_s: float = 0.8
    clip: float = 2.0
    warmup_min_rr: int = 3
    warmup_s: float = 30.0


def rr_transform(pre: float, post: float, local: float, long: float, cfg: RrConfig) -> np.ndarray:
    """Las 8 características para un latido (valores en segundos)."""
    vals = (pre / local, post / local, post / pre, pre / long, local / long,
            pre / cfg.ref_rr_s, local / cfg.ref_rr_s, long / cfg.ref_rr_s)
    return np.clip(np.log2(np.asarray(vals, dtype=np.float64)), -cfg.clip, cfg.clip)


@dataclass
class RrBeat:
    r_index: int
    warm: bool
    features: np.ndarray  # (8,)
    pre_s: float
    post_s: float


class RrFeatureTracker:
    """Versión latido a latido. Cada latido se emite cuando llega el siguiente
    (o al vencer ``max_rr_s`` con ``on_timeout``)."""

    def __init__(self, fs: float, cfg: RrConfig = RrConfig()) -> None:
        self.fs = fs
        self.cfg = cfg
        self._max = cfg.max_rr_s
        self._hist: deque[float] = deque(maxlen=cfg.local_beats)
        self._ema: float | None = None
        self._last: int | None = None
        self._warm_start: int | None = None
        self._pending: tuple[int, float, float, float, bool] | None = None

    def reset(self) -> None:
        self._last = None
        self._warm_start = None
        self._hist.clear()
        self._pending = None

    def _finalize(self, post_s: float) -> RrBeat:
        r, pre, local, long, warm = self._pending  # type: ignore[misc]
        self._pending = None
        post = min(post_s, self._max)
        return RrBeat(r, warm, rr_transform(pre, post, local, long, self.cfg), pre, post)

    def on_beat(self, r: int) -> RrBeat | None:
        if self._last is None:
            self._last = r
            if self._warm_start is None:
                self._warm_start = r
            return None
        pre = (r - self._last) / self.fs
        out = self._finalize(pre) if self._pending is not None else None
        valid = 0.0 < pre <= self._max
        pre_c = min(pre, self._max)
        local = (sum(self._hist) / len(self._hist)) if self._hist else pre_c
        long = self._ema if self._ema is not None else pre_c
        warm = (valid and len(self._hist) >= self.cfg.warmup_min_rr
                and (r - self._warm_start) / self.fs >= self.cfg.warmup_s)
        self._pending = (r, pre_c, local, long, warm)
        if valid:
            self._hist.append(pre)
            self._ema = pre if self._ema is None else (
                self._ema + (1.0 - math.exp(-pre / self.cfg.long_tau_s)) * (pre - self._ema))
        else:
            self._hist.clear()
            self._warm_start = r
        self._last = r
        return out

    @property
    def pending_index(self) -> int | None:
        """Índice R del latido pendiente (el que se emitirá en el próximo
        latido o timeout). El primer latido tras un reinicio solo es ancla y
        nunca queda pendiente."""
        return self._pending[0] if self._pending is not None else None

    def on_timeout(self, now: int) -> RrBeat | None:
        """Emite el latido pendiente si han pasado ``max_rr_s`` sin otro latido."""
        if self._pending is not None and (now - self._pending[0]) / self.fs >= self._max:
            return self._finalize(self._max)
        return None


def compute_rr_features(r_indices: np.ndarray, fs: float, cfg: RrConfig = RrConfig()):
    """Versión por lotes para un registro sin reinicios externos.

    Devuelve ``(features (n, 8), warm (n,), has_features (n,))``. El primer
    latido (sin ``pre``) y el último (sin ``post``) tienen ``has_features=False``.
    """
    r = np.asarray(r_indices, dtype=np.int64)
    n = len(r)
    feats = np.zeros((n, N_RR_FEATURES))
    warm = np.zeros(n, dtype=bool)
    has = np.zeros(n, dtype=bool)
    if n < 3:
        return feats, warm, has
    pre = np.diff(r) / fs                       # pre[k-1] = RR que termina en el latido k
    valid = (pre > 0) & (pre <= cfg.max_rr_s)
    pre_c = np.minimum(pre, cfg.max_rr_s)
    # Media exponencial de largo plazo (secuencial por naturaleza).
    ema_before = np.empty(len(pre))
    ema = None
    for j, (p, ok) in enumerate(zip(pre, valid)):
        ema_before[j] = ema if ema is not None else pre_c[j]
        if ok:
            ema = p if ema is None else ema + (1.0 - math.exp(-p / cfg.long_tau_s)) * (p - ema)
    # Inicio del segmento actual (tras el último RR inválido).
    seg_start = np.zeros(len(pre), dtype=np.int64)  # índice (en pre) del primer RR del segmento
    warm_start = np.zeros(len(pre), dtype=np.int64)  # muestra de inicio del calentamiento
    s, ws = 0, r[0]
    for j in range(len(pre)):
        seg_start[j], warm_start[j] = s, ws
        if not valid[j]:
            s, ws = j + 1, r[j + 1]
    for k in range(1, n - 1):
        j = k - 1
        hist = pre[max(seg_start[j], j - cfg.local_beats):j]
        local = hist.mean() if hist.size else pre_c[j]
        post = min(pre[j + 1], cfg.max_rr_s)
        feats[k] = rr_transform(pre_c[j], post, local, ema_before[j], cfg)
        warm[k] = (valid[j] and hist.size >= cfg.warmup_min_rr
                   and (r[k] - warm_start[j]) / fs >= cfg.warmup_s)
        has[k] = True
    return feats, warm, has


def quantize_int8(x: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Cuantización idéntica a la del firmware: redondeo "mitad lejos de cero"
    de x/scale, más el punto cero, saturado a [-128, 127]."""
    v = np.asarray(x, dtype=np.float64) / scale
    q = np.sign(v) * np.floor(np.abs(v) + 0.5) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)
