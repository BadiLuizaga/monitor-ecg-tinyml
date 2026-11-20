"""Configuración y cadena de preprocesamiento compartida entrenamiento / firmware.

``PreprocessConfig`` se guarda junto al modelo (``models/preprocess_config.json``)
y se exporta a C++ para que el ESP32 aplique exactamente la misma cadena:

    señal -> (remuestreo) -> filtro causal -> pico R -> ventana -> z-score -> modelo
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import filters, normalization, segmentation
from .rr_features import RrConfig


@dataclass
class PreprocessConfig:
    # PROVISIONAL: 360 Hz es la frecuencia de MIT-BIH. Debe ajustarse a la
    # frecuencia real del sensor cuando se confirme (y reentrenar).
    fs_hz: float = 360.0
    band_hz: tuple[float, float] = filters.DEFAULT_BAND_HZ
    filter_order: int = filters.DEFAULT_ORDER
    notch_hz: float | None = None
    causal: bool = True
    before_s: float = segmentation.WINDOW_BEFORE_S
    after_s: float = segmentation.WINDOW_AFTER_S
    normalization: str = "zscore"
    refine_search_s: float = 0.05
    # Características RR (ver rr_features.py). Valores fijos, no ajustados a datos.
    rr_local_beats: int = 10
    rr_long_tau_s: float = 300.0
    rr_max_s: float = 3.0
    rr_ref_s: float = 0.8
    rr_clip: float = 2.0
    rr_warmup_min_rr: int = 3
    rr_warmup_s: float = 30.0

    def rr_config(self) -> RrConfig:
        return RrConfig(self.rr_local_beats, self.rr_long_tau_s, self.rr_max_s, self.rr_ref_s,
                        self.rr_clip, self.rr_warmup_min_rr, self.rr_warmup_s)

    @property
    def window(self) -> tuple[int, int]:
        return segmentation.window_lengths(self.fs_hz, self.before_s, self.after_s)

    @property
    def input_length(self) -> int:
        return sum(self.window)

    def sos(self) -> np.ndarray:
        return filters.design_preprocessing_sos(self.fs_hz, self.band_hz, self.filter_order, self.notch_hz)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["band_hz"] = list(self.band_hz)
        d["input_length"] = self.input_length
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessConfig":
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "band_hz" in fields:
            fields["band_hz"] = tuple(fields["band_hz"])
        return cls(**fields)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PreprocessConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def filter_signal(x: np.ndarray, fs_in: float, cfg: PreprocessConfig) -> np.ndarray:
    """Remuestrea a ``cfg.fs_hz`` (si hace falta) y filtra."""
    x = filters.resample_signal(x, fs_in, cfg.fs_hz)
    return filters.apply_sos(x, cfg.sos(), causal=cfg.causal)


def beats_from_signal(x: np.ndarray, fs_in: float, r_peaks: np.ndarray, cfg: PreprocessConfig,
                      refine: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Señal cruda + picos R (en ``fs_in``) -> latidos normalizados ``(n, L)``.

    Devuelve también los índices de los picos conservados.
    """
    xf = filter_signal(x, fs_in, cfg)
    peaks = filters.resample_indices(r_peaks, fs_in, cfg.fs_hz)
    if refine:
        peaks = segmentation.refine_peaks(xf, peaks, cfg.fs_hz, cfg.refine_search_s)
    beats, kept = segmentation.extract_beats(xf, peaks, cfg.fs_hz, cfg.before_s, cfg.after_s)
    return normalization.normalize(beats, cfg.normalization), kept


# Constantes del firmware (device-firmware/include/integration_config.h).
DEVICE_RING_SIZE = 1024       # ECG_BEAT_RING_SIZE: muestras filtradas en el buffer circular
DEVICE_MAX_PENDING = 4        # ECG_MAX_PENDING_BEATS: latidos esperando ventana y/o RR
DEVICE_MAX_WINDOW = 256       # ECG_MAX_BEAT_WINDOW: muestras guardadas por latido pendiente


def simulate_device(x: np.ndarray, fs: float, cfg: PreprocessConfig, lead_off: np.ndarray | None = None,
                    prefiltered: bool = False) -> list[dict]:
    """Réplica muestra a muestra de ``EcgPipeline`` (device-firmware/src/ecg_pipeline.cpp).

    Cada latido que el ``RrFeatureTracker`` deja pendiente ocupa una entrada de
    una cola (máx. ``DEVICE_MAX_PENDING``; si se llena se descarta la más
    antigua). La ventana del latido se COPIA en su entrada en cuanto está
    completa (0,45 s tras el R), así que no depende de que siga en el buffer
    circular aunque el RR siguiente tarde (pausas: emisión por timeout a los
    ``rr_max_s``). Se clasifica, en orden, cuando la entrada tiene ventana y RR.

    Orden por muestra (idéntico en C++):
      1. transición de electrodo -> vaciar cola y reiniciar RR;
      2. detección -> on_beat (puede emitir el RR del anterior); si el nuevo
         latido queda pendiente, nueva entrada;
      3. on_timeout;
      4. copiar ventanas que se completan;
      5. clasificar las entradas completas del frente de la cola.

    Devuelve, por cada latido clasificado: índice R, si está caliente, RR,
    ventana normalizada, muestra en la que se clasifica y si su RR salió por
    timeout. Es la referencia de la prueba C3 y de la evaluación "realista".
    """
    from .rr_features import RrFeatureTracker

    y = np.asarray(x, dtype=np.float64) if prefiltered else filters.apply_sos(x, cfg.sos(), causal=True)
    nb, na = cfg.window
    if nb + na > DEVICE_MAX_WINDOW:
        raise ValueError(f"ventana de {nb + na} muestras > {DEVICE_MAX_WINDOW} (CONFIG_MISMATCH en el firmware)")
    det = segmentation.StreamingRDetector(fs)
    rr = RrFeatureTracker(fs, cfg.rr_config())
    lo_mask = np.zeros(len(y), dtype=bool) if lead_off is None else np.asarray(lead_off, dtype=bool)
    queue: list[dict] = []   # {"r", "window" | None, "rr": RrBeat | None, "timeout"}
    out: list[dict] = []
    lead = False

    def attach(b, timeout: bool) -> None:
        for e in queue:
            if e["r"] == b.r_index:
                e["rr"], e["timeout"] = b, timeout
                return

    for idx in range(len(y)):
        if lo_mask[idx] != lead:                                   # 1
            lead = bool(lo_mask[idx])
            queue.clear()
            rr.reset()
        r = det.process(y[idx])
        if r is not None and not lead:                             # 2
            b = rr.on_beat(r)
            if b is not None:
                attach(b, False)
            if rr.pending_index == r:
                if len(queue) == DEVICE_MAX_PENDING:
                    queue.pop(0)
                queue.append({"r": r, "window": None, "rr": None, "timeout": False})
        if not lead:                                               # 3
            b = rr.on_timeout(idx)
            if b is not None:
                attach(b, True)
        for e in list(queue):                                      # 4
            if e["window"] is None and idx + 1 >= e["r"] + na:
                start = e["r"] - nb
                if start < 0 or idx + 1 - start > DEVICE_RING_SIZE:
                    queue.remove(e)                                # ventana no disponible
                else:
                    e["window"] = y[start:e["r"] + na].astype(np.float64).copy()
        while queue and queue[0]["window"] is not None and queue[0]["rr"] is not None:   # 5
            e = queue.pop(0)
            b = e["rr"]
            out.append({"r_index": e["r"], "warm": b.warm, "rr": b.features,
                        "beat": normalization.zscore(e["window"]), "classify_at": idx,
                        "timeout": e["timeout"]})
    return out
