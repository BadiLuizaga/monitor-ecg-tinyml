"""Genera vectores de prueba Python -> C++ para test_host.cpp.

Produce ``device-firmware/build/test_vectors.h`` con una señal sintética
(cuantizada a int16), la salida esperada del filtro (scipy.sosfilt) y los
picos R esperados (StreamingRDetector de Python). Así se comprueba que el
firmware y el entrenamiento procesan la señal de la misma forma.

Uso (desde la raíz del repositorio):
    python device-firmware/test/host/make_vectors.py
"""

from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "ai-model"), str(REPO / "communication")]

from simulator.mock_ecg_device import SyntheticEcgSource  # noqa: E402
from src.preprocessing import PreprocessConfig  # noqa: E402
from src.preprocessing.filters import apply_sos  # noqa: E402
from src.preprocessing.segmentation import detect_r_peaks  # noqa: E402
from src.export.export_model import c_float  # noqa: E402
from src.preprocessing.pipeline import simulate_device  # noqa: E402
from src.preprocessing.rr_features import RrFeatureTracker, quantize_int8  # noqa: E402

FS = 360.0
SECONDS = 20
C3_SECONDS = 110
RESET = -1


def main() -> None:
    cfg = PreprocessConfig(fs_hz=FS)
    x_mv, _ = SyntheticEcgSource(seed=11, pvc_probability=0.1, sample_rate_hz=FS).read(int(FS * SECONDS))
    counts = np.clip(np.round(x_mv / 0.005), -32768, 32767).astype(np.int16)
    y = apply_sos(counts.astype(float), cfg.sos(), causal=True)
    peaks = detect_r_peaks(y, FS)

    out = REPO / "device-firmware" / "build" / "test_vectors.h"
    out.parent.mkdir(parents=True, exist_ok=True)

    def arr(name, ctype, values, fmt):
        body = ",".join(fmt(v) for v in values)
        return f"static const {ctype} {name}[] = {{{body}}};\nstatic const unsigned {name}_LEN = {len(values)};\n"

    # --- C1: RrFeatureTracker (latidos con prematuros, pausas y un reinicio) ---
    rng = np.random.default_rng(3)
    rr = rng.normal(0.8, 0.05, 700)
    rr[rng.random(700) < 0.08] *= 0.65
    rr[100], rr[300], rr[301], rr[500] = 4.5, 3.0, 0.21, 2.95
    r_seq = np.concatenate([[100], 100 + np.cumsum(np.round(rr * FS))]).astype(np.int64)
    events = list(r_seq[:400]) + [RESET] + list(r_seq[400:] + 5000)
    tracker = RrFeatureTracker(FS, cfg.rr_config())
    c1 = []
    for e in events:
        if e == RESET:
            tracker.reset()
        else:
            b = tracker.on_beat(int(e))
            if b is not None:
                c1.append(b)

    # --- C2: cuantización int8 (incluye casos exactamente a mitad) ---
    xs = np.concatenate([rng.normal(0, 3, 60), [0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 300.0, -300.0, 0.0]])
    xs = xs.astype(np.float32)
    scales = np.array([0.0461284779, 0.0156673305, 1.0, 0.5], dtype=np.float32)
    zps = [-18, -1, 0, 3]
    c2 = np.concatenate([quantize_int8(xs.astype(np.float64), float(s), z) for s, z in zip(scales, zps)])

    # --- C3: pipeline completo (latido + RR) con electrodo suelto y una pausa ---
    x3, _ = SyntheticEcgSource(seed=21, pvc_probability=0.1, sample_rate_hz=FS).read(int(FS * C3_SECONDS))
    c3_counts = np.clip(np.round(x3 / 0.005), -32768, 32767).astype(np.int16)
    lead = np.zeros(len(c3_counts), dtype=bool)
    lead[int(40 * FS):int(42 * FS)] = True
    c3_counts[lead] = 0
    c3_counts[int(95 * FS):int(103 * FS)] = 0  # pausa de 8 s sin latidos: fuerza onTimeout
    c3 = simulate_device(c3_counts.astype(np.float64), FS, cfg, lead_off=lead)
    c3_timeouts = sum(b["timeout"] for b in c3)
    if c3_timeouts < 1:
        raise SystemExit("C3: la señal de prueba no produce ninguna emisión por timeout")

    out.write_text(
        "// GENERADO por make_vectors.py\n#pragma once\n"
        f"#define TV_FS {FS}f\n"
        + arr("TV_INPUT", "short", counts, str)
        + arr("TV_FILTERED", "float", y, c_float)
        + arr("TV_PEAKS", "unsigned", peaks, str)
        + f"#define TV_RESET 0xFFFFFFFFu\n"
        + arr("TV_C1_EVENTS", "unsigned", [int(e) & 0xFFFFFFFF for e in events], str)
        + arr("TV_C1_R", "unsigned", [b.r_index for b in c1], str)
        + arr("TV_C1_WARM", "unsigned char", [int(b.warm) for b in c1], str)
        + arr("TV_C1_FEAT", "float", np.concatenate([b.features for b in c1]), c_float)
        + arr("TV_C2_X", "float", xs, c_float)
        + arr("TV_C2_SCALE", "float", scales, c_float)
        + arr("TV_C2_ZP", "int", zps, str)
        + arr("TV_C2_Q", "signed char", c2, str)
        + arr("TV_C3_INPUT", "short", c3_counts, str)
        + arr("TV_C3_LEADOFF", "unsigned char", lead.astype(int), str)
        + arr("TV_C3_R", "unsigned", [b["r_index"] for b in c3], str)
        + arr("TV_C3_WARM", "unsigned char", [int(b["warm"]) for b in c3], str)
        + arr("TV_C3_AT", "unsigned", [b["classify_at"] for b in c3], str)
        + arr("TV_C3_TIMEOUT", "unsigned char", [int(b["timeout"]) for b in c3], str)
        + arr("TV_C3_RR", "float", np.concatenate([b["rr"] for b in c3]), c_float)
        + arr("TV_C3_BEAT", "float", np.concatenate([b["beat"] for b in c3]), c_float),
        encoding="utf-8")
    print(f"{out}: {len(counts)} muestras, {len(peaks)} picos; C1 {len(c1)} latidos; "
          f"C2 {len(c2)} valores; C3 {len(c3)} latidos clasificados "
          f"({sum(b['warm'] for b in c3)} calientes, {c3_timeouts} por timeout)")


if __name__ == "__main__":
    main()
