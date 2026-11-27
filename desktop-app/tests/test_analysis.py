import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_desktop.analysis import HeartRateEstimator, RPeakDetector  # noqa: E402

FS = 360.0


def pattern(n, bpm, seed=7):
    """Réplica de TestPatternSource (firmware) para probar sin hardware."""
    period = int(FS * 60.0 / bpm)
    half = int(FS * 0.02) + 1
    center = period // 2
    rng = seed
    out = []
    for i in range(n):
        ph = i % period
        v = 0
        if ph + half >= center and ph <= center + half:
            dist = abs(ph - center)
            v = 1000 - (1000 * dist) // half
        rng = (rng * 1664525 + 1013904223) & 0xFFFFFFFF
        v += ((rng >> 24) & 0x1F) - 16
        out.append(v)
    return out


def run(signal):
    det, hr = RPeakDetector(FS), HeartRateEstimator(FS)
    peaks = []
    for x in signal:
        r = det.process(x)
        if r is not None:
            peaks.append(r)
            hr.add_peak(r)
    return peaks, hr


def test_detecta_todos_los_latidos_y_bpm():
    peaks, hr = run(pattern(int(FS * 30), 75))
    period = int(FS * 60 / 75)
    # el primer QRS está en period/2; los de los 2 s de aprendizaje no se detectan
    centros = [period // 2 + k * period for k in range(100) if period // 2 + k * period < int(FS * 30)]
    esperados = [c for c in centros if c >= int(2 * FS)]
    assert len(peaks) == len(esperados)
    assert all(abs(p - c) <= 3 for p, c in zip(peaks, esperados))  # R localizado a ±3 muestras
    assert hr.bpm is not None and abs(hr.bpm - 75) < 1.5


def test_bpm_distintos_ritmos():
    for bpm in (50, 100, 140):
        _, hr = run(pattern(int(FS * 30), bpm))
        assert abs(hr.bpm - bpm) / bpm < 0.03, (bpm, hr.bpm)


def test_sin_senal_no_hay_bpm():
    _, hr = run([0] * int(FS * 10))
    assert hr.bpm is None


def test_pausa_reinicia_el_ritmo():
    hr = HeartRateEstimator(FS)
    for k in range(5):
        hr.add_peak(int(k * 0.8 * FS))
    assert abs(hr.bpm - 75) < 0.5
    hr.add_peak(int(10 * FS))  # pausa de >2 s: ritmo anterior descartado
    assert hr.bpm is None
