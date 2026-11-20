"""simulate_device: réplica Python del pipeline del firmware.

* El latido anterior a una pausa larga se clasifica al vencer el timeout,
  con su ventana guardada (no depende del buffer circular).
* No se pierde ningún latido: todos los detectados se clasifican salvo el
  ancla tras cada reinicio y el último (aún sin RR siguiente).
* Los latidos dentro del calentamiento se marcan como no aptos para alertar.
* Las constantes coinciden con device-firmware/include/integration_config.h.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from simulator.mock_ecg_device import SyntheticEcgSource
from src.preprocessing import PreprocessConfig
from src.preprocessing import pipeline as pl
from src.preprocessing.filters import apply_sos
from src.preprocessing.segmentation import StreamingRDetector

FS = 360.0
CFG = PreprocessConfig()


def _signal(seconds=90, pause=None, lead=None, seed=4):
    x, _ = SyntheticEcgSource(seed=seed, pvc_probability=0.05, sample_rate_hz=FS).read(int(FS * seconds))
    c = np.clip(np.round(x / 0.005), -32768, 32767).astype(np.float64)
    mask = np.zeros(len(c), dtype=bool)
    if lead:
        mask[int(lead[0] * FS):int(lead[1] * FS)] = True
        c[mask] = 0
    if pause:
        c[int(pause[0] * FS):int(pause[1] * FS)] = 0
    return c, mask


def _detections(c, mask):
    y = apply_sos(c, CFG.sos())
    det = StreamingRDetector(FS)
    return [r for i, v in enumerate(y) if (r := det.process(v)) is not None and not mask[i]]


def test_beat_before_long_pause_is_classified_by_timeout():
    c, mask = _signal(pause=(60, 68))
    out = pl.simulate_device(c, FS, CFG, lead_off=mask)
    timeouts = [b for b in out if b["timeout"]]
    assert len(timeouts) >= 1
    det = _detections(c, mask)
    before_pause = max(r for r in det if r < 60 * FS)
    t = next(b for b in timeouts if b["r_index"] == before_pause)
    assert t["classify_at"] - t["r_index"] >= int(CFG.rr_max_s * FS)  # esperó al timeout
    # Su ventana es la del latido (z-score de 252 muestras con el R en su sitio).
    assert t["beat"].shape == (CFG.input_length,) and int(np.argmax(t["beat"])) == pytest.approx(CFG.window[0], abs=3)
    # La ventana salió de la copia, no del buffer circular (que ya no la tendría).
    assert t["classify_at"] + 1 - (t["r_index"] - CFG.window[0]) > pl.DEVICE_RING_SIZE


def test_no_beat_is_lost():
    c, mask = _signal(pause=(60, 68), lead=(35, 37))
    out = pl.simulate_device(c, FS, CFG, lead_off=mask)
    det = _detections(c, mask)
    classified = {b["r_index"] for b in out}
    anchors = {det[0], min(r for r in det if r > 37 * FS)}   # inicio y tras el electrodo suelto
    lost_at_lead_off = {max(r for r in det if r < 35 * FS)}  # pendiente al soltarse el electrodo
    expected = set(det) - anchors - lost_at_lead_off - {det[-1]}
    assert classified == expected


def test_warmup_beats_are_not_alert_eligible():
    c, mask = _signal(pause=(60, 68))
    out = pl.simulate_device(c, FS, CFG, lead_off=mask)
    for b in out:
        if b["r_index"] < CFG.rr_warmup_s * FS:
            assert not b["warm"]
    # Tras la pausa (> rr_max_s) el calentamiento vuelve a empezar.
    after = [b for b in out if 68 * FS < b["r_index"] < (68 + CFG.rr_warmup_s - 1) * FS]
    assert after and not any(b["warm"] for b in after)
    assert any(b["warm"] for b in out)


def test_window_larger_than_device_buffer_is_rejected():
    with pytest.raises(ValueError):
        pl.simulate_device(np.zeros(100), 500.0, PreprocessConfig(fs_hz=500.0))


def test_constants_match_firmware():
    h = (Path(__file__).resolve().parents[2] / "device-firmware" / "include" / "integration_config.h").read_text(
        encoding="utf-8")

    def define(name):
        return int(re.search(rf"#define\s+{name}\s+(\d+)", h).group(1))

    assert define("ECG_BEAT_RING_SIZE") == pl.DEVICE_RING_SIZE
    assert define("ECG_MAX_PENDING_BEATS") == pl.DEVICE_MAX_PENDING
    assert define("ECG_MAX_BEAT_WINDOW") == pl.DEVICE_MAX_WINDOW
