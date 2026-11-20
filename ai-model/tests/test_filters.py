import numpy as np
import pytest

from src.preprocessing import filters

FS = 360.0


def _tone(f_hz: float, seconds: float = 10.0, fs: float = FS) -> np.ndarray:
    t = np.arange(int(seconds * fs)) / fs
    return np.sin(2 * np.pi * f_hz * t)


def _gain(y: np.ndarray, x: np.ndarray, skip: int = 2 * int(FS)) -> float:
    """Ganancia RMS ignorando el transitorio inicial del filtro causal."""
    return float(np.sqrt(np.mean(y[skip:] ** 2) / np.mean(x[skip:] ** 2)))


def test_bandpass_passes_qrs_band():
    x = _tone(10.0)
    assert _gain(filters.bandpass_filter(x, FS), x) == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("f_hz", [0.05, 100.0])
def test_bandpass_rejects_baseline_and_high_freq(f_hz):
    x = _tone(f_hz, seconds=60.0)
    assert _gain(filters.bandpass_filter(x, FS), x) < 0.1


@pytest.mark.parametrize("mains_hz", [50.0, 60.0])
def test_notch_removes_mains(mains_hz):
    x = _tone(mains_hz)
    y = filters.apply_sos(x, filters.design_notch(FS, mains_hz))
    assert _gain(y, x) < 0.05


def test_preprocessing_cascade_includes_notch():
    sos = filters.design_preprocessing_sos(FS, notch_hz=50.0)
    assert sos.shape == (len(filters.design_bandpass(FS)) + 1, 6)


def test_zero_phase_has_no_delay_and_causal_has_delay():
    x = np.zeros(2000)
    x[1000] = 1.0
    sos = filters.design_bandpass(FS)
    assert np.argmax(np.abs(filters.apply_sos(x, sos, causal=False))) == 1000
    assert np.argmax(np.abs(filters.apply_sos(x, sos, causal=True))) > 1000


@pytest.mark.parametrize("band", [(0.0, 40.0), (40.0, 10.0), (0.5, 200.0)])
def test_invalid_band_raises(band):
    with pytest.raises(ValueError):
        filters.design_bandpass(FS, *band)


@pytest.mark.parametrize("fs_out", [250.0, 500.0, 128.0])
def test_resample_length_and_indices(fs_out):
    x = _tone(5.0, seconds=4.0)
    y = filters.resample_signal(x, FS, fs_out)
    assert len(y) == pytest.approx(len(x) * fs_out / FS, abs=1)
    assert filters.resample_indices(np.array([360]), FS, fs_out)[0] == round(fs_out)
