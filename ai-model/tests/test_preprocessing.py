import numpy as np
import pytest

from simulator.mock_ecg_device import SyntheticEcgSource
from src.preprocessing import PreprocessConfig, beats_from_signal, filter_signal
from src.preprocessing.normalization import minmax, normalize, zscore
from src.preprocessing.segmentation import detect_r_peaks, extract_beats, match_peaks, refine_peaks
from src.training import dataset_loader as dl


def _synthetic(seconds=60, seed=7, **kw):
    src = SyntheticEcgSource(seed=seed, **kw)
    x, ann = src.read(int(src.sample_rate_hz * seconds))
    return x, np.array([a.sample_index for a in ann]), np.array([a.symbol for a in ann])


def test_extract_beats_shape_and_edges():
    x = np.arange(1000, dtype=float)
    beats, kept = extract_beats(x, np.array([10, 500, 990]), fs=100.0, before_s=0.25, after_s=0.45)
    assert beats.shape == (1, 70)
    assert kept.tolist() == [1]
    assert beats[0, 25] == 500  # el pico R queda en la posición "before"


def test_refine_peaks_moves_to_local_max():
    x = np.zeros(100)
    x[52] = 1.0
    assert refine_peaks(x, np.array([48]), fs=100.0, search_s=0.05).tolist() == [52]


def test_zscore_and_minmax():
    beats = np.random.default_rng(0).normal(3, 5, size=(4, 50))
    z = zscore(beats)
    assert np.allclose(z.mean(axis=1), 0, atol=1e-5) and np.allclose(z.std(axis=1), 1, atol=1e-4)
    m = minmax(beats)
    assert np.allclose(m.min(axis=1), -1) and np.allclose(m.max(axis=1), 1)
    assert np.all(np.isfinite(zscore(np.zeros((1, 10)))))  # señal plana: sin división por cero
    with pytest.raises(ValueError):
        normalize(beats, "desconocida")


def test_zscore_is_gain_invariant():
    """Clave mientras no conozcamos la ganancia del ADC real."""
    beat = np.random.default_rng(1).normal(size=60)
    assert np.allclose(zscore(beat), zscore(beat * 1234.5 + 77), atol=1e-4)


@pytest.mark.parametrize("hr,pvc,noise", [(60, 0.0, 0.02), (100, 0.1, 0.05), (45, 0.05, 0.08)])
def test_r_detector_on_synthetic_ecg(hr, pvc, noise):
    x, ref, _ = _synthetic(heart_rate_bpm=hr, pvc_probability=pvc, noise_mv=noise)
    cfg = PreprocessConfig()
    det = detect_r_peaks(filter_signal(x, 360.0, cfg), 360.0)
    skip = 2 * 360
    tp, fp, fn = match_peaks(det[det > skip], ref[ref > skip], tolerance=int(0.15 * 360))
    assert tp / (tp + fn) > 0.97 and tp / (tp + fp) > 0.97


def test_match_peaks_counts():
    assert match_peaks(np.array([10, 50, 300]), np.array([12, 100, 298]), tolerance=5) == (2, 1, 1)


def test_pipeline_produces_model_input():
    x, ref, _ = _synthetic(seconds=20)
    cfg = PreprocessConfig()
    beats, kept = beats_from_signal(x, 360.0, ref, cfg)
    assert beats.shape == (len(kept), cfg.input_length) and beats.dtype == np.float32
    assert np.allclose(beats.mean(axis=1), 0, atol=1e-4)


def test_pipeline_resamples_to_model_rate():
    x, ref, _ = _synthetic(seconds=20)
    cfg = PreprocessConfig(fs_hz=250.0)
    beats, _ = beats_from_signal(x, 360.0, ref, cfg)
    assert beats.shape[1] == cfg.input_length == round(0.25 * 250) + round(0.45 * 250)


def test_config_roundtrip(tmp_path):
    cfg = PreprocessConfig(fs_hz=250.0, notch_hz=60.0)
    cfg.save(tmp_path / "cfg.json")
    assert PreprocessConfig.load(tmp_path / "cfg.json") == cfg


def test_inter_patient_split_is_disjoint_and_excludes_paced():
    assert not set(dl.DS1) & set(dl.DS2)
    assert not set(dl.PACED) & (set(dl.DS1) | set(dl.DS2))
    assert len(dl.DS1) == len(dl.DS2) == 22 and len(dl.ALL_RECORDS) == 48


def test_aami_mapping():
    assert {dl.AAMI_MAP[s] for s in "NLRej"} == {"N"}
    assert {dl.AAMI_MAP[s] for s in "AaJS"} == {"S"}
    assert dl.AAMI_MAP["V"] == dl.AAMI_MAP["E"] == "V"
    assert "+" not in dl.AAMI_MAP  # anotaciones de ritmo, no latidos
