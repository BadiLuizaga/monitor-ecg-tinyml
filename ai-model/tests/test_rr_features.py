"""P1, P2 y P3b: características RR.

P1  la versión por lotes (entrenamiento) y la latido a latido (espejo del
    firmware) coinciden en MIT-BIH (solo registros de DS1) y en señales sintéticas.
P2  valores calculados a mano: recorte, pausas, reinicio y calentamiento.
P3b causalidad: alterar latidos posteriores a k+1 no cambia las características de k.
"""

import math

import numpy as np
import pytest

from src.preprocessing import PreprocessConfig
from src.preprocessing.rr_features import (
    N_RR_FEATURES, RR_FEATURE_NAMES, RrConfig, RrFeatureTracker, compute_rr_features, quantize_int8, rr_transform)
from src.training import dataset_loader as dl

FS = 360.0
CFG = RrConfig()


def _streaming(r, fs=FS, cfg=CFG):
    tr = RrFeatureTracker(fs, cfg)
    return {b.r_index: b for b in (tr.on_beat(int(x)) for x in r) if b is not None}


def _assert_same(r, fs=FS, cfg=CFG):
    feats, warm, has = compute_rr_features(r, fs, cfg)
    st = _streaming(r, fs, cfg)
    idx = np.flatnonzero(has)
    assert set(int(r[k]) for k in idx) == set(st)
    for k in idx:
        b = st[int(r[k])]
        assert np.max(np.abs(b.features - feats[k])) < 1e-9
        assert b.warm == warm[k]
    return feats, warm, has


def _synthetic_r(seed=0, n=600, with_events=True):
    rng = np.random.default_rng(seed)
    rr = rng.normal(0.8, 0.05, n)
    rr[rng.random(n) < 0.08] *= 0.65           # prematuros
    if with_events:
        rr[100] = 4.5                           # pausa > max_rr
        rr[300] = 3.0                           # exactamente en el límite (válido)
        rr[301] = 0.21                          # casi refractario
    return np.concatenate([[100], 100 + np.cumsum(np.round(rr * FS))]).astype(np.int64)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_p1_batch_equals_streaming_synthetic(seed):
    _assert_same(_synthetic_r(seed))


@pytest.mark.parametrize("record", [201, 208, 209])  # DS1: con pausas, F y S
def test_p1_batch_equals_streaming_mitbih(record):
    if not (dl.DEFAULT_DATA_DIR / f"{record}.dat").exists():
        pytest.skip("MIT-BIH no descargado")
    rd = dl.load_record(record)
    r = rd.ann_samples[np.isin(rd.ann_symbols, list(dl.BEAT_SYMBOLS))]
    feats, warm, has = _assert_same(r, rd.fs)
    assert warm.sum() > 0.9 * has.sum()


def test_p2_hand_computed_values():
    # RR: 1.0, 1.0, 1.0, 0.5 (prematuro), 1.5 (compensatorio) s a 100 Hz
    fs = 100.0
    r = np.array([0, 100, 200, 300, 350, 500])
    cfg = RrConfig(warmup_s=0.0, warmup_min_rr=2)
    feats, warm, has = compute_rr_features(r, fs, cfg)
    k = 4  # el latido prematuro: pre=0.5, post=1.5, local=mean(1,1,1)=1
    a = 1 - math.exp(-1 / 300)
    ema = 1.0 + a * (1.0 - 1.0)  # tras 1.0, 1.0, 1.0 -> 1.0
    expected = rr_transform(0.5, 1.5, 1.0, ema, cfg)
    assert np.allclose(feats[k], expected)
    assert np.allclose(feats[k][:3], [-1.0, math.log2(1.5), math.log2(3.0)])
    assert warm[k] and not has[0] and not has[-1]
    assert not warm[1]  # sin historial suficiente


def test_p2_clip_and_pause_reset():
    fs = 100.0
    cfg = RrConfig(warmup_s=0.0, warmup_min_rr=1)
    r = np.array([0, 100, 200, 300, 900, 1000, 1100, 1200])  # pausa de 6 s
    feats, warm, has = compute_rr_features(r, fs, cfg)
    assert not warm[4]                       # pre inválido -> no apto
    assert feats[4][5] == pytest.approx(math.log2(3.0 / 0.8))  # pre recortado a 3 s
    assert feats[3][1] == pytest.approx(math.log2(3.0 / 1.0))  # post recortado a 3 s
    assert not warm[5] and warm[6]           # el historial local se reinició
    assert np.all(np.abs(feats) <= cfg.clip)


def test_p2_warmup_by_time():
    fs = 100.0
    r = np.arange(0, 100 * 60, 100)  # 1 latido/s durante 60 s
    feats, warm, has = compute_rr_features(r, fs, CFG)
    first_warm = np.flatnonzero(warm)[0]
    assert (r[first_warm] - r[0]) / fs == pytest.approx(CFG.warmup_s)


def test_p2_streaming_reset_and_timeout():
    tr = RrFeatureTracker(FS, CFG)
    for x in (0, 360, 720):
        tr.on_beat(x)
    b = tr.on_timeout(720 + int(3 * FS))  # sin latido en 3 s
    assert b is not None and b.r_index == 720 and b.post_s == pytest.approx(3.0)
    assert tr.on_timeout(720 + int(4 * FS)) is None
    tr.reset()
    assert tr.on_beat(5000) is None          # tras reinicio, el primer latido solo ancla
    b2 = tr.on_beat(5360)
    assert b2 is None                        # el latido 5360 queda pendiente
    b3 = tr.on_beat(5720)
    assert b3.r_index == 5360 and not b3.warm


def test_p3b_causality():
    r = _synthetic_r(5, with_events=False)
    feats, _, _ = compute_rr_features(r, FS, CFG)
    k = 200
    r2 = r.copy()
    r2[k + 2:] += 37  # altera todo lo posterior a k+1
    feats2, _, _ = compute_rr_features(r2, FS, CFG)
    assert np.array_equal(feats[:k + 1], feats2[:k + 1])


def test_feature_names_and_config_roundtrip(tmp_path):
    assert len(RR_FEATURE_NAMES) == N_RR_FEATURES == 8
    cfg = PreprocessConfig()
    assert cfg.rr_config() == RrConfig()
    cfg.save(tmp_path / "c.json")
    assert PreprocessConfig.load(tmp_path / "c.json").rr_config() == cfg.rr_config()


def test_quantize_rounds_half_away_from_zero():
    q = quantize_int8(np.array([0.5, -0.5, 1.5, -1.5, 1000.0, -1000.0]), 1.0, 0)
    assert q.tolist() == [1, -1, 2, -2, 127, -128]
    assert quantize_int8(np.array([0.25]), 0.5, -3).tolist() == [-2]


def test_f_weight_cap_only_limits_f():
    import numpy as np
    from src.training.rr_experiment import IDX, class_weights

    y = np.array([0] * 1000 + [1] * 50 + [2] * 100 + [3] * 5)
    base = class_weights(y, "balanced")
    capped = class_weights(y, "balanced", f_cap=5.0)
    assert base[IDX["F"]] > 5.0 and capped[IDX["F"]] == 5.0
    assert all(capped[k] == base[k] for k in base if k != IDX["F"])
    assert len(capped) == 4  # F sigue siendo una clase independiente
