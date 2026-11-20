"""Pruebas de la ruta modelo -> TFLite int8 -> archivos para el firmware.

Usan datos sintéticos: verifican que la ruta funciona, NO la calidad del
modelo (eso lo mide src/evaluation/evaluate.py con MIT-BIH DS2).
"""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from src.evaluation.metrics import classification_report, confusion_matrix  # noqa: E402
from src.export import export_model as ex  # noqa: E402
from src.preprocessing import PreprocessConfig  # noqa: E402
from src.training.train import (  # noqa: E402
    balanced_class_weights, build_model, cap_per_class, stratified_representative)

L = PreprocessConfig().input_length


@pytest.fixture(scope="module")
def trained():
    rng = np.random.default_rng(0)
    t = np.linspace(-1, 1, L)
    # Dos "morfologías" separables: pico estrecho vs pico ancho.
    X = np.concatenate([np.exp(-(t / 0.03) ** 2) + 0.1 * rng.standard_normal((200, L)),
                        np.exp(-(t / 0.15) ** 2) + 0.1 * rng.standard_normal((200, L))]).astype(np.float32)
    y = np.repeat([0, 1], 200)
    tf.keras.utils.set_random_seed(0)
    model = build_model(L, 2)
    # Suficientes pasos para que se estabilicen las estadísticas de BatchNorm.
    model.fit(X[..., None], y, epochs=30, batch_size=16, verbose=0)
    return model, X, y


def test_model_is_small(trained):
    model, _, _ = trained
    assert model.output_shape == (None, 2)
    assert model.count_params() < 15_000


def test_int8_conversion_is_tflm_compatible(trained):
    model, X, y = trained
    tfl = ex.convert_to_tflite(model, X[stratified_representative(y, 50, np.random.default_rng(0))], int8=True)
    ok, ops, unsupported = ex.check_tflm_compatibility(tfl)
    assert ok, unsupported
    assert "CONV_2D" in ops and "FULLY_CONNECTED" in ops
    qp = ex.quant_params(tfl)
    assert qp["input_dtype"] == qp["output_dtype"] == "int8"
    assert qp["input_shape"] == [1, L, 1]
    assert len(tfl) < 50_000


def test_int8_predictions_match_float(trained):
    model, X, y = trained
    tfl = ex.convert_to_tflite(model, X[stratified_representative(y, 50, np.random.default_rng(0))], int8=True)
    p8 = ex.predict_tflite(tfl, X).argmax(1)
    pf = model.predict(X[..., None], verbose=0).argmax(1)
    assert np.mean(p8 == pf) > 0.95
    assert np.mean(pf == y) > 0.9


def test_int8_requires_representative_data(trained):
    with pytest.raises(ValueError):
        ex.convert_to_tflite(trained[0], None, int8=True)


def test_firmware_sources_are_generated(trained, tmp_path):
    model, X, _ = trained
    tfl = ex.convert_to_tflite(model, X[:50], int8=True)
    h, cc = ex.write_model_sources(tfl, tmp_path)
    text = cc.read_text()
    assert "alignas(16)" in text and f"g_ecg_model_data_len = {len(tfl)};" in text
    assert text.count("0x") == len(tfl)
    cfg = PreprocessConfig()
    mc = ex.write_model_config(ex.quant_params(tfl), {"classes": ["N", "V"]}, cfg, tmp_path).read_text()
    assert f"ECG_MODEL_INPUT_LENGTH     {L}" in mc and "{'N', 'V'}" in mc
    fc = ex.write_filter_coeffs(cfg, tmp_path).read_text()
    assert f"ECG_FILTER_NUM_SECTIONS {len(cfg.sos())}" in fc


def test_class_weights_and_capping():
    y = np.array([0] * 90 + [1] * 10)
    w = balanced_class_weights(y, 3)
    assert w[1] > w[0] and w[2] == 0.0
    X = np.zeros((100, 3))
    _, yc = cap_per_class(X, y, 20, np.random.default_rng(0))
    assert np.bincount(yc).tolist() == [20, 10]


def test_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 1, 0])
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm.tolist() == [[2, 1, 0], [0, 2, 0], [1, 0, 0]]
    rep = classification_report(y_true, y_pred, ("N", "S", "V"))
    assert rep["per_class"]["S"]["sensitivity"] == 1.0
    assert rep["per_class"]["S"]["ppv"] == pytest.approx(2 / 3)
    assert rep["per_class"]["V"]["f1"] == 0.0
    assert rep["accuracy"] == pytest.approx(4 / 6)


def test_split_tail_keeps_every_record_in_training():
    from src.training.train import split_tail

    rec = np.array([1] * 10 + [2] * 20)
    val = split_tail(rec, 0.2)
    assert val.tolist() == [False] * 8 + [True] * 2 + [False] * 16 + [True] * 4


# --- P4 / P5: modelo de dos entradas (latido + RR) -------------------------

@pytest.fixture(scope="module")
def trained_rr():
    from src.preprocessing.rr_features import N_RR_FEATURES

    rng = np.random.default_rng(1)
    t = np.linspace(-1, 1, L)
    n = 200
    X = (np.exp(-(t / 0.03) ** 2) + 0.1 * rng.standard_normal((2 * n, L))).astype(np.float32)
    # Misma morfología; solo el RR distingue las clases (como un latido S).
    R = rng.normal(0, 0.1, (2 * n, N_RR_FEATURES)).astype(np.float32)
    R[n:, 0] -= 0.8  # log2(pre/local) negativo: prematuro
    y = np.repeat([0, 1], n)
    tf.keras.utils.set_random_seed(0)
    model = build_model(L, 2, rr_features=N_RR_FEATURES)
    model.fit([X[..., None], R], y, epochs=30, batch_size=16, verbose=0)
    # Calibración representativa de TODAS las clases (igual que train.py).
    rep = stratified_representative(y, 50, np.random.default_rng(0))
    return model, X, R, y, rep


def test_p4_rr_model_converts_int8_and_is_tflm_compatible(trained_rr):
    model, X, R, _, rep = trained_rr
    tfl = ex.convert_to_tflite(model, X[rep], int8=True, representative_rr=R[rep])
    ok, ops, unsupported = ex.check_tflm_compatibility(tfl)
    assert ok, unsupported
    assert "CONCATENATION" in ops
    for q in ex.concat_quantization(tfl):  # TFLM exige la misma cuantización
        assert len(set(q)) == 1, q
    qp = ex.quant_params(tfl)
    assert qp["input_shape"] == [1, L, 1] and qp["rr_shape"] == [1, 8]
    assert qp["input_dtype"] == qp["rr_dtype"] == qp["output_dtype"] == "int8"
    assert {qp["beat_input_position"], qp["rr_input_position"]} == {0, 1}
    assert len(tfl) < 64 * 1024


def test_p4_rr_model_uses_rr_and_int8_matches_float(trained_rr):
    model, X, R, y, rep = trained_rr
    tfl = ex.convert_to_tflite(model, X[rep], int8=True, representative_rr=R[rep])
    pf = model.predict([X[..., None], R], verbose=0).argmax(1)
    p8 = ex.predict_tflite(tfl, X, R).argmax(1)
    assert np.mean(pf == y) > 0.9          # solo el RR permite separar las clases
    assert np.mean(p8 == pf) >= 0.98
    sat = ex.input_saturation(tfl, X, R)   # sobre TODOS los datos, no solo la calibración
    assert sat["beat"] < ex.MAX_INPUT_SATURATION and sat["rr"] < ex.MAX_INPUT_SATURATION, sat
    with pytest.raises(ValueError):
        ex.predict_tflite(tfl, X)          # sin RR no se puede inferir
    with pytest.raises(ValueError):
        ex.convert_to_tflite(model, X[:10], int8=True)  # faltan datos representativos RR


def test_p4_saturation_detects_unrepresentative_calibration(trained_rr):
    """El fallo encontrado antes: calibrar solo con la clase 0 satura los RR
    prematuros. La comprobación de saturación debe detectarlo."""
    model, X, R, y, _ = trained_rr
    only_class0 = np.flatnonzero(y == 0)[:100]
    tfl = ex.convert_to_tflite(model, X[only_class0], int8=True, representative_rr=R[only_class0])
    assert ex.input_saturation(tfl, X, R)["rr"] >= ex.MAX_INPUT_SATURATION


def test_stratified_representative_covers_every_class():
    y = np.array([0] * 1000 + [1] * 30 + [2] * 400 + [3] * 5)
    idx = stratified_representative(y, 150, np.random.default_rng(0))
    assert np.bincount(y[idx]).tolist() == [150, 30, 150, 5]
    assert len(set(idx.tolist())) == len(idx)


def test_p5_model_config_has_rr_contract(trained_rr, tmp_path):
    from src.preprocessing.rr_features import RR_FEATURE_NAMES

    model, X, R, _, rep = trained_rr
    tfl = ex.convert_to_tflite(model, X[rep], int8=True, representative_rr=R[rep])
    qp = ex.quant_params(tfl)
    cfg = PreprocessConfig()
    text = ex.write_model_config(qp, {"classes": ["N", "S"]}, cfg, tmp_path).read_text(encoding="utf-8")
    assert f"#define ECG_MODEL_RR_INPUT_INDEX   {qp['rr_input_position']}" in text
    assert f"#define ECG_MODEL_BEAT_INPUT_INDEX {qp['beat_input_position']}" in text
    assert "#define ECG_MODEL_RR_FEATURES      8" in text
    assert ", ".join(RR_FEATURE_NAMES) in text
    assert f"#define ECG_MODEL_RR_LOCAL_BEATS   {cfg.rr_local_beats}" in text


def test_p5_integration_config_matches_python_rr_config():
    import re
    from pathlib import Path

    from src.preprocessing.rr_features import RR_FEATURE_NAMES

    h = (Path(__file__).resolve().parents[2] / "device-firmware" / "include" / "integration_config.h").read_text(
        encoding="utf-8")

    def define(name):
        m = re.search(rf"#define\s+{name}\s+(.+?)\s*(//.*)?$", h, re.M)
        assert m, name
        return m.group(1).strip().rstrip("f").strip('"')

    cfg = PreprocessConfig()
    assert int(define("ECG_RR_LOCAL_BEATS")) == cfg.rr_local_beats
    assert float(define("ECG_RR_LONG_TAU_S")) == cfg.rr_long_tau_s
    assert float(define("ECG_RR_MAX_S")) == cfg.rr_max_s
    assert float(define("ECG_RR_REF_S")) == cfg.rr_ref_s
    assert float(define("ECG_RR_CLIP")) == cfg.rr_clip
    assert int(define("ECG_RR_WARMUP_MIN_RR")) == cfg.rr_warmup_min_rr
    assert float(define("ECG_RR_WARMUP_S")) == cfg.rr_warmup_s
    assert int(define("ECG_RR_NUM_FEATURES")) == len(RR_FEATURE_NAMES)
    assert define("ECG_RR_FEATURE_ORDER") == ",".join(RR_FEATURE_NAMES)
