"""Evaluación del modelo en el conjunto de prueba inter-paciente DS2.

Evalúa el modelo Keras (float) y, si existe, el modelo TFLite int8 que se
desplegará, para cuantificar la pérdida por cuantización. Opcionalmente mide
el detector de picos R frente a las anotaciones.

Uso::

    python -m src.evaluation.evaluate
    python -m src.evaluation.evaluate --detector   # añade la evaluación del detector (lento)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..preprocessing.pipeline import PreprocessConfig, filter_signal
from ..preprocessing.segmentation import detect_r_peaks, match_peaks
from ..training import dataset_loader as dl
from .metrics import classification_report, format_report

DEFAULT_MODEL_DIR = dl.AI_MODEL_DIR / "models"


def evaluate_detector(records: tuple[int, ...], cfg: PreprocessConfig, data_dir: Path,
                      tolerance_s: float = 0.15) -> dict:
    """Sensibilidad y VPP del detector de picos R (tolerancia ±150 ms, estándar EC57)."""
    tp = fp = fn = 0
    for r in records:
        rd = dl.load_record(r, data_dir)
        is_beat = np.isin([dl.AAMI_MAP.get(s, "") for s in rd.ann_symbols], list("NSVFQ"))
        ref = np.round(rd.ann_samples[is_beat] * cfg.fs_hz / rd.fs).astype(np.int64)
        det = detect_r_peaks(filter_signal(rd.signal_mv, rd.fs, cfg), cfg.fs_hz)
        skip = int(2.0 * cfg.fs_hz)  # periodo de aprendizaje del detector
        a, b, c = match_peaks(det[det > skip], ref[ref > skip], int(tolerance_s * cfg.fs_hz))
        tp, fp, fn = tp + a, fp + b, fn + c
    return {"tp": tp, "fp": fp, "fn": fn,
            "sensitivity": tp / max(tp + fn, 1), "ppv": tp / max(tp + fp, 1)}


def main() -> None:
    import tensorflow as tf

    from ..export.export_model import predict_tflite

    p = argparse.ArgumentParser(description="Evalúa el modelo en DS2")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--data-dir", type=Path, default=dl.DEFAULT_DATA_DIR)
    p.add_argument("--detector", action="store_true")
    p.add_argument("--records", default="", help="subconjunto de DS2 separado por comas")
    args = p.parse_args()

    cfg = PreprocessConfig.load(args.model_dir / "preprocess_config.json")
    info = json.loads((args.model_dir / "model_info.json").read_text(encoding="utf-8"))
    classes = tuple(info["classes"])
    records = tuple(int(r) for r in args.records.split(",")) if args.records else dl.DS2
    X, y, _ = dl.build_beat_dataset(records, cfg, args.data_dir, classes)

    results: dict = {"records": list(records), "status": "EXPERIMENTAL - no es una validación clínica"}
    model = tf.keras.models.load_model(args.model_dir / "ecg_model.keras")
    pred = model.predict(X[..., None], batch_size=1024, verbose=0).argmax(1)
    results["keras_float"] = classification_report(y, pred, classes)
    print("== Keras (float) ==\n" + format_report(results["keras_float"]))

    tfl_path = args.model_dir / "ecg_model_int8.tflite"
    if tfl_path.exists():
        pred8 = predict_tflite(tfl_path.read_bytes(), X).argmax(1)
        results["tflite_int8"] = classification_report(y, pred8, classes)
        results["float_int8_agreement"] = float(np.mean(pred == pred8))
        print("\n== TFLite int8 ==\n" + format_report(results["tflite_int8"]))
        print(f"concordancia float/int8: {results['float_int8_agreement']:.4f}")

    if args.detector:
        results["r_detector"] = evaluate_detector(records, cfg, args.data_dir)
        print("\n== Detector de picos R ==\n", results["r_detector"])

    (args.model_dir / "eval_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
