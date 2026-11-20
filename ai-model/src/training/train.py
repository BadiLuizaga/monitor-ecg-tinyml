"""Entrenamiento del clasificador de latidos (CNN 1D pequeña para TinyML).

Arquitectura pensada para TensorFlow Lite Micro: solo usa operadores
soportados por TFLM (CONV_2D, MAX_POOL_2D, MEAN, FULLY_CONNECTED, SOFTMAX,
RESHAPE/EXPAND_DIMS), con pocos miles de parámetros.

Uso::

    python -m src.training.train --download --epochs 30
    python -m src.training.train --quick        # prueba rápida de la ruta

Genera en ``ai-model/models/``:
``ecg_model.keras``, ``preprocess_config.json``, ``model_info.json``,
``train_history.json`` y ``representative.npy`` (para cuantizar: hasta 150
latidos de cada clase, ver ``stratified_representative``).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..preprocessing.pipeline import PreprocessConfig
from . import dataset_loader as dl

DEFAULT_MODEL_DIR = dl.AI_MODEL_DIR / "models"


def build_model(input_length: int, n_classes: int, filters: tuple[int, ...] = (16, 32, 32),
                kernels: tuple[int, ...] = (7, 5, 3), dense: int = 32, rr_features: int = 0,
                rr_dense: int = 16, dropout: float = 0.2):
    """CNN 1D sobre el latido; con ``rr_features > 0`` añade la rama RR.

    Rama RR: Dense(16, ReLU). Fusión: Concatenate -> Dense(32, ReLU) ->
    Dropout (solo entrenamiento) -> Softmax. Todas las operaciones resultantes
    están soportadas por TFLM (CONCATENATION incluida); BatchNorm se fusiona con
    la convolución al convertir a TFLite.
    """
    import tensorflow as tf

    layers = tf.keras.layers
    inp = layers.Input(shape=(input_length, 1), name="beat")
    x = inp
    for f, k in zip(filters, kernels):
        x = layers.Conv1D(f, k, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    inputs = [inp]
    if rr_features:
        rr_in = layers.Input(shape=(rr_features,), name="rr")
        inputs.append(rr_in)
        x = layers.Concatenate()([x, layers.Dense(rr_dense, activation="relu")(rr_in)])
    x = layers.Dense(dense, activation="relu")(x)
    if rr_features and dropout:
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation="softmax", name="probs")(x)
    model = tf.keras.Model(inputs if rr_features else inp, out,
                           name="ecg_beat_rr_cnn" if rr_features else "ecg_beat_cnn")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def balanced_class_weights(y: np.ndarray, n_classes: int, max_weight: float = 50.0) -> dict[int, float]:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = np.where(counts > 0, counts.sum() / (n_classes * np.maximum(counts, 1)), 0.0)
    return {i: float(min(v, max_weight)) for i, v in enumerate(w)}


REPRESENTATIVE_PER_CLASS = 150


def stratified_representative(y: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    """Índices para calibrar la cuantización int8: hasta ``per_class`` latidos de
    CADA clase, sin repetición. Con una muestra aleatoria simple los latidos
    raros (S, V, F), que son los que tienen RR extremos, pueden quedar fuera y
    saturarse en el modelo int8."""
    idx = [rng.permutation(np.flatnonzero(y == c))[:per_class] for c in np.unique(y)]
    return rng.permutation(np.concatenate(idx))


def cap_per_class(X: np.ndarray, y: np.ndarray, max_per_class: int | None, rng: np.random.Generator):
    if not max_per_class:
        return X, y
    idx = np.concatenate([
        rng.permutation(np.flatnonzero(y == c))[:max_per_class] for c in np.unique(y)])
    idx = rng.permutation(idx)
    return X[idx], y[idx]


def split_records(records: tuple[int, ...], n_val: int, rng: np.random.Generator):
    """Validación por paciente (registros completos)."""
    recs = list(rng.permutation(records))
    return tuple(sorted(recs[n_val:])), tuple(sorted(recs[:n_val]))


def split_tail(rec_ids: np.ndarray, fraction: float) -> np.ndarray:
    """Máscara de validación: el último ``fraction`` (en el tiempo) de cada registro.

    Mantiene todas las clases en entrenamiento aunque se concentren en pocos
    pacientes (p. ej. casi todos los latidos F de DS1 están en el registro 208).
    Es una validación intra-paciente, solo para la parada temprana: la prueba
    honesta es DS2 (pacientes nunca vistos).
    """
    val = np.zeros(len(rec_ids), dtype=bool)
    for r in np.unique(rec_ids):
        idx = np.flatnonzero(rec_ids == r)  # los latidos están en orden temporal
        val[idx[int(len(idx) * (1 - fraction)):]] = True
    return val


def train(args: argparse.Namespace) -> Path:
    import tensorflow as tf

    rng = np.random.default_rng(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    cfg = PreprocessConfig(fs_hz=args.fs)
    classes = tuple(args.classes.split(","))
    if args.download:
        dl.download_mitbih(args.data_dir)

    records = dl.DS1[:8] if args.quick else dl.DS1  # --quick: solo verifica la ruta
    if args.val_mode == "records":
        train_recs, val_recs = split_records(records, args.val_records, rng)
        X_tr, y_tr, _ = dl.build_beat_dataset(train_recs, cfg, args.data_dir, classes)
        X_va, y_va, _ = dl.build_beat_dataset(val_recs, cfg, args.data_dir, classes)
    else:
        train_recs = val_recs = records
        X, y, rec = dl.build_beat_dataset(records, cfg, args.data_dir, classes)
        is_val = split_tail(rec, args.val_fraction)
        X_tr, y_tr, X_va, y_va = X[~is_val], y[~is_val], X[is_val], y[is_val]
    print("entrenamiento:", dl.class_counts(y_tr, classes))
    print("validación   :", dl.class_counts(y_va, classes))
    X_tr, y_tr = cap_per_class(X_tr, y_tr, args.max_per_class, rng)

    model = build_model(cfg.input_length, len(classes))
    model.summary(print_fn=lambda s: print("  " + s))
    t0 = time.time()
    hist = model.fit(
        X_tr[..., None], y_tr, validation_data=(X_va[..., None], y_va),
        epochs=args.epochs, batch_size=args.batch_size, verbose=2,
        class_weight=balanced_class_weights(y_tr, len(classes)),
        callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=8, restore_best_weights=True)],
    )

    out = Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "ecg_model.keras")
    cfg.save(out / "preprocess_config.json")
    rep_idx = stratified_representative(y_tr, REPRESENTATIVE_PER_CLASS, rng)
    np.save(out / "representative.npy", X_tr[rep_idx].astype(np.float32))
    (out / "train_history.json").write_text(
        json.dumps({k: [float(v) for v in vals] for k, vals in hist.history.items()}, indent=2), encoding="utf-8")
    info = {
        "classes": list(classes),
        "input_length": cfg.input_length,
        "fs_hz": cfg.fs_hz,
        "params": int(model.count_params()),
        "train_records": list(map(int, train_recs)),
        "val_records": list(map(int, val_recs)),
        "val_mode": args.val_mode,
        "train_class_counts": dl.class_counts(y_tr, classes),
        "test_records": list(dl.DS2),
        "epochs_run": len(hist.history["loss"]),
        "train_seconds": round(time.time() - t0, 1),
        "quick_run": bool(args.quick),
        "seed": args.seed,
        "status": "EXPERIMENTAL - no validado en dispositivo ni clínicamente",
    }
    (out / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"modelo guardado en {out}")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena el clasificador de latidos")
    p.add_argument("--data-dir", type=Path, default=dl.DEFAULT_DATA_DIR)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--download", action="store_true", help="descarga MIT-BIH si falta")
    p.add_argument("--fs", type=float, default=360.0, help="frecuencia de trabajo del modelo (Hz)")
    p.add_argument("--classes", default=",".join(dl.DEFAULT_CLASSES))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-per-class", type=int, default=15000, help="limita la clase N (0 = sin límite)")
    p.add_argument("--val-mode", choices=("records", "tail"), default="records",
                   help="records: registros completos (por defecto; mejor en DS2); tail: último 15%% de cada registro")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--val-records", type=int, default=4, help="solo con --val-mode records")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="pocos registros: solo verifica la ruta")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
