"""Experimento del modelo v2 (latido + RR): validación cruzada, modelo final y DS2.

Reglas fijadas ANTES de ejecutar (ver docs/model_documentation.md, sección 8):

* Validación cruzada de 5 grupos por paciente en DS1 (``cv.CV_FOLDS``). Para el
  grupo de prueba i, la parada temprana usa el grupo (i+1) mod 5 y se entrena
  con los otros 3: el grupo de prueba nunca interviene en el entrenamiento.
* Dos esquemas de pesos de clase: "balanced" (tope 50) y "sqrt". Se elige el
  que cumple más criterios (V completo; S Se >= 0,50); empate -> mayor F1 macro
  de N, S y V. Elegir con los mismos datos que se reportan es algo optimista.
* "Validación satisfactoria" (condición para el modelo final): criterio V y
  criterios técnicos cumplidos sobre las predicciones fuera de grupo.
* S genera alertas experimentales solo si Se >= 0,50 y VPP >= 0,50 en la CV.
* Falsa alerta: la clase predicha del latido que la dispara no es la real
  (lógica de ecg::AlertManager: confianza >= 0,8, >= 2 s entre alertas).
* Iteración 2 (``cv2``, la última iteración de desarrollo, fijada antes de
  ejecutarla a la vista de los resultados de ``cv``): un único esquema,
  "balanced" con el peso de F limitado a ``F_WEIGHT_CAP`` (F sigue siendo una
  clase independiente) y ``FIXED_EPOCHS`` épocas fijas (mediana de las mejores
  épocas de la iteración 1 con "balanced") sin parada temprana. Al no haber
  parada temprana no se necesita grupo de validación y se entrena con los 4
  grupos que no son de prueba. Criterios de aceptación sin cambios. Como se
  diseñó a la vista de la iteración 1, la CV ya es en parte datos de desarrollo.
* DS2 es una evaluación exploratoria: se ejecuta UNA sola vez (``ds2`` se
  niega a repetirse) y el modelo no se modifica después.

Uso::

    python -m src.training.rr_experiment cv
    python -m src.training.rr_experiment cv2
    python -m src.training.rr_experiment final
    python -m src.training.rr_experiment ds2
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from ..evaluation.metrics import classification_report, simulate_alerts
from ..preprocessing.pipeline import PreprocessConfig
from ..preprocessing.rr_features import N_RR_FEATURES
from . import dataset_loader as dl
from .cv import CV_FOLDS, verify_folds
from .train import REPRESENTATIVE_PER_CLASS, balanced_class_weights, build_model, stratified_representative

CLASSES = dl.DEFAULT_CLASSES
MODEL_DIR = dl.AI_MODEL_DIR / "models"
CACHE_DIR = dl.AI_MODEL_DIR / "data" / "processed"
SEED = 42
MAX_EPOCHS = 60
PATIENCE = 8
BATCH = 128
MAX_PER_CLASS = 15000
WEIGHT_SCHEMES = ("balanced", "sqrt")
V_CRITERIA = {"sensitivity": 0.80, "ppv": 0.85}
S_MIN_SENSITIVITY = 0.50
S_ALERT_RULE = {"sensitivity": 0.50, "ppv": 0.50}
F_WEIGHT_CAP = 5.0
FIXED_EPOCHS = 8
REPORTS = {"cv": "cv_v2_report.json", "cv2": "cv_v2_iter2_report.json"}
TECH = {"int8_agreement": 0.98, "max_saturation": 0.005, "max_bytes": 64 * 1024}
IDX = {c: i for i, c in enumerate(CLASSES)}


# --------------------------------------------------------------------------
# Datos (con caché en data/processed, fuera de git)
# --------------------------------------------------------------------------

def load_set(name: str, records: tuple[int, ...], cfg: PreprocessConfig, source: str) -> dl.BeatDataset:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"v2_{name}_{source}.npz"
    if path.exists():
        z = np.load(path)
        return dl.BeatDataset(z["X"], z["R"], z["y"], z["rec"], z["r_index"])
    t0 = time.time()
    ds = dl.build_dataset(records, cfg, rr_source=source)
    np.savez(path, X=ds.X, R=ds.R, y=ds.y, rec=ds.rec, r_index=ds.r_index)
    print(f"  dataset {name}/{source}: {len(ds)} latidos ({time.time() - t0:.0f} s)", flush=True)
    return ds


def in_records(ds: dl.BeatDataset, records) -> dl.BeatDataset:
    return ds.subset(np.isin(ds.rec, list(records)))


# --------------------------------------------------------------------------
# Entrenamiento y predicción
# --------------------------------------------------------------------------

def class_weights(y: np.ndarray, scheme: str, f_cap: float | None = None) -> dict[int, float]:
    w = balanced_class_weights(y, len(CLASSES))
    if scheme == "sqrt":
        w = {k: float(np.sqrt(v)) for k, v in w.items()}
    if f_cap is not None:
        w[IDX["F"]] = min(w[IDX["F"]], f_cap)
    return w


def cap_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:MAX_PER_CLASS] for c in np.unique(y)])
    return rng.permutation(idx)


def fit(train: dl.BeatDataset, scheme: str, val: dl.BeatDataset | None = None, epochs: int | None = None,
        seed: int = SEED, f_cap: float | None = None):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    sel = cap_indices(train.y, rng)
    tr = train.subset(sel)
    cfg = PreprocessConfig()
    model = build_model(cfg.input_length, len(CLASSES), rr_features=N_RR_FEATURES)
    kw = {}
    if val is not None:
        kw["validation_data"] = ([val.X[..., None], val.R], val.y)
        kw["callbacks"] = [tf.keras.callbacks.EarlyStopping("val_loss", patience=PATIENCE,
                                                            restore_best_weights=True)]
    hist = model.fit([tr.X[..., None], tr.R], tr.y, epochs=epochs or MAX_EPOCHS, batch_size=BATCH, verbose=0,
                     class_weight=class_weights(tr.y, scheme, f_cap), **kw)
    best = int(np.argmin(hist.history["val_loss"])) + 1 if val is not None else len(hist.history["loss"])
    rep = stratified_representative(tr.y, REPRESENTATIVE_PER_CLASS, rng)
    return model, best, tr, rep, hist.history


def to_int8(model, tr: dl.BeatDataset, rep: np.ndarray) -> bytes:
    from ..export import export_model as ex

    return ex.convert_to_tflite(model, tr.X[rep], int8=True, representative_rr=tr.R[rep])


def predict_float(model, ds: dl.BeatDataset) -> np.ndarray:
    return model.predict([ds.X[..., None], ds.R], batch_size=2048, verbose=0)


def predict_int8(tfl: bytes, ds: dl.BeatDataset) -> np.ndarray:
    from ..export import export_model as ex

    return ex.predict_tflite(tfl, ds.X, ds.R)


# --------------------------------------------------------------------------
# Informes
# --------------------------------------------------------------------------

def evaluate(ds: dl.BeatDataset, probs: np.ndarray, fs: float) -> dict:
    pred = probs.argmax(1)
    conf = probs.max(1)
    rep = classification_report(ds.y, pred, CLASSES)
    per_rec = {}
    for r in np.unique(ds.rec):
        m = ds.rec == r
        rr = classification_report(ds.y[m], pred[m], CLASSES)
        per_rec[int(r)] = {"n": int(m.sum()), "accuracy": rr["accuracy"],
                           "per_class": {c: {k: rr["per_class"][c][k] for k in ("support", "sensitivity", "ppv")}
                                         for c in CLASSES},
                           "confusion_matrix": rr["confusion_matrix"]}
    rep["per_record"] = per_rec
    rep["alerts_V"] = simulate_alerts(ds.rec, ds.r_index, ds.y, pred, conf, fs, (IDX["V"],))
    rep["alerts_V_S"] = simulate_alerts(ds.rec, ds.r_index, ds.y, pred, conf, fs, (IDX["V"], IDX["S"]))
    rep["alerts_S_only"] = simulate_alerts(ds.rec, ds.r_index, ds.y, pred, conf, fs, (IDX["S"],))
    return rep


def criteria(rep: dict) -> dict:
    v, s = rep["per_class"]["V"], rep["per_class"]["S"]
    return {
        "V_sensitivity_ok": v["sensitivity"] >= V_CRITERIA["sensitivity"],
        "V_ppv_ok": v["ppv"] >= V_CRITERIA["ppv"],
        "S_sensitivity_ok": s["sensitivity"] >= S_MIN_SENSITIVITY,
        "S_alerts_allowed": s["sensitivity"] >= S_ALERT_RULE["sensitivity"] and s["ppv"] >= S_ALERT_RULE["ppv"],
    }


def score(rep: dict) -> tuple:
    c = criteria(rep)
    met = int(c["V_sensitivity_ok"] and c["V_ppv_ok"]) + int(c["S_sensitivity_ok"])
    macro_nsv = float(np.mean([rep["per_class"][k]["f1"] for k in ("N", "S", "V")]))
    return met, macro_nsv


def summary_table(rep: dict) -> str:
    lines = [f"    {'':4}{'n':>7} {'Se':>6} {'VPP':>6} {'Esp':>6} {'F1':>6}"]
    for c in CLASSES:
        m = rep["per_class"][c]
        lines.append(f"    {c:4}{m['support']:>7} {m['sensitivity']:>6.3f} {m['ppv']:>6.3f} "
                     f"{m['specificity']:>6.3f} {m['f1']:>6.3f}")
    a = rep["alerts_V"]
    lines.append(f"    exactitud={rep['accuracy']:.3f}  F1macro={rep['macro_f1']:.3f}  "
                 f"falsas alertas V/h={a['false_alerts_per_hour']:.2f} ({a['false_alerts']}/{a['alerts']})")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Subcomandos
# --------------------------------------------------------------------------

def cmd_cv(args) -> None:
    run_cv(WEIGHT_SCHEMES, REPORTS["cv"], fixed_epochs=None, f_cap=None)


def cmd_cv2(args) -> None:
    run_cv(("balanced",), REPORTS["cv2"], fixed_epochs=FIXED_EPOCHS, f_cap=F_WEIGHT_CAP)


def run_cv(schemes, report_name: str, fixed_epochs: int | None, f_cap: float | None) -> None:
    from ..export import export_model as ex

    cfg = PreprocessConfig()
    verify_folds()  # lanza FoldError si la asignación no es válida
    ds1 = load_set("DS1", dl.DS1, cfg, "annotations")
    ds1_det = load_set("DS1", dl.DS1, cfg, "detector")
    results = {}
    for scheme in schemes:
        n = len(ds1)
        pf, p8 = np.zeros((n, 4), np.float32), np.zeros((n, 4), np.float32)
        pf_det = np.zeros((len(ds1_det), 4), np.float32)
        folds_info = []
        for i, test_recs in enumerate(CV_FOLDS):
            t0 = time.time()
            if fixed_epochs is None:
                val_recs = CV_FOLDS[(i + 1) % 5]
                train_recs = tuple(r for j, f in enumerate(CV_FOLDS) if j not in (i, (i + 1) % 5) for r in f)
                assert not set(test_recs) & set(train_recs) and not set(val_recs) & set(train_recs)
                model, best, tr, rep, _ = fit(in_records(ds1, train_recs), scheme, val=in_records(ds1, val_recs),
                                              f_cap=f_cap)
            else:
                val_recs = ()
                train_recs = tuple(r for j, f in enumerate(CV_FOLDS) if j != i for r in f)
                assert not set(test_recs) & set(train_recs)
                model, best, tr, rep, _ = fit(in_records(ds1, train_recs), scheme, epochs=fixed_epochs, f_cap=f_cap)
            tfl = to_int8(model, tr, rep)
            test_m = np.isin(ds1.rec, test_recs)
            test = ds1.subset(test_m)
            pf[test_m] = predict_float(model, test)
            p8[test_m] = predict_int8(tfl, test)
            det_m = np.isin(ds1_det.rec, test_recs)
            pf_det[det_m] = predict_float(model, ds1_det.subset(det_m))
            ok_ops, ops, _ = ex.check_tflm_compatibility(tfl)
            concat_ok = all(len(set(q)) == 1 for q in ex.concat_quantization(tfl))
            sat = ex.input_saturation(tfl, test.X, test.R)
            info = {"fold": i + 1, "test": list(test_recs), "val": list(val_recs), "train": list(train_recs),
                    "best_epoch": best, "train_counts": dl.class_counts(tr.y, CLASSES),
                    "class_weights": class_weights(tr.y, scheme, f_cap),
                    "tflite_bytes": len(tfl), "tflm_ops_ok": ok_ops, "concat_quant_equal": concat_ok,
                    "saturation_on_test": sat,
                    "int8_agreement": float(np.mean(pf[test_m].argmax(1) == p8[test_m].argmax(1))),
                    "seconds": round(time.time() - t0, 1)}
            folds_info.append(info)
            print(f"[{scheme}] grupo {i + 1}: mejor época {best}, F en entrenamiento={info['train_counts']['F']}, "
                  f"concordancia int8={info['int8_agreement']:.4f}, saturación={sat}, {info['seconds']} s", flush=True)
        res = {"folds": folds_info, "float": evaluate(ds1, pf, cfg.fs_hz), "int8": evaluate(ds1, p8, cfg.fs_hz),
               "float_detector_rr": evaluate(ds1_det, pf_det, cfg.fs_hz),
               "int8_float_agreement": float(np.mean(pf.argmax(1) == p8.argmax(1)))}
        res["criteria_float"] = criteria(res["float"])
        res["criteria_int8"] = criteria(res["int8"])
        res["technical"] = {
            "int8_agreement_ok": res["int8_float_agreement"] >= TECH["int8_agreement"],
            "saturation_ok": all(v < TECH["max_saturation"] for f in folds_info for v in f["saturation_on_test"].values()),
            "size_ok": all(f["tflite_bytes"] <= TECH["max_bytes"] for f in folds_info),
            "ops_ok": all(f["tflm_ops_ok"] and f["concat_quant_equal"] for f in folds_info),
        }
        results[scheme] = res
        print(f"== [{scheme}] validación cruzada (fuera de grupo), int8:\n{summary_table(res['int8'])}", flush=True)
    chosen = max(schemes, key=lambda s: score(results[s]["int8"]))
    ch = results[chosen]
    satisfactory = (ch["criteria_int8"]["V_sensitivity_ok"] and ch["criteria_int8"]["V_ppv_ok"]
                    and all(ch["technical"].values()))
    out = {"rules": __doc__, "results": results, "chosen_scheme": chosen,
           "chosen_by": "criterios cumplidos y F1 macro N/S/V sobre las predicciones int8 fuera de grupo",
           "validation_satisfactory": satisfactory,
           "S_alerts_allowed": ch["criteria_int8"]["S_alerts_allowed"],
           "f_weight_cap": f_cap, "fixed_epochs": fixed_epochs,
           "final_epochs": fixed_epochs or int(np.median([f["best_epoch"] for f in ch["folds"]]))}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / report_name).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"esquema elegido: {chosen}; validación satisfactoria: {satisfactory}; "
          f"alertas S permitidas: {out['S_alerts_allowed']}; épocas del modelo final: {out['final_epochs']}")


def cmd_final(args) -> None:
    from ..export import export_model as ex

    # Se usa la última iteración ejecutada (cv2 si existe).
    path = MODEL_DIR / REPORTS["cv2"]
    if not path.exists():
        path = MODEL_DIR / REPORTS["cv"]
    cvrep = json.loads(path.read_text(encoding="utf-8"))
    if not cvrep["validation_satisfactory"]:
        raise SystemExit("La validación cruzada no fue satisfactoria: no se entrena el modelo final.")
    cfg = PreprocessConfig()
    ds1 = load_set("DS1", dl.DS1, cfg, "annotations")
    # Conservar el modelo v1 (no se borra: se mueve a models/v1/).
    info_path = MODEL_DIR / "model_info.json"
    if info_path.exists() and "rr_features" not in json.loads(info_path.read_text(encoding="utf-8")):
        v1 = MODEL_DIR / "v1"
        v1.mkdir(exist_ok=True)
        for f in MODEL_DIR.iterdir():
            if f.is_file() and not f.name.startswith(("cv_", "v2_")):
                shutil.move(str(f), v1 / f.name)
    scheme, epochs = cvrep["chosen_scheme"], cvrep["final_epochs"]
    t0 = time.time()
    f_cap = cvrep.get("f_weight_cap")
    model, _, tr, rep, hist = fit(ds1, scheme, epochs=epochs, f_cap=f_cap)
    model.save(MODEL_DIR / "ecg_model.keras")
    cfg.save(MODEL_DIR / "preprocess_config.json")
    np.save(MODEL_DIR / "representative.npy", tr.X[rep].astype(np.float32))
    np.save(MODEL_DIR / "representative_rr.npy", tr.R[rep].astype(np.float32))
    rng = np.random.default_rng(SEED + 1)
    chk = rng.choice(np.setdiff1d(np.arange(len(ds1)), rep), size=min(3000, len(ds1)), replace=False)
    np.savez(MODEL_DIR / "saturation_check.npz", X=ds1.X[chk], R=ds1.R[chk])
    (MODEL_DIR / "train_history.json").write_text(
        json.dumps({k: [float(v) for v in vals] for k, vals in hist.items()}, indent=2), encoding="utf-8")
    info = {"version": 2, "classes": list(CLASSES), "input_length": cfg.input_length, "fs_hz": cfg.fs_hz,
            "rr_features": N_RR_FEATURES, "params": int(model.count_params()), "train_records": list(dl.DS1),
            "train_counts": dl.class_counts(tr.y, CLASSES), "weight_scheme": scheme, "epochs": epochs,
            "f_weight_cap": f_cap, "cv_report": path.name,
            "seed": SEED, "train_seconds": round(time.time() - t0, 1),
            "S_alerts_allowed_by_cv": cvrep["S_alerts_allowed"],
            "status": "EXPERIMENTAL - no validado en dispositivo ni clínicamente"}
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    report = ex.export(MODEL_DIR)
    print(json.dumps({k: report[k] for k in ("keras_params", "tflite_int8_bytes", "ops", "tflm_compatible_ops",
                                             "concatenation_quantization_equal", "input_saturation",
                                             "input_saturation_ok", "float_int8_agreement_on_representative")},
                     indent=2))
    if not (report["tflm_compatible_ops"] and report["input_saturation_ok"]):
        raise SystemExit("La exportación no supera las comprobaciones técnicas.")


def cmd_ds2(args) -> None:
    import tensorflow as tf

    from ..export import export_model as ex

    guard = MODEL_DIR / "ds2_v2_evaluated.json"
    if guard.exists():
        raise SystemExit(f"DS2 ya se evaluó ({guard}). Es una evaluación de una sola vez: no se repite.")
    info = json.loads((MODEL_DIR / "model_info.json").read_text(encoding="utf-8"))
    if info.get("version") != 2:
        raise SystemExit("El modelo actual no es el v2 final.")
    guard.write_text(json.dumps({"started": time.strftime("%Y-%m-%d %H:%M:%S"),
                                 "model_sha_bytes": (MODEL_DIR / "ecg_model_int8.tflite").stat().st_size}),
                     encoding="utf-8")
    cfg = PreprocessConfig.load(MODEL_DIR / "preprocess_config.json")
    model = tf.keras.models.load_model(MODEL_DIR / "ecg_model.keras")
    tfl = (MODEL_DIR / "ecg_model_int8.tflite").read_bytes()
    out = {"status": "EVALUACIÓN EXPLORATORIA: DS2 se consultó 3 veces con la v1; no es una prueba intacta. "
                     "No representa validación clínica.", "evaluated_once": True}
    for source in ("annotations", "detector"):
        ds2 = load_set("DS2", dl.DS2, cfg, source)
        pf, p8 = predict_float(model, ds2), predict_int8(tfl, ds2)
        no202 = ds2.subset(ds2.rec != 202)
        out[source] = {"float": evaluate(ds2, pf, cfg.fs_hz), "int8": evaluate(ds2, p8, cfg.fs_hz),
                       "int8_without_202": evaluate(no202, p8[ds2.rec != 202], cfg.fs_hz),
                       "int8_float_agreement": float(np.mean(pf.argmax(1) == p8.argmax(1))),
                       "saturation": ex.input_saturation(tfl, ds2.X, ds2.R)}
        out[source]["criteria_int8"] = criteria(out[source]["int8"])
        print(f"== DS2 ({source}), int8:\n{summary_table(out[source]['int8'])}", flush=True)
    (MODEL_DIR / "eval_ds2_v2.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("command", choices=("cv", "cv2", "final", "ds2"))
    args = p.parse_args()
    {"cv": cmd_cv, "cv2": cmd_cv2, "final": cmd_final, "ds2": cmd_ds2}[args.command](args)


if __name__ == "__main__":
    main()
