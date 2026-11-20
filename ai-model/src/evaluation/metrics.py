"""Métricas de clasificación de latidos (solo NumPy).

Con clases muy desbalanceadas (≈90 % N) la exactitud global no es
informativa: se reportan sensibilidad, VPP (precisión), especificidad y F1
por clase, que son las métricas habituales en la literatura (AAMI EC57).
"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Filas = clase real, columnas = clase predicha."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)), 1)
    return cm


def _div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def per_class_metrics(cm: np.ndarray, labels: tuple[str, ...]) -> dict[str, dict[str, float]]:
    total = cm.sum()
    out = {}
    for i, name in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        sens = _div(tp, tp + fn)
        ppv = _div(tp, tp + fp)
        out[name] = {
            "support": int(cm[i].sum()),
            "sensitivity": sens,
            "ppv": ppv,
            "specificity": _div(tn, tn + fp),
            "f1": _div(2 * sens * ppv, sens + ppv),
        }
    return out


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, labels: tuple[str, ...]) -> dict:
    cm = confusion_matrix(y_true, y_pred, len(labels))
    per_class = per_class_metrics(cm, labels)
    present = [m["f1"] for m in per_class.values() if m["support"] > 0]
    return {
        "labels": list(labels),
        "confusion_matrix": cm.tolist(),
        "accuracy": _div(np.trace(cm), cm.sum()),
        "macro_f1": float(np.mean(present)) if present else 0.0,
        "per_class": per_class,
    }


def simulate_alerts(rec: np.ndarray, r_index: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                    conf: np.ndarray, fs: float, eligible: tuple[int, ...], min_conf: float = 0.80,
                    window: int = 8, min_abnormal: int = 1, cooldown_s: float = 2.0,
                    record_seconds: float = 1805.0) -> dict:
    """Misma lógica que ``ecg::AlertManager`` (device-firmware/src/alert_manager.cpp),
    aplicada latido a latido por registro.

    Falsa alerta: la clase predicha del latido que la dispara no es su clase
    real. Las horas se cuentan por registro completo (MIT-BIH: ~30 min).
    """
    per_record = {}
    for r in np.unique(rec):
        m = np.flatnonzero(rec == r)
        m = m[np.argsort(r_index[m])]
        win = [False] * window
        pos, last = 0, -10 ** 12
        alerts = false = 0
        for i in m:
            abnormal = int(y_pred[i]) in eligible and conf[i] >= min_conf
            win[pos] = abnormal
            pos = (pos + 1) % window
            if abnormal and sum(win) >= min_abnormal and r_index[i] - last >= int(cooldown_s * fs):
                last = int(r_index[i])
                alerts += 1
                false += int(y_pred[i] != y_true[i])
        per_record[int(r)] = {"alerts": alerts, "false_alerts": false,
                              "false_alerts_per_hour": false / (record_seconds / 3600.0)}
    hours = len(per_record) * record_seconds / 3600.0
    tot_a = sum(v["alerts"] for v in per_record.values())
    tot_f = sum(v["false_alerts"] for v in per_record.values())
    return {"alerts": tot_a, "false_alerts": tot_f, "hours": hours,
            "false_alerts_per_hour": tot_f / hours if hours else 0.0, "per_record": per_record}


def format_report(report: dict) -> str:
    lines = [f"{'clase':>6} {'n':>7} {'Se':>7} {'VPP':>7} {'Esp':>7} {'F1':>7}"]
    for name, m in report["per_class"].items():
        lines.append(f"{name:>6} {m['support']:>7} {m['sensitivity']:>7.3f} {m['ppv']:>7.3f} "
                     f"{m['specificity']:>7.3f} {m['f1']:>7.3f}")
    lines.append(f"exactitud={report['accuracy']:.3f}  F1 macro={report['macro_f1']:.3f}")
    lines.append("matriz de confusión (fila=real, columna=predicha):")
    lines += ["  " + " ".join(f"{v:>6}" for v in row) for row in report["confusion_matrix"]]
    return "\n".join(lines)
