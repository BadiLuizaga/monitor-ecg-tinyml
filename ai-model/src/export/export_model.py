"""Exportación del modelo para TensorFlow Lite Micro (ESP32).

1. Convierte el modelo Keras a TFLite con cuantización **int8 completa**
   (pesos, activaciones, entrada y salida) usando un conjunto representativo.
2. Comprueba que todos los operadores están en la lista de TFLM.
3. Genera los archivos para el firmware:
   * ``device-firmware/src/generated/model_data.cc`` (array del modelo, alineado a 16 bytes)
   * ``device-firmware/include/generated/model_data.h``
   * ``device-firmware/include/generated/model_config.h`` (longitud de entrada,
     parámetros de cuantización, clases, frecuencia)
   * ``device-firmware/include/generated/preprocessing_coeffs.h`` (biquads del filtro)

Uso::

    python -m src.export.export_model                 # modelo de ai-model/models
    python -m src.export.export_model --filters-only  # solo coeficientes del filtro

IMPORTANTE: que el modelo convierta y use operadores soportados NO significa
que quepa o sea suficientemente rápido. Hay que medir en el ESP32 real el
tamaño de arena (``interpreter.arena_used_bytes()``) y la latencia.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..preprocessing.pipeline import PreprocessConfig

AI_MODEL_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = AI_MODEL_DIR.parent
DEFAULT_MODEL_DIR = AI_MODEL_DIR / "models"
FIRMWARE_DIR = REPO_DIR / "device-firmware"

# Subconjunto de operadores registrables en TFLM (MicroMutableOpResolver).
# Fuente: tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h
TFLM_SUPPORTED_OPS = {
    "ADD", "AVERAGE_POOL_2D", "CONCATENATION", "CONV_2D", "DEPTHWISE_CONV_2D",
    "DEQUANTIZE", "EXPAND_DIMS", "FULLY_CONNECTED", "LOGISTIC", "MAX_POOL_2D",
    "MEAN", "MUL", "PAD", "QUANTIZE", "RELU", "RELU6", "RESHAPE", "SHAPE",
    "SOFTMAX", "SQUEEZE", "STRIDED_SLICE", "SUB", "TANH", "TRANSPOSE",
}

HEADER_BANNER = ("// ARCHIVO GENERADO por ai-model/src/export/export_model.py - no editar a mano.\n"
                 "// Generado: {date}\n")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _has_rr_input(model) -> bool:
    return len(model.inputs) == 2


def convert_to_tflite(model, representative: np.ndarray | None = None, int8: bool = True,
                      representative_rr: np.ndarray | None = None) -> bytes:
    """Convierte a TFLite. Para el modelo de dos entradas, los datos
    representativos se pasan POR NOMBRE ('beat', 'rr'): el modelo convertido
    no conserva el orden de las entradas de Keras."""
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if int8:
        if representative is None:
            raise ValueError("la cuantización int8 requiere datos representativos")
        two = _has_rr_input(model)
        if two and (representative_rr is None or len(representative_rr) != len(representative)):
            raise ValueError("el modelo con RR requiere datos representativos RR de la misma longitud")

        def rep_gen():
            for k, x in enumerate(representative[:, :, None].astype(np.float32)):
                if two:
                    yield {"beat": x[None, ...], "rr": representative_rr[k:k + 1].astype(np.float32)}
                else:
                    yield [x[None, ...]]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    return converter.convert()


def make_interpreter(tflite_model: bytes):
    import tensorflow as tf

    # Sin delegados (XNNPACK): así solo aparecen los operadores del modelo y
    # se usan kernels de referencia, más parecidos a los de TFLM.
    interp = tf.lite.Interpreter(
        model_content=tflite_model,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    interp.allocate_tensors()
    return interp


def tflite_ops(tflite_model: bytes) -> list[str]:
    interp = make_interpreter(tflite_model)
    return sorted({op["op_name"] for op in interp._get_ops_details()})


def check_tflm_compatibility(tflite_model: bytes) -> tuple[bool, list[str], list[str]]:
    ops = tflite_ops(tflite_model)
    unsupported = [op for op in ops if op not in TFLM_SUPPORTED_OPS]
    return not unsupported, ops, unsupported


def input_details_by_role(interp) -> dict:
    """{'beat': detalle, 'rr': detalle?}. Se identifican por NOMBRE, no por
    posición: el conversor puede reordenar las entradas (se ha observado 'rr'
    como entrada 0). El firmware usa los índices exportados en model_config.h."""
    roles = {}
    for pos, d in enumerate(interp.get_input_details()):
        name = d["name"]
        # Nombres típicos: "serving_default_beat:0", "serving_default_rr:0".
        role = "rr" if name.split(":")[0].endswith("rr") else "beat"
        roles[role] = {**d, "position": pos}
    if "beat" not in roles:
        raise ValueError(f"no se encontró la entrada 'beat': {[d['name'] for d in interp.get_input_details()]}")
    return roles


def quant_params(tflite_model: bytes) -> dict:
    interp = make_interpreter(tflite_model)
    roles = input_details_by_role(interp)
    i, o = roles["beat"], interp.get_output_details()[0]
    qp = {
        "input_shape": i["shape"].tolist(), "input_dtype": np.dtype(i["dtype"]).name,
        "input_scale": float(i["quantization"][0]), "input_zero_point": int(i["quantization"][1]),
        "beat_input_position": i["position"],
        "output_shape": o["shape"].tolist(), "output_dtype": np.dtype(o["dtype"]).name,
        "output_scale": float(o["quantization"][0]), "output_zero_point": int(o["quantization"][1]),
    }
    if "rr" in roles:
        r = roles["rr"]
        qp.update({"rr_shape": r["shape"].tolist(), "rr_dtype": np.dtype(r["dtype"]).name,
                   "rr_scale": float(r["quantization"][0]), "rr_zero_point": int(r["quantization"][1]),
                   "rr_input_position": r["position"]})
    return qp


MAX_INPUT_SATURATION = 0.005  # < 0,5 % de valores de entrada fuera del rango int8


def input_saturation(tflite_model: bytes, X: np.ndarray, R: np.ndarray | None = None) -> dict[str, float]:
    """Fracción de valores de cada entrada que quedan FUERA del rango int8
    (antes de saturar). Un valor exactamente en el límite no cuenta.

    Detecta calibraciones poco representativas: p. ej. si los datos de
    calibración no contienen latidos prematuros, sus RR se saturan.
    """
    interp = make_interpreter(tflite_model)
    res = {}
    for role, arr in (("beat", X), ("rr", R)):
        d = input_details_by_role(interp).get(role)
        if d is None or arr is None or d["dtype"] != np.int8:
            continue
        scale, zp = d["quantization"]
        v = np.asarray(arr, dtype=np.float32).astype(np.float64) / scale
        q = np.sign(v) * np.floor(np.abs(v) + 0.5) + zp
        res[role] = float(np.mean((q < -128) | (q > 127)))
    return res


def concat_quantization(tflite_model: bytes) -> list[list[tuple[float, int]]]:
    """(escala, punto cero) de entradas y salida de cada CONCATENATION.
    TFLM exige que coincidan en int8."""
    interp = make_interpreter(tflite_model)
    det = interp.get_tensor_details()
    return [[(float(det[t]["quantization"][0]), int(det[t]["quantization"][1]))
             for t in list(op["inputs"]) + list(op["outputs"])]
            for op in interp._get_ops_details() if op["op_name"] == "CONCATENATION"]


def predict_tflite(tflite_model: bytes, X: np.ndarray, R: np.ndarray | None = None,
                   return_int8: bool = False) -> np.ndarray:
    """Probabilidades (float) para ``X (n, L)`` y, si el modelo lo usa, ``R (n, 8)``.

    La cuantización de las entradas es la misma que hace el firmware
    (rr_features.quantize_int8). Con ``return_int8`` devuelve la salida int8
    cruda (vectores de referencia para TFLM).
    """
    from ..preprocessing.rr_features import quantize_int8

    interp = make_interpreter(tflite_model)
    roles = input_details_by_role(interp)
    out = interp.get_output_details()[0]
    out_scale, out_zp = out["quantization"]
    if "rr" in roles and R is None:
        raise ValueError("este modelo necesita las características RR")
    n_out = out["shape"][-1]
    res = np.empty((len(X), n_out), dtype=np.int8 if return_int8 else np.float32)
    for k in range(len(X)):
        for role, arr in (("beat", X), ("rr", R)):
            if role not in roles:
                continue
            d = roles[role]
            v = np.asarray(arr[k], dtype=np.float32).reshape(d["shape"])
            if d["dtype"] == np.int8:
                v = quantize_int8(v, d["quantization"][0], d["quantization"][1])
            interp.set_tensor(d["index"], v)
        interp.invoke()
        y = interp.get_tensor(out["index"])[0]
        if return_int8:
            res[k] = y
        else:
            res[k] = (y.astype(np.float32) - out_zp) * out_scale if out["dtype"] == np.int8 else y
    return res


def c_float(v: float) -> str:
    """Literal float válido en C++ (``1`` -> ``1.0f``, ``1e-05`` -> ``1e-05f``)."""
    s = f"{float(v):.12g}"
    if not any(c in s for c in ".eEn"):  # 'n' cubre inf/nan
        s += ".0"
    return s + "f"


def c_array(data: bytes, per_line: int = 12) -> str:
    rows = [", ".join(f"0x{b:02x}" for b in data[i:i + per_line]) for i in range(0, len(data), per_line)]
    return "  " + ",\n  ".join(rows)


def write_model_sources(tflite_model: bytes, firmware_dir: Path = FIRMWARE_DIR) -> tuple[Path, Path]:
    inc, src = firmware_dir / "include" / "generated", firmware_dir / "src" / "generated"
    inc.mkdir(parents=True, exist_ok=True)
    src.mkdir(parents=True, exist_ok=True)
    banner = HEADER_BANNER.format(date=_now())
    h = inc / "model_data.h"
    h.write_text(banner + (
        "#pragma once\n#include <cstddef>\n#include <cstdint>\n\n"
        "#define ECG_HAVE_MODEL_DATA 1\n\n"
        "extern const uint8_t g_ecg_model_data[];\n"
        "extern const size_t g_ecg_model_data_len;\n"), encoding="utf-8")
    cc = src / "model_data.cc"
    cc.write_text(banner + (
        '#include "generated/model_data.h"\n\n'
        "// Alineado a 16 bytes como exige TFLM.\n"
        f"alignas(16) const uint8_t g_ecg_model_data[] = {{\n{c_array(tflite_model)}\n}};\n"
        f"const size_t g_ecg_model_data_len = {len(tflite_model)};\n"), encoding="utf-8")
    return h, cc


def write_model_config(qp: dict, info: dict, cfg: PreprocessConfig, firmware_dir: Path = FIRMWARE_DIR) -> Path:
    path = firmware_dir / "include" / "generated" / "model_config.h"
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = info["classes"]
    nb, na = cfg.window
    text = HEADER_BANNER.format(date=_now()) + f"""#pragma once
#include <cstdint>

// Debe coincidir con ai-model/models/preprocess_config.json
#define ECG_MODEL_FS_HZ            {cfg.fs_hz:.1f}f
#define ECG_MODEL_INPUT_LENGTH     {cfg.input_length}
#define ECG_MODEL_WINDOW_BEFORE    {nb}
#define ECG_MODEL_WINDOW_AFTER     {na}
#define ECG_MODEL_NUM_CLASSES      {len(classes)}
#define ECG_MODEL_INPUT_SCALE      {c_float(qp['input_scale'])}
#define ECG_MODEL_INPUT_ZERO_POINT {qp['input_zero_point']}
#define ECG_MODEL_OUTPUT_SCALE     {c_float(qp['output_scale'])}
#define ECG_MODEL_OUTPUT_ZERO_POINT {qp['output_zero_point']}

// Posición de la entrada del latido en el modelo TFLite (no asumir 0).
#define ECG_MODEL_BEAT_INPUT_INDEX {qp.get('beat_input_position', 0)}

// Clases AAMI en el orden de salida del modelo.
static const char ECG_MODEL_CLASSES[ECG_MODEL_NUM_CLASSES] = {{{", ".join(f"'{c}'" for c in classes)}}};
"""
    if "rr_scale" in qp:
        from ..preprocessing.rr_features import RR_FEATURE_NAMES

        text += f"""
// Segunda entrada: características RR (ai-model/src/preprocessing/rr_features.py).
// Orden: {", ".join(RR_FEATURE_NAMES)}
#define ECG_MODEL_HAS_RR_INPUT     1
#define ECG_MODEL_RR_INPUT_INDEX   {qp['rr_input_position']}
#define ECG_MODEL_RR_FEATURES      {qp['rr_shape'][-1]}
#define ECG_MODEL_RR_SCALE         {c_float(qp['rr_scale'])}
#define ECG_MODEL_RR_ZERO_POINT    {qp['rr_zero_point']}
// Parámetros RR con los que se entrenó (el firmware comprueba que coinciden
// con integration_config.h mediante static_assert).
#define ECG_MODEL_RR_LOCAL_BEATS   {cfg.rr_local_beats}
#define ECG_MODEL_RR_LONG_TAU_S    {c_float(cfg.rr_long_tau_s)}
#define ECG_MODEL_RR_MAX_S         {c_float(cfg.rr_max_s)}
#define ECG_MODEL_RR_REF_S         {c_float(cfg.rr_ref_s)}
#define ECG_MODEL_RR_CLIP          {c_float(cfg.rr_clip)}
#define ECG_MODEL_RR_WARMUP_MIN_RR {cfg.rr_warmup_min_rr}
#define ECG_MODEL_RR_WARMUP_S      {c_float(cfg.rr_warmup_s)}
"""
    path.write_text(text, encoding="utf-8")
    return path


def write_filter_coeffs(cfg: PreprocessConfig, firmware_dir: Path = FIRMWARE_DIR) -> Path:
    """Biquads en forma SOS: b0 b1 b2 a1 a2 (a0 = 1)."""
    sos = cfg.sos()
    path = firmware_dir / "include" / "generated" / "preprocessing_coeffs.h"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ",\n".join(
        "  {" + ", ".join(c_float(v) for v in (s[0], s[1], s[2], s[4], s[5])) + "}" for s in sos)
    notch = f"{cfg.notch_hz}" if cfg.notch_hz else "ninguno"
    text = HEADER_BANNER.format(date=_now()) + f"""#pragma once

// Filtro: Butterworth paso banda {cfg.band_hz[0]}-{cfg.band_hz[1]} Hz, orden {cfg.filter_order}; notch: {notch}.
// Diseñado para fs = {cfg.fs_hz} Hz.
// PROVISIONAL: si la frecuencia real del sensor es distinta, regenerar con
//   python -m src.export.export_model --filters-only --fs <Hz>
#define ECG_FILTER_FS_HZ {cfg.fs_hz:.1f}f
#define ECG_FILTER_NUM_SECTIONS {len(sos)}

// Cada fila: b0, b1, b2, a1, a2
static const float ECG_FILTER_SOS[ECG_FILTER_NUM_SECTIONS][5] = {{
{rows}
}};
"""
    path.write_text(text, encoding="utf-8")
    return path


def export(model_dir: Path, firmware_dir: Path = FIRMWARE_DIR) -> dict:
    import tensorflow as tf

    cfg = PreprocessConfig.load(model_dir / "preprocess_config.json")
    info = json.loads((model_dir / "model_info.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(model_dir / "ecg_model.keras")
    rep = np.load(model_dir / "representative.npy")
    rr_path = model_dir / "representative_rr.npy"
    rep_rr = np.load(rr_path) if _has_rr_input(model) else None
    if _has_rr_input(model) and rep_rr is None:
        raise FileNotFoundError(rr_path)

    tfl_float = convert_to_tflite(model, int8=False)
    tfl_int8 = convert_to_tflite(model, rep, int8=True, representative_rr=rep_rr)
    (model_dir / "ecg_model_float.tflite").write_bytes(tfl_float)
    (model_dir / "ecg_model_int8.tflite").write_bytes(tfl_int8)

    ok, ops, unsupported = check_tflm_compatibility(tfl_int8)
    concat_q = concat_quantization(tfl_int8)
    concat_ok = all(len(set(q)) == 1 for q in concat_q)
    qp = quant_params(tfl_int8)
    # Concordancia float vs int8 sobre los datos representativos.
    keras_in = [rep[..., None], rep_rr] if rep_rr is not None else rep[..., None]
    agree = float(np.mean(model.predict(keras_in, verbose=0).argmax(1)
                          == predict_tflite(tfl_int8, rep, rep_rr).argmax(1)))
    # Saturación sobre un conjunto distinto de la calibración, si existe
    # (latidos de validación guardados por train.py); si no, la calibración.
    chk_x, chk_r = rep, rep_rr
    if (model_dir / "saturation_check.npz").exists():
        z = np.load(model_dir / "saturation_check.npz")
        chk_x, chk_r = z["X"], (z["R"] if "R" in z else None)
    saturation = input_saturation(tfl_int8, chk_x, chk_r)
    saturation_ok = all(v < MAX_INPUT_SATURATION for v in saturation.values())

    write_model_sources(tfl_int8, firmware_dir)
    write_model_config(qp, info, cfg, firmware_dir)
    write_filter_coeffs(cfg, firmware_dir)

    report = {
        "keras_params": int(model.count_params()),
        "tflite_float_bytes": len(tfl_float),
        "tflite_int8_bytes": len(tfl_int8),
        "ops": ops,
        "tflm_compatible_ops": ok and concat_ok,
        "unsupported_ops": unsupported,
        "concatenation_quantization_equal": concat_ok,
        "input_saturation": saturation,
        "input_saturation_limit": MAX_INPUT_SATURATION,
        "input_saturation_ok": saturation_ok,
        "quantization": qp,
        "float_int8_agreement_on_representative": agree,
        "arena_bytes": "PENDIENTE: medir en ESP32 con interpreter.arena_used_bytes()",
        "latency_ms": "PENDIENTE: medir en ESP32",
    }
    (model_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Exporta el modelo y los filtros al firmware")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--firmware-dir", type=Path, default=FIRMWARE_DIR)
    p.add_argument("--filters-only", action="store_true")
    p.add_argument("--fs", type=float, default=360.0, help="solo con --filters-only")
    p.add_argument("--notch", type=float, default=None, help="solo con --filters-only (50 o 60)")
    args = p.parse_args()
    if args.filters_only:
        path = write_filter_coeffs(PreprocessConfig(fs_hz=args.fs, notch_hz=args.notch), args.firmware_dir)
        print(f"coeficientes escritos en {path}")
        return
    report = export(args.model_dir, args.firmware_dir)
    print(json.dumps(report, indent=2))
    if not report["tflm_compatible_ops"]:
        raise SystemExit(f"operadores no soportados por TFLM: {report['unsupported_ops']}")
    if not report["input_saturation_ok"]:
        raise SystemExit(f"saturación int8 de las entradas >= {MAX_INPUT_SATURATION:.1%}: "
                         f"{report['input_saturation']} (revisar los datos de calibración)")


if __name__ == "__main__":
    main()
