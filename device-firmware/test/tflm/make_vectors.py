"""Vectores de referencia para la prueba T1 (TFLM en PC) con el modelo v1.

Genera, a partir del modelo int8 exportado (ai-model/models/ecg_model_int8.tflite):

* ``vectors/beats.bin``: latidos float32 (ya filtrados y en z-score) de DS1
  (caché ``ai-model/data/processed/v2_DS1_annotations.npz``) más casos límite.
  No se usa DS2.
* ``vectors/expected.bin``: por latido, la entrada int8 cuantizada en Python
  (misma regla que el firmware) y la salida int8 del intérprete TFLite de
  Python con kernels de referencia (BUILTIN_REF) y con los kernels por
  defecto sin delegados (BUILTIN_WITHOUT_DEFAULT_DELEGATES).
* ``vectors/meta.json``: forma, cuantización y operadores esperados.

Comprueba además que ``device-firmware/src/generated/model_data.cc`` contiene
exactamente los bytes del .tflite (es el modelo que se compila en la prueba).

Formato binario (little-endian):
  beats.bin    = int32 n, int32 L, float32[n][L]
  expected.bin = por latido: int8[L] entrada, int8[C] salida REF, int8[C] salida por defecto

Uso: python device-firmware/test/tflm/make_vectors.py
"""

from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AI = ROOT / "ai-model"
sys.path.insert(0, str(AI))

from src.export import export_model as ex  # noqa: E402
from src.preprocessing.rr_features import quantize_int8  # noqa: E402

PER_CLASS = 250
SEED = 1234


def model_cc_bytes(path: Path) -> bytes:
    text = path.read_text(encoding="utf-8")
    body = text[text.index("g_ecg_model_data[] = {"):text.index("};")]
    return bytes(int(h, 16) for h in re.findall(r"0x([0-9a-fA-F]{2})", body))


def run_interp(tfl: bytes, q_inputs: np.ndarray, resolver) -> np.ndarray:
    import tensorflow as tf

    it = tf.lite.Interpreter(model_content=tfl, experimental_op_resolver_type=resolver)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    res = np.zeros((len(q_inputs), out["shape"][-1]), np.int8)
    for i, q in enumerate(q_inputs):
        it.set_tensor(inp["index"], q.reshape(inp["shape"]))
        it.invoke()
        res[i] = it.get_tensor(out["index"]).reshape(-1)
    return res


def main() -> None:
    import tensorflow as tf

    tfl_path = AI / "models" / "ecg_model_int8.tflite"
    tfl = tfl_path.read_bytes()
    info = json.loads((AI / "models" / "model_info.json").read_text(encoding="utf-8"))
    if "rr_features" in info:
        raise SystemExit("T1 es solo para el modelo v1 (sin entrada RR); model_info.json indica otro modelo.")
    cc = model_cc_bytes(ROOT / "device-firmware" / "src" / "generated" / "model_data.cc")
    if cc != tfl:
        raise SystemExit("model_data.cc no coincide con ecg_model_int8.tflite")

    it = ex.make_interpreter(tfl)
    ins, outs = it.get_input_details(), it.get_output_details()
    if len(ins) != 1:
        raise SystemExit(f"Se esperaba 1 entrada (modelo v1) y hay {len(ins)}")
    scale, zp = ins[0]["quantization"]
    L = int(np.prod(ins[0]["shape"]))
    C = int(outs[0]["shape"][-1])

    z = np.load(AI / "data" / "processed" / "v2_DS1_annotations.npz")
    X, y, rec = z["X"].astype(np.float32), z["y"], z["rec"]
    assert X.shape[1] == L
    rng = np.random.default_rng(SEED)
    idx = np.concatenate([rng.permutation(np.flatnonzero(y == c))[:PER_CLASS] for c in range(C)])
    beats = X[idx]
    labels = y[idx].astype(int).tolist()
    # Casos límite: ceros, saturación positiva/negativa, alternancia extrema,
    # ruido gaussiano y valores exactamente en el punto medio de redondeo.
    t = np.arange(L)
    edge = np.stack([
        np.zeros(L), np.full(L, 50.0), np.full(L, -50.0), np.where(t % 2 == 0, 20.0, -20.0),
        rng.standard_normal(L), rng.standard_normal(L) * 3.0,
        ((t % 7) - 3 + 0.5) * scale,  # mitades exactas de escala
    ]).astype(np.float32)
    beats = np.concatenate([beats, edge])
    labels += [-1] * len(edge)

    q = quantize_int8(beats, scale, zp)
    ref = run_interp(tfl, q, tf.lite.experimental.OpResolverType.BUILTIN_REF)
    opt = run_interp(tfl, q, tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)

    out_dir = HERE / "vectors"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "beats.bin", "wb") as f:
        f.write(struct.pack("<ii", len(beats), L))
        f.write(beats.astype("<f4").tobytes())
    with open(out_dir / "expected.bin", "wb") as f:
        for i in range(len(beats)):
            f.write(q[i].tobytes() + ref[i].tobytes() + opt[i].tobytes())
    meta = {
        "model": str(tfl_path.relative_to(ROOT)), "model_bytes": len(tfl),
        "n": len(beats), "input_length": L, "num_classes": C,
        "input_shape": ins[0]["shape"].tolist(), "input_dtype": str(ins[0]["dtype"].__name__),
        "input_scale": float(scale), "input_zero_point": int(zp),
        "output_shape": outs[0]["shape"].tolist(), "output_dtype": str(outs[0]["dtype"].__name__),
        "output_scale": float(outs[0]["quantization"][0]), "output_zero_point": int(outs[0]["quantization"][1]),
        "ops": ex.tflite_ops(tfl), "labels": labels, "source": "DS1 (caché v2_DS1_annotations) + casos límite",
        "records": sorted(set(int(r) for r in rec[idx])),
        "python_ref_vs_default_exact": float(np.mean(np.all(ref == opt, axis=1))),
        "tensorflow": tf.__version__,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"vectores: {len(beats)} latidos ({len(edge)} casos límite), L={L}, C={C}, "
          f"ops={meta['ops']}, REF==por defecto en {meta['python_ref_vs_default_exact']:.4f}")


if __name__ == "__main__":
    main()
