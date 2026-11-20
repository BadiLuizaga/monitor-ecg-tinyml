"""Diagnóstico de T1: compara capa a capa TFLM (diag_tflm.cpp) con TFLite (Python).

Para cada latido de vectors/ y cada nodo del modelo v1 compara la salida int8
de TFLM con la del intérprete TFLite de Python (kernels BUILTIN_REF y por
defecto), con ``experimental_preserve_all_tensors=True``. Las dos partes usan
la MISMA entrada int8 (la de Python), así que una diferencia solo puede venir
de los kernels o de la configuración del intérprete.

Escribe vectors/layer_diag.json y muestra:
* por nodo: latidos con diferencias, |diff| máximo, escala y punto cero;
* el primer nodo que difiere en cada latido;
* si en ese nodo las entradas son idénticas (=> diferencia del kernel).

Uso: python device-firmware/test/tflm/diag_layers.py
"""

from __future__ import annotations

import json
import struct
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VEC = HERE / "vectors"


def load_vectors():
    raw = (VEC / "beats.bin").read_bytes()
    n, L = struct.unpack("<ii", raw[:8])
    exp = np.frombuffer((VEC / "expected.bin").read_bytes(), np.int8).reshape(n, L + 8)
    return n, L, exp[:, :L]


def load_tflm(n_expected: int):
    raw = (VEC / "tflm_layers.bin").read_bytes()
    n, k = struct.unpack("<ii", raw[:8])
    assert n == n_expected
    hdr = np.frombuffer(raw[8:8 + 8 * k], "<i4").reshape(k, 2)
    per_beat = int(hdr[:, 1].sum())
    data = np.frombuffer(raw[8 + 8 * k:], np.uint8).reshape(n, per_beat)
    return hdr, data


def python_layers(tfl: bytes, q: np.ndarray, resolver):
    import tensorflow as tf

    it = tf.lite.Interpreter(model_content=tfl, experimental_op_resolver_type=resolver,
                             experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    ops = it._get_ops_details()
    inp = it.get_input_details()[0]
    outs = []
    for x in q:
        it.set_tensor(inp["index"], x.reshape(inp["shape"]))
        it.invoke()
        outs.append([it.get_tensor(op["outputs"][0]).copy() for op in ops])
    return it, ops, outs


def main() -> None:
    import tensorflow as tf

    tfl = (ROOT / "ai-model" / "models" / "ecg_model_int8.tflite").read_bytes()
    n, L, q = load_vectors()
    hdr, tdata = load_tflm(n)
    res = {"REF": tf.lite.experimental.OpResolverType.BUILTIN_REF,
           "DEFAULT": tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES}
    report = {"n": n, "nodes": [], "first_diff": {}}
    it_ref, ops, py_ref = python_layers(tfl, q, res["REF"])
    _, _, py_def = python_layers(tfl, q, res["DEFAULT"])
    tensors = {t["index"]: t for t in it_ref.get_tensor_details()}
    assert [op["outputs"][0] for op in ops] == hdr[:, 0].tolist(), "orden de nodos distinto"

    # Salidas de TFLM por nodo con el tipo de su tensor.
    offs = np.concatenate([[0], np.cumsum(hdr[:, 1])])
    tflm = []
    for b in range(n):
        row = []
        for j, op in enumerate(ops):
            t = tensors[op["outputs"][0]]
            row.append(tdata[b, offs[j]:offs[j + 1]].view(t["dtype"]).reshape(t["shape"]))
        tflm.append(row)

    for name, py in (("REF", py_ref), ("DEFAULT", py_def)):
        first = []
        node_stats = []
        for j, op in enumerate(ops):
            d = np.array([np.abs(tflm[b][j].astype(np.int64) - py[b][j].astype(np.int64)).max() for b in range(n)])
            hist = Counter(np.concatenate([np.abs(tflm[b][j].astype(np.int64) - py[b][j].astype(np.int64)).ravel()
                                           for b in range(n)]).tolist())
            t = tensors[op["outputs"][0]]
            node_stats.append({"node": j, "op": op["op_name"], "tensor": t["name"], "shape": t["shape"].tolist(),
                               "dtype": t["dtype"].__name__, "scale": float(t["quantization"][0]),
                               "zero_point": int(t["quantization"][1]),
                               "beats_with_diff": int((d > 0).sum()), "max_abs_diff": int(d.max()),
                               "elements_by_abs_diff": {str(k): int(v) for k, v in sorted(hist.items())}})
        for b in range(n):
            fd = next((j for j in range(len(ops))
                       if not np.array_equal(tflm[b][j], py[b][j])), None)
            first.append(fd)
        report["nodes" if name == "REF" else "nodes_default"] = node_stats
        report["first_diff"][name] = {
            "beats_without_any_diff": int(sum(f is None for f in first)),
            "first_diff_node_counts": {f"{j}:{ops[j]['op_name']}": c
                                       for j, c in sorted(Counter(f for f in first if f is not None).items())},
            "per_beat": [f for f in first],
        }
        print(f"== TFLM frente a TFLite {name}: latidos sin ninguna diferencia "
              f"{report['first_diff'][name]['beats_without_any_diff']}/{n}")
        print("   primer nodo que difiere:", report["first_diff"][name]["first_diff_node_counts"])
        print(f"   {'nodo':>4} {'op':<16} {'forma':<16} {'dtype':<6} {'escala':>11} {'zp':>5} {'lat.':>5} {'máx':>4}")
        for s in node_stats:
            print(f"   {s['node']:>4} {s['op']:<16} {str(s['shape']):<16} {s['dtype']:<6} {s['scale']:>11.6g} "
                  f"{s['zero_point']:>5} {s['beats_with_diff']:>5} {s['max_abs_diff']:>4}")
    (VEC / "layer_diag.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    np.savez_compressed(VEC / "layer_values.npz",
                        **{f"py_ref_{j}": np.stack([py_ref[b][j] for b in range(n)]) for j in range(len(ops))},
                        **{f"tflm_{j}": np.stack([tflm[b][j] for b in range(n)]) for j in range(len(ops))})


if __name__ == "__main__":
    sys.exit(main())
