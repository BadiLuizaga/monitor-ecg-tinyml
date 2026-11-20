"""Validación cruzada agrupada por paciente dentro de DS1.

Los 5 grupos están fijados de antemano (aprobados antes de entrenar) y
reparten los registros con más latidos S y F. ``verify_folds`` comprueba
automáticamente, antes de cualquier entrenamiento:

1. que cada registro de DS1 aparece exactamente una vez;
2. que no hay registros de DS2 ni con marcapasos;
3. que los pares de registros del mismo paciente no quedan repartidos entre
   grupos (y documenta la fuga conocida 201 (DS1) / 202 (DS2));
4. la distribución de clases por grupo (en especial S y F).

Uso::

    python -m src.training.cv --verify
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ..preprocessing.pipeline import PreprocessConfig
from . import dataset_loader as dl

CV_FOLDS: tuple[tuple[int, ...], ...] = (
    (101, 106, 208, 220),
    (108, 109, 209, 215, 122),
    (112, 114, 201, 203),
    (115, 116, 207, 223, 124),
    (118, 119, 205, 230),
)


class FoldError(ValueError):
    """La asignación de grupos no es válida: no se debe entrenar."""


def check_fold_structure(folds=CV_FOLDS) -> list[str]:
    """Comprobaciones estructurales (sin cargar datos). Lanza FoldError si fallan."""
    problems = []
    flat = [r for f in folds for r in f]
    counts = Counter(flat)
    dup = sorted(r for r, c in counts.items() if c > 1)
    missing = sorted(set(dl.DS1) - set(flat))
    extra = sorted(set(flat) - set(dl.DS1))
    if dup:
        problems.append(f"registros repetidos: {dup}")
    if missing:
        problems.append(f"registros de DS1 que faltan: {missing}")
    if extra:
        problems.append(f"registros ajenos a DS1 (DS2/marcapasos): {extra}")
    fold_of = {r: i for i, f in enumerate(folds) for r in f}
    for a, b in dl.SAME_PATIENT:
        if a in fold_of and b in fold_of and fold_of[a] != fold_of[b]:
            problems.append(f"mismo paciente en grupos distintos: {a} y {b}")
    if problems:
        raise FoldError("; ".join(problems))
    notes = []
    for a, b in dl.SAME_PATIENT:
        sets = {r: ("DS1" if r in dl.DS1 else "DS2" if r in dl.DS2 else "otro") for r in (a, b)}
        if sets[a] != sets[b]:
            notes.append(f"fuga conocida de la división estándar: {a} ({sets[a]}) y {b} ({sets[b]}) "
                         f"son del mismo paciente; se reportará DS2 también sin {b if sets[b] == 'DS2' else a}")
    return notes


def fold_class_distribution(folds=CV_FOLDS, cfg: PreprocessConfig | None = None,
                            data_dir: Path = dl.DEFAULT_DATA_DIR) -> list[dict]:
    """Latidos por clase en cada grupo (los mismos latidos que se usarán para entrenar)."""
    cfg = cfg or PreprocessConfig()
    out = []
    for i, recs in enumerate(folds):
        ds = dl.build_dataset(recs, cfg, data_dir)
        per_rec = {int(r): dl.class_counts(ds.y[ds.rec == r], dl.DEFAULT_CLASSES) for r in recs}
        out.append({"fold": i + 1, "records": list(recs), "total": dl.class_counts(ds.y, dl.DEFAULT_CLASSES),
                    "per_record": per_rec})
    return out


def verify_folds(folds=CV_FOLDS, cfg: PreprocessConfig | None = None, data_dir: Path = dl.DEFAULT_DATA_DIR,
                 min_folds_with_class: dict[str, int] | None = None) -> dict:
    notes = check_fold_structure(folds)
    dist = fold_class_distribution(folds, cfg, data_dir)
    warnings = []
    for c, need in (min_folds_with_class or {"S": 5, "V": 5, "F": 2}).items():
        have = sum(1 for d in dist if d["total"][c] > 0)
        if have < need:
            warnings.append(f"la clase {c} solo aparece en {have} de {len(folds)} grupos")
    for c in ("S", "F"):
        tot = sum(d["total"][c] for d in dist)
        top = max(dist, key=lambda d: d["total"][c])
        if tot and top["total"][c] / tot > 0.5:
            warnings.append(f"el {100 * top['total'][c] / tot:.0f} % de los latidos {c} de DS1 está en el grupo "
                            f"{top['fold']}: la estimación de {c} dependerá mucho de ese grupo")
    return {"ok": True, "notes": notes, "warnings": warnings, "folds": dist}


def main() -> None:
    p = argparse.ArgumentParser(description="Verifica los grupos de validación cruzada")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--out", type=Path, default=dl.AI_MODEL_DIR / "models" / "cv_folds_report.json")
    args = p.parse_args()
    try:
        rep = verify_folds()
    except FoldError as exc:
        raise SystemExit(f"GRUPOS INVÁLIDOS, no entrenar: {exc}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"{'grupo':>5} {'registros':<28} {'N':>6} {'S':>5} {'V':>5} {'F':>5}")
    for d in rep["folds"]:
        t = d["total"]
        print(f"{d['fold']:>5} {','.join(map(str, d['records'])):<28} {t['N']:>6} {t['S']:>5} {t['V']:>5} {t['F']:>5}")
    for n in rep["notes"]:
        print("NOTA:", n)
    for w in rep["warnings"]:
        print("AVISO:", w)
    print(f"informe: {args.out}")


if __name__ == "__main__":
    main()
