"""Carga de MIT-BIH Arrhythmia Database y construcción del conjunto de latidos.

* Descarga desde PhysioNet con ``wfdb`` dentro de ``ai-model/data/raw/mitdb``.
* Mapeo de anotaciones MIT-BIH a las clases AAMI EC57 (N, S, V, F, Q).
* División **inter-paciente** DS1 (entrenamiento) / DS2 (prueba) de
  de Chazal et al. (2004): ningún paciente aparece en ambos conjuntos, lo que
  evita métricas infladas por memorizar la morfología de un paciente.
  Se excluyen los registros con marcapasos (102, 104, 107, 217).

Uso::

    python -m src.training.dataset_loader --download
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..preprocessing import segmentation
from ..preprocessing.filters import resample_indices
from ..preprocessing.normalization import normalize
from ..preprocessing.pipeline import PreprocessConfig, filter_signal, simulate_device
from ..preprocessing.rr_features import N_RR_FEATURES, compute_rr_features

AI_MODEL_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = AI_MODEL_DIR / "data" / "raw" / "mitdb"

DS1 = (101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203,
       205, 207, 208, 209, 215, 220, 223, 230)
DS2 = (100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214,
       219, 221, 222, 228, 231, 232, 233, 234)
PACED = (102, 104, 107, 217)
ALL_RECORDS = tuple(sorted(DS1 + DS2 + PACED))

AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
# Símbolos MIT-BIH que son latidos (el resto son ritmo, ruido, artefactos...).
# Los RR se calculan con TODOS los latidos, aunque luego no se clasifiquen.
BEAT_SYMBOLS = frozenset("NLRBAaJSVrFejnE/fQ?")

# Registros del mismo paciente (PhysioNet): 201 y 202. Con la división estándar
# 201 está en DS1 y 202 en DS2: fuga conocida que se documenta y se aísla.
SAME_PATIENT = ((201, 202),)

# Q casi no aparece fuera de los registros con marcapasos: por defecto se
# entrena con 4 clases.
DEFAULT_CLASSES = ("N", "S", "V", "F")


@dataclass
class RecordData:
    record: int
    signal_mv: np.ndarray
    fs: float
    ann_samples: np.ndarray
    ann_symbols: np.ndarray
    lead: str


def _inject_os_truststore() -> None:
    """Usa los certificados del sistema si ``truststore`` está instalado.

    Necesario en redes con inspección TLS (antivirus/proxy) donde el bundle de
    certifi no basta. No desactiva la verificación de certificados.
    """
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass


def download_mitbih(data_dir: Path = DEFAULT_DATA_DIR, records: tuple[int, ...] = ALL_RECORDS) -> Path:
    import wfdb

    _inject_os_truststore()
    data_dir.mkdir(parents=True, exist_ok=True)
    missing = [str(r) for r in records if not (data_dir / f"{r}.dat").exists()]
    if missing:
        wfdb.dl_database("mitdb", dl_dir=str(data_dir), records=missing,
                         annotators=["atr"], keep_subdirs=False)
    return data_dir


def load_record(record: int, data_dir: Path = DEFAULT_DATA_DIR, lead: str = "MLII") -> RecordData:
    import wfdb

    path = str(data_dir / str(record))
    header = wfdb.rdheader(path)
    channel = header.sig_name.index(lead) if lead in header.sig_name else 0
    rec = wfdb.rdrecord(path, channels=[channel])
    ann = wfdb.rdann(path, "atr")
    return RecordData(record, rec.p_signal[:, 0].astype(np.float64), float(rec.fs),
                      np.asarray(ann.sample, dtype=np.int64), np.asarray(ann.symbol),
                      header.sig_name[channel])


@dataclass
class BeatDataset:
    X: np.ndarray        # (n, L) latidos normalizados
    R: np.ndarray        # (n, 8) características RR
    y: np.ndarray        # (n,) índice de clase
    rec: np.ndarray      # (n,) registro
    r_index: np.ndarray  # (n,) posición del pico R (a cfg.fs_hz)

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "BeatDataset":
        return BeatDataset(self.X[mask], self.R[mask], self.y[mask], self.rec[mask], self.r_index[mask])

    @staticmethod
    def concat(parts: list["BeatDataset"]) -> "BeatDataset":
        return BeatDataset(*(np.concatenate([getattr(p, f) for p in parts])
                             for f in ("X", "R", "y", "rec", "r_index")))


def _record_dataset(rd: RecordData, cfg: PreprocessConfig, classes: tuple[str, ...], rr_source: str) -> BeatDataset:
    class_idx = {c: i for i, c in enumerate(classes)}
    is_beat = np.isin(rd.ann_symbols, list(BEAT_SYMBOLS))
    labels = np.array([AAMI_MAP.get(s, "") for s in rd.ann_symbols[is_beat]])
    xf = filter_signal(rd.signal_mv, rd.fs, cfg)
    peaks = segmentation.refine_peaks(xf, resample_indices(rd.ann_samples[is_beat], rd.fs, cfg.fs_hz),
                                      cfg.fs_hz, cfg.refine_search_s)
    rec_col = lambda n: np.full(n, rd.record, dtype=np.int64)  # noqa: E731

    if rr_source == "annotations":
        feats, warm, has = compute_rr_features(peaks, cfg.fs_hz, cfg.rr_config())
        keep = np.flatnonzero(has & warm & np.isin(labels, classes))
        beats, kept = segmentation.extract_beats(xf, peaks[keep], cfg.fs_hz, cfg.before_s, cfg.after_s)
        sel = keep[kept]
        return BeatDataset(normalize(beats, cfg.normalization), feats[sel].astype(np.float32),
                           np.array([class_idx[c] for c in labels[sel]], dtype=np.int64),
                           rec_col(len(sel)), peaks[sel])

    if rr_source == "detector":  # como en el dispositivo: picos y RR del detector
        out = [b for b in simulate_device(xf, cfg.fs_hz, cfg, prefiltered=True) if b["warm"]]
        tol = int(round(0.15 * cfg.fs_hz))
        X, R, y, ri = [], [], [], []
        for b in out:
            d = b["r_index"]
            j = int(np.searchsorted(peaks, d))
            cand = [k for k in (j - 1, j) if 0 <= k < len(peaks) and abs(int(peaks[k]) - d) <= tol]
            if not cand:
                continue  # detección falsa: afectó a los RR, pero no tiene etiqueta
            k = min(cand, key=lambda k: abs(int(peaks[k]) - d))
            if labels[k] in class_idx:
                X.append(b["beat"]); R.append(b["rr"]); y.append(class_idx[labels[k]]); ri.append(d)
        return BeatDataset(np.asarray(X, dtype=np.float32).reshape(-1, cfg.input_length),
                           np.asarray(R, dtype=np.float32).reshape(-1, N_RR_FEATURES),
                           np.asarray(y, dtype=np.int64), rec_col(len(y)), np.asarray(ri, dtype=np.int64))
    raise ValueError(f"rr_source desconocido: {rr_source!r}")


def build_dataset(records: tuple[int, ...], cfg: PreprocessConfig, data_dir: Path = DEFAULT_DATA_DIR,
                  classes: tuple[str, ...] = DEFAULT_CLASSES, rr_source: str = "annotations") -> BeatDataset:
    """Latidos + características RR de los registros indicados.

    * ``rr_source="annotations"``: picos de las anotaciones (entrenamiento).
    * ``rr_source="detector"``: picos y RR del detector, como en el dispositivo
      (evaluación realista). Los latidos se etiquetan emparejándolos con las
      anotaciones a ±150 ms; las detecciones falsas cuentan para los RR pero no
      se evalúan.

    Solo incluye latidos "calientes" (ver rr_features.py) que tienen RR siguiente.
    """
    return BeatDataset.concat([_record_dataset(load_record(r, data_dir), cfg, classes, rr_source) for r in records])


def build_beat_dataset(records: tuple[int, ...], cfg: PreprocessConfig, data_dir: Path = DEFAULT_DATA_DIR,
                       classes: tuple[str, ...] = DEFAULT_CLASSES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compatibilidad con el código anterior: ``(X, y, rec)``."""
    ds = build_dataset(records, cfg, data_dir, classes)
    return ds.X, ds.y, ds.rec


def class_counts(y: np.ndarray, classes: tuple[str, ...]) -> dict[str, int]:
    return {c: int(np.sum(y == i)) for i, c in enumerate(classes)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga y resumen de MIT-BIH")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download_mitbih(args.data_dir)
        print(f"MIT-BIH disponible en {args.data_dir}")
    cfg = PreprocessConfig()
    for name, recs in (("DS1", DS1), ("DS2", DS2)):
        _, y, _ = build_beat_dataset(recs, cfg, args.data_dir)
        print(name, class_counts(y, DEFAULT_CLASSES))


if __name__ == "__main__":
    main()
