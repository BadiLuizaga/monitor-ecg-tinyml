# ai-model/

Procesamiento de ECG, entrenamiento, evaluación y exportación del clasificador de latidos para TinyML.

> ⚠️ Modelo **experimental**, de uso académico. No es un dispositivo médico ni sirve para diagnóstico.

## Ruta completa

```sh
cd ai-model
python -m src.training.dataset_loader --download   # 1. descarga MIT-BIH (~90 MB) a data/raw/mitdb
python -m src.training.train                       # 2-6. preprocesa y entrena (DS1)
python -m src.evaluation.evaluate --detector       # 7. evalúa en DS2 (inter-paciente)
python -m src.export.export_model                  # 8. TFLite int8 + archivos C++ para el firmware
```

| Paso | Módulo |
|---|---|
| Carga MIT-BIH, mapeo AAMI y división DS1/DS2 | `src/training/dataset_loader.py` |
| Filtro causal 0,5–40 Hz (+ notch opcional) y remuestreo | `src/preprocessing/filters.py` |
| Detección de picos R y segmentación | `src/preprocessing/segmentation.py` |
| Normalización z-score por latido | `src/preprocessing/normalization.py` |
| Cadena común entrenamiento ↔ firmware, y réplica del pipeline del dispositivo (`simulate_device`) | `src/preprocessing/pipeline.py` |
| Características RR (8; versión latido a latido y por lotes) | `src/preprocessing/rr_features.py` |
| Grupos de validación cruzada por paciente y su verificación | `src/training/cv.py` (`python -m src.training.cv --verify`) |
| CNN 1D: v1 solo latido (7 252 parámetros); v2 latido + RR (~7 900) | `src/training/train.py` |
| Métricas por clase | `src/evaluation/metrics.py`, `evaluate.py` |
| Conversión int8, comprobación TFLM y generación de C++ | `src/export/export_model.py` |

Resultados y limitaciones: ver [`docs/model_documentation.md`](../docs/model_documentation.md).

Opciones útiles:

- `train --quick`: prueba la ruta con pocos registros en unos minutos.
- `train --fs 250`: entrena para un sensor que muestree a 250 Hz.
- `export_model --filters-only --fs <Hz> [--notch 50]`: solo regenera los coeficientes del filtro.

## Pruebas

```sh
python -m pytest ai-model/tests   # desde la raíz del repositorio
```
