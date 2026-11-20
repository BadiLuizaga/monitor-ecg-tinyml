# models/

Artefactos que generan `train.py`, `evaluate.py` y `export_model.py`. **No se suben a git**, porque se regeneran.

| Archivo | Contenido |
|---|---|
| `ecg_model.keras` | Modelo entrenado (float) |
| `ecg_model_float.tflite` / `ecg_model_int8.tflite` | Modelos convertidos; el int8 es el que se despliega |
| `preprocess_config.json` | Cadena de preprocesamiento usada. **El firmware debe coincidir con ella** |
| `model_info.json` | Clases, registros usados, semilla y estado |
| `representative.npy` | 500 latidos para la cuantización int8 |
| `train_history.json`, `eval_report.json`, `export_report.json` | Historial de entrenamiento y métricas |

Los archivos para el firmware se escriben en `device-firmware/include/generated/` y `device-firmware/src/generated/`.
