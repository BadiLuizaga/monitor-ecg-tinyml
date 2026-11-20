# Constantes compartidas

Cada valor tiene un único origen. Este documento solo los enumera: si hay que cambiar alguno, se cambia en su origen.

| Constante | Valor actual | Estado | Origen |
|---|---|---|---|
| Versión del protocolo | 1 | Definido | `communication/protocol/protocol_config.json` |
| UUID (servicio y características) | ver `uuid_config.md` | **PROVISIONAL** | `protocol_config.json` |
| Frecuencia de muestreo del sensor | desconocida (se usa 360 Hz en simulación) | **PENDIENTE HW** | `device-firmware/include/hardware_config.h` |
| Resolución del ADC | desconocida | **PENDIENTE HW** | `hardware_config.h` |
| Factor mV por cuenta | desconocido | **PENDIENTE HW** | `hardware_config.h` |
| Número de derivaciones | 1 (supuesto) | **PENDIENTE HW** | `hardware_config.h` |
| Formato de muestra BLE | int16 LE | Definido | `protocol_config.json` |
| Frecuencia de trabajo del modelo | 360 Hz (la de MIT-BIH) | Provisional | `ai-model/models/preprocess_config.json` |
| Filtro | Butterworth 0,5–40 Hz, orden 2, causal | Definido | `ai-model/src/preprocessing/filters.py` |
| Notch de red | desactivado (50 o 60 Hz, configurable) | Pendiente de evaluar con la señal real | `PreprocessConfig.notch_hz` |
| Ventana del latido | 0,25 s antes y 0,45 s después del R (252 muestras a 360 Hz) | Definido | `ai-model/src/preprocessing/segmentation.py` |
| Normalización | z-score por latido | Definido | `ai-model/src/preprocessing/normalization.py` |
| Clases del modelo | N, S, V, F (AAMI) | Definido | `ai-model/models/model_info.json` |
| Características RR (v2) | 8, en el orden de `ECG_RR_FEATURE_ORDER` | Definido | `ai-model/src/preprocessing/rr_features.py`, `integration_config.h` |
| RR medio local | Media de los 10 RR anteriores, sin incluir el actual | Definido | `PreprocessConfig.rr_local_beats` / `ECG_RR_LOCAL_BEATS` |
| RR de largo plazo | Media exponencial con constante de 300 s | Definido | `rr_long_tau_s` / `ECG_RR_LONG_TAU_S` |
| RR máximo / timeout | 3 s | Definido | `rr_max_s` / `ECG_RR_MAX_S` |
| RR de referencia y recorte | 0,8 s; log2 recortado a ±2 | Definido | `rr_ref_s`, `rr_clip` |
| Calentamiento RR | 3 RR y 30 s (sin alertas) | Provisional | `rr_warmup_*` / `ECG_RR_WARMUP_*` |
| Ventana guardada por latido pendiente | 256 muestras (~4 KB con 4 pendientes) | Definido (aumentar si fs > 365 Hz) | `ECG_MAX_BEAT_WINDOW` |
| Calibración int8 | Hasta 150 latidos por clase | Definido | `train.REPRESENTATIVE_PER_CLASS` |
| Saturación int8 máxima | < 0,5 % por entrada | Definido | `export_model.MAX_INPUT_SATURATION` |
| Clases que generan alerta | V | Provisional (S y F no son fiables todavía) | `integration_config.h` |
| Confianza mínima para alertar | 0,80 | Provisional | `integration_config.h` |
| Tiempo mínimo entre alertas | 2 s | Provisional | `integration_config.h` |
| MTU mínimo para el stream | 185 | Definido | `protocol_config.json` |
| Tamaño máximo de JSON | 180 B (notify/write) y 512 B (`DEVICE_INFO`) | Definido | `protocol_config.json` |
| Timeout de enlace | 3 s | Provisional | `protocol_config.json` |
| Backoff de reconexión | 1→30 s, ×2, 10 intentos | Provisional | `protocol_config.json` |
