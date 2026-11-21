// integration_config.h - Parámetros LÓGICOS de integración (no físicos).
//
// Los valores del protocolo deben coincidir con
// communication/protocol/protocol_config.json; la prueba
// communication/tests/test_protocol.py::test_firmware_header_matches_config
// falla si divergen.
#pragma once

// --- Protocolo BLE (v1) ----------------------------------------------------
#define ECG_PROTOCOL_VERSION 1
#define ECG_DEVICE_NAME_PREFIX "ECG-TINYML"
#define ECG_FW_VERSION "0.1.0-dev"

// PROVISIONAL: UUID generados aleatoriamente para desarrollo.
#define ECG_SERVICE_UUID "61860001-2b45-4f52-a512-0ba5fd3bf0a4"
#define ECG_CHAR_ECG_DATA_UUID "61860002-2b45-4f52-a512-0ba5fd3bf0a4"
#define ECG_CHAR_ALERT_UUID "61860003-2b45-4f52-a512-0ba5fd3bf0a4"
#define ECG_CHAR_STATUS_UUID "61860004-2b45-4f52-a512-0ba5fd3bf0a4"
#define ECG_CHAR_DEVICE_INFO_UUID "61860005-2b45-4f52-a512-0ba5fd3bf0a4"
#define ECG_CHAR_CONTROL_UUID "61860006-2b45-4f52-a512-0ba5fd3bf0a4"

#define ECG_PACKET_HEADER_SIZE 10
#define ECG_ATT_OVERHEAD 3
#define ECG_DEFAULT_MTU 23
#define ECG_MAX_SAMPLES_PER_PACKET 255
#define ECG_JSON_MAX_BYTES 180
#define ECG_JSON_MAX_READ_BYTES 512
#define ECG_MIN_MTU_FOR_JSON 185

// Nombre anunciado: "<prefijo>-<id>". En el ESP32 el id son los 4 últimos
// dígitos hexadecimales de la MAC (deviceIdFromMac); sin hardware se puede
// pasar un id externo. El nombre va en la respuesta al escaneo, que admite
// como máximo 29 caracteres.
#define ECG_DEVICE_ID_MAX_LEN 8
#define ECG_DEVICE_NAME_MAX_LEN 29

// Cola SPSC de escrituras de CONTROL (callback BLE -> loop()). Potencia de 2.
// Si está llena, se descarta la escritura NUEVA y la app no recibe cmd_ack.
#define ECG_CONTROL_QUEUE_SLOTS 4

// Cada cuánto se refresca el valor de lectura de STATUS (uptime_ms).
#define ECG_STATUS_READ_REFRESH_MS 100

#define ECG_FLAG_LEAD_OFF 1
#define ECG_FLAG_SOURCE_OVERFLOW 2
#define ECG_FLAG_SIMULATED 4
#define ECG_FLAG_SATURATED 8

// --- Detección de latidos (mismos valores que segmentation.py) -------------
#define ECG_DET_INTEGRATION_S 0.150f
#define ECG_DET_REFRACTORY_S 0.200f
#define ECG_DET_LEARNING_S 2.0f
#define ECG_DET_THRESHOLD_RATIO 0.25f
// Límite de memoria del detector: soporta fs <= 1000 Hz.
#define ECG_DET_MAX_INTEGRATION_SAMPLES 150

// --- Características RR (segunda entrada del modelo) ----------------------
// Deben coincidir con PreprocessConfig (ai-model/src/preprocessing/pipeline.py);
// lo verifica ai-model/tests/test_model.py::test_p5_integration_config_matches_python_rr_config
// y, si el modelo exportado usa RR, un static_assert contra generated/model_config.h.
#define ECG_RR_LOCAL_BEATS 10
#define ECG_RR_LONG_TAU_S 300.0f
#define ECG_RR_MAX_S 3.0f
#define ECG_RR_REF_S 0.8f
#define ECG_RR_CLIP 2.0f
#define ECG_RR_WARMUP_MIN_RR 3
#define ECG_RR_WARMUP_S 30.0f
#define ECG_RR_NUM_FEATURES 8
#define ECG_RR_FEATURE_ORDER "log2_pre_local,log2_post_local,log2_post_pre,log2_pre_long,log2_local_long,log2_pre_ref,log2_local_ref,log2_long_ref"

// --- Ventana del latido / buffer -------------------------------------------
// Si existe include/generated/model_config.h se usan sus valores.
#define ECG_BEAT_RING_SIZE 1024  // muestras filtradas almacenadas (potencia de 2)
#define ECG_MAX_PENDING_BEATS 4
// Muestras de ventana copiadas en cada latido pendiente (la ventana se guarda en
// cuanto está completa, así no depende del buffer circular aunque el RR
// siguiente tarde hasta ECG_RR_MAX_S). 256 cubre la ventana de 0,7 s hasta
// ~365 Hz; si el sensor muestrea más rápido, aumentarlo (RAM: 4 x N x 4 bytes).
#define ECG_MAX_BEAT_WINDOW 256

// --- Alertas (PROVISIONAL: ajustar tras evaluar el modelo) -----------------
#define ECG_ALERT_MIN_CONFIDENCE 0.80f
#define ECG_ALERT_WINDOW_BEATS 8       // ventana deslizante de latidos
#define ECG_ALERT_MIN_ABNORMAL 1       // latidos anómalos en la ventana para alertar
#define ECG_ALERT_COOLDOWN_S 2.0f
// Clases que pueden generar alertas. Solo "V": en la evaluación DS2 el modelo
// actual NO detecta S ni F (sensibilidad ~0) y confunde latidos N con F, lo
// que produciría falsas alertas. Ver docs/model_documentation.md.
#define ECG_ALERT_ELIGIBLE_CLASSES "V"

// --- TFLM -------------------------------------------------------------------
// PROVISIONAL: el tamaño real se mide en el ESP32 con arena_used_bytes().
#define ECG_TFLM_ARENA_BYTES (16 * 1024)
