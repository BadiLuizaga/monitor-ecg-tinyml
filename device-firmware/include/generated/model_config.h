// ARCHIVO GENERADO por ai-model/src/export/export_model.py - no editar a mano.
// Generado: 2026-11-19 20:41 UTC
#pragma once
#include <cstdint>

// Debe coincidir con ai-model/models/preprocess_config.json
#define ECG_MODEL_FS_HZ            360.0f
#define ECG_MODEL_INPUT_LENGTH     252
#define ECG_MODEL_WINDOW_BEFORE    90
#define ECG_MODEL_WINDOW_AFTER     162
#define ECG_MODEL_NUM_CLASSES      4
#define ECG_MODEL_INPUT_SCALE      0.0461284779012f
#define ECG_MODEL_INPUT_ZERO_POINT -18
#define ECG_MODEL_OUTPUT_SCALE     0.00390625f
#define ECG_MODEL_OUTPUT_ZERO_POINT -128

// Posición de la entrada del latido en el modelo TFLite (no asumir 0).
#define ECG_MODEL_BEAT_INPUT_INDEX 0

// Clases AAMI en el orden de salida del modelo.
static const char ECG_MODEL_CLASSES[ECG_MODEL_NUM_CLASSES] = {'N', 'S', 'V', 'F'};
