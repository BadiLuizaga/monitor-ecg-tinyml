// ARCHIVO GENERADO por ai-model/src/export/export_model.py - no editar a mano.
// Generado: 2026-11-19 20:41 UTC
#pragma once

// Filtro: Butterworth paso banda 0.5-40.0 Hz, orden 2; notch: ninguno.
// Diseñado para fs = 360.0 Hz.
// PROVISIONAL: si la frecuencia real del sensor es distinta, regenerar con
//   python -m src.export.export_model --filters-only --fs <Hz>
#define ECG_FILTER_FS_HZ 360.0f
#define ECG_FILTER_NUM_SECTIONS 2

// Cada fila: b0, b1, b2, a1, a2
static const float ECG_FILTER_SOS[ECG_FILTER_NUM_SECTIONS][5] = {
  {0.0787623532949f, 0.15752470659f, 0.0787623532949f, -1.06709242864f, 0.384234247441f},
  {1.0f, -2.0f, 1.0f, -1.98766422111f, 0.987741721009f}
};
