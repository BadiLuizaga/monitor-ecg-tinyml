// hardware_config.h - Configuración FÍSICA del dispositivo.
//
// ESTE ARCHIVO ESTÁ INCOMPLETO A PROPÓSITO.
// Todos los valores dependen del hardware que aún no se ha entregado y deben
// ser completados por (o con) el responsable de hardware. No hay pines,
// voltajes ni configuración de ADC inventados: solo marcadores.
//
// Mientras ECG_HW_CONFIG_CONFIRMED sea 0, el firmware solo puede compilarse
// con la fuente simulada (TestPatternSource) y lo indica en DEVICE_INFO
// ("simulated": true).
#pragma once

#define ECG_HW_CONFIG_CONFIRMED 0

// --- Placa -----------------------------------------------------------------
// PENDIENTE: modelo exacto (ESP32, ESP32-S3, ESP32-C3...), RAM/PSRAM y flash.
#define ECG_HW_MODEL "TBD"

// --- Adquisición ------------------------------------------------------------
// PENDIENTE: frecuencia de muestreo real y cómo se genera (timer, DRDY del
// front-end...). El valor por defecto coincide con MIT-BIH SOLO para poder
// probar con la fuente simulada.
#define ECG_HW_SAMPLE_RATE_HZ 360.0f  // PROVISIONAL

// PENDIENTE: resolución efectiva del ADC (bits). 0 = desconocida.
#define ECG_HW_ADC_BITS 0

// PENDIENTE: factor de conversión cuentas -> mV (ganancia del front-end y
// referencia del ADC). 0 = sin calibrar; se envía null en DEVICE_INFO.
#define ECG_HW_MV_PER_COUNT 0.0f

// PENDIENTE: número de derivaciones.
#define ECG_HW_LEADS 1

// PENDIENTE: ¿el front-end ofrece detección de electrodo suelto (lead-off)?
#define ECG_HW_HAS_LEAD_OFF_DETECTION 0

// --- Pines ------------------------------------------------------------------
// PENDIENTE: definir SOLO cuando el responsable de hardware confirme el
// conexionado. No asignar pines por suposición.
// #define ECG_HW_PIN_ECG_ADC      ?
// #define ECG_HW_PIN_LEAD_OFF_P   ?
// #define ECG_HW_PIN_LEAD_OFF_N   ?
// #define ECG_HW_PIN_STATUS_LED   ?

#if ECG_HW_CONFIG_CONFIRMED
#if ECG_HW_ADC_BITS == 0
#error "ECG_HW_ADC_BITS debe definirse al confirmar el hardware"
#endif
#endif
