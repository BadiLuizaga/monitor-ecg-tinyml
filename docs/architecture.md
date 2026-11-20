# Arquitectura del sistema

> Proyecto académico. **No es un dispositivo médico** ni una herramienta de diagnóstico clínico.

## Visión general

```
┌──────────────── Dispositivo (ESP32) ────────────────┐        BLE (GATT)        ┌──── App Flutter ────┐
│ Sensor ECG ─► IEcgSource ─┬─► EcgPacketizer ────────┼── ECG_DATA (binario) ───►│ gráfica en vivo     │
│  (PENDIENTE)              │                         │                          │                     │
│                           └─► filtro IIR causal     │                          │                     │
│                               ─► detector R         │                          │                     │
│                               ─► RR (+1 latido)     │                          │                     │
│                               ─► ventana 0,7 s      │                          │                     │
│                                  (copiada)          │                          │                     │
│                               ─► z-score            │                          │                     │
│                               ─► CNN int8 (TFLM)    │                          │                     │
│                                  latido + 8 RR      │                          │                     │
│                               ─► AlertManager ──────┼── ALERT (JSON) ─────────►│ aviso / voz         │
│ estado / errores ───────────────────────────────────┼── STATUS / DEVICE_INFO ─►│ estado              │
│ comandos ◄──────────────────────────────────────────┼── CONTROL (JSON) ◄───────│ start/stop/ack      │
└─────────────────────────────────────────────────────┘                          └─────────────────────┘

┌────────── Desarrollo (PC) ──────────┐
│ MIT-BIH ─► ai-model (Python) ─► modelo int8 + coeficientes del filtro ─► device-firmware/.../generated/
│ simulador ─► mismos bytes que el dispositivo (sin BLE real)
└─────────────────────────────────────┘
```

## Principios de diseño

1. **Una sola cadena de preprocesamiento.** `PreprocessConfig` (Python) define frecuencia, filtro, ventana y normalización. El firmware recibe los coeficientes **generados**, no copiados a mano. Además, el filtro es **causal** también en el entrenamiento, para que el modelo vea lo mismo que verá en el ESP32.
2. **Una sola fuente de verdad del protocolo.** `communication/protocol/protocol_config.json`. Las pruebas fallan si el firmware diverge.
3. **Lógica separada del hardware.** `hardware_config.h` concentra todo lo físico, y está pendiente. El resto del firmware depende solo de interfaces (`IEcgSource`, `IBleTransport`, `IInferenceEngine`).
4. **Nada simulado se presenta como real.** El flag `SIMULATED` en los paquetes, `"simulated": true` en `DEVICE_INFO` y en las alertas, `hw_model: "SIMULATOR"`, y el `#error` en `main.cpp` para ESP32 lo garantizan.
5. **Python y C++ hacen exactamente lo mismo.** `simulate_device` (Python) replica muestra a muestra el `EcgPipeline` de C++. Las pruebas C1–C3 comprueban que son equivalentes: mismos latidos, en la misma muestra y con las mismas entradas al modelo.
6. **Clasificación con un latido de retraso.** Las características RR necesitan el RR siguiente. La ventana de cada latido se copia al completarse, así que el latido anterior a una pausa larga también se clasifica (por *timeout*, a los 3 s). Durante el calentamiento RR (30 s) no hay alertas.
7. **Sin modelo no hay clasificación.** Si TFLM no está disponible, el dispositivo transmite el ECG y declara `MODEL_NOT_LOADED`. Nunca inventa resultados.

## Estado por módulo

| Módulo | Implementado | Probado | Simulado | Pendiente |
|---|---|---|---|---|
| Preprocesamiento (Python) | ✅ | ✅ pytest | — | Notch según la red eléctrica del lugar de prueba |
| Modelo v1 (solo morfología) | ✅ | ✅ en MIT-BIH DS2 | — | Sustituir por la v2 |
| Modelo v2 (latido + RR): características, pipeline y exportación | ✅ | ✅ P1–P5 y C1–C3 | — | Cerrado: la validación cruzada no cumplió los criterios en sus dos iteraciones. No hay modelo final ni evaluación de DS2, y el firmware sigue con la v1. T1 con la v1 en PC, no aprobada |
| Protocolo BLE (códec Python) | ✅ | ✅ pytest | — | UUID definitivos; prueba con radio real |
| Simulador | ✅ | ✅ pytest | ✅ (es el simulador) | — |
| Firmware: lógica | ✅ | ✅ Pruebas en PC (g++) | Fuente de señal | Compilar para ESP32 |
| Firmware: BLE ESP32 | ❌ | — | — | `Esp32BleTransport` |
| Firmware: adquisición | ❌ | — | — | `HardwareEcgSource` (requiere las especificaciones) |
| Firmware: TFLM | Esqueleto | ❌ | — | Integrar la librería y medir arena y latencia |
| App móvil | Contrato definido | — | — | Responsabilidad de Eva |

Detalles: [integration.md](integration.md) · [bluetooth_documentation.md](bluetooth_documentation.md) · [model_documentation.md](model_documentation.md) · [requirements.md](requirements.md)
