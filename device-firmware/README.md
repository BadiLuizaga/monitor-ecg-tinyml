# device-firmware/

Lógica del firmware para el ESP32, escrita en C++17 portable y **separada de la configuración física**.

> **Estado:** la lógica está **compilada y probada en PC** (g++ 16.2, C++17, `-Wall -Wextra -Werror`). Las pruebas de `test/host/` pasan: bytes idénticos a Python, filtro y detector equivalentes a Python y pipeline completo. **No se ha compilado para ESP32** y `inference_tflm.cpp` no se ha compilado, porque falta la librería TFLM. **No hay port a ESP32 todavía:** `main.cpp` se detiene con `#error` si se compila para ESP32, para evitar desplegar por error un firmware incompleto.

## Estructura

| Archivo | Responsabilidad | Depende del hardware |
|---|---|---|
| `include/hardware_config.h` | Placa, frecuencia de muestreo, ADC, pines. **Todo PENDIENTE** | **Sí** |
| `include/integration_config.h` | UUID, protocolo, detector y alertas | No |
| `include/ble_protocol.h`, `src/ble_protocol.cpp` | Paquetes binarios, JSON, parser de comandos, `IBleTransport` y `EcgPacketizer` | No |
| `include/signal_processing.h`, `src/signal_processing.cpp` | Biquads, detector de picos R, z-score, `RrFeatureTracker` y `quantizeInt8` (equivalentes a los de Python) | No |
| `include/ecg_source.h`, `src/ecg_source.cpp` | Interfaz `IEcgSource` y `TestPatternSource` **simulada** | Interfaz no; la fuente real sí |
| `include/inference.h`, `src/inference_tflm.cpp` | `IInferenceEngine`, *stub* y motor TFLM. El motor está **compilado y ejecutado en PC** con el modelo v1 (T1, no aprobada) y **no se ha probado en el ESP32** | Requiere TFLM. Ver `test/tflm/README.md` |
| `include/alert_manager.h`, `src/alert_manager.cpp` | Reglas de alerta (ventana, confianza, tiempo mínimo, clases) | No |
| `include/ecg_pipeline.h`, `src/ecg_pipeline.cpp` | Orquestación completa | No |
| `include/generated/*`, `src/generated/*` | **Generados** por `ai-model/src/export/export_model.py` | — |
| `src/main.cpp` | Punto de entrada: demo en PC o `#error` en ESP32 | — |
| `test/host/` | Pruebas en PC y vectores de paridad con Python | — |

## Flujo de datos

```
IEcgSource ──► EcgPacketizer ──► IBleTransport (ECG_DATA)
     │
     └─► BiquadCascade ─► RPeakDetector ─► RrFeatureTracker (emite el latido k al llegar el k+1, o a los 3 s)
                              │                       │ 8 características RR
                              └─► cola de pendientes: ventana de 252 muestras COPIADA al completarse
                                                      │
                     z-score ─► IInferenceEngine (TFLM int8: latido + RR) ─► AlertManager ─► IBleTransport (ALERT)
                                                                       (no alerta en calentamiento RR)
```

## Pruebas en PC

```sh
sh device-firmware/test/host/run_host_tests.sh   # requiere g++ (C++17) y el Python del proyecto
```

Comprueban:

- Bytes idénticos a los del códec Python.
- JSON y parser de comandos.
- Que el filtro y el detector C++ coinciden con Python sobre una señal de referencia.
- El pipeline completo con un transporte simulado: integridad de muestras, MTU insuficiente, lead-off y que no se inventan clasificaciones sin modelo.
- **C1:** características RR idénticas a Python (697 latidos, con pausas y reinicio).
- **C2:** cuantización int8 bit a bit igual que Python.
- **C3:** el pipeline entrega al modelo lo mismo que `simulate_device` (Python): los mismos 112 latidos, en la misma muestra.
  - Es obligatorio que al menos un latido se clasifique por *timeout*, con la ventana ya fuera del buffer circular.
  - Ninguna alerta debe emitirse durante el calentamiento.

Las pruebas se compilan con g++ 16.2 (w64devkit 2.11.0 en `.tools/w64devkit`), con `-Wall -Wextra -Werror`.

## Pendiente para el ESP32

Ver [`docs/integration.md`](../docs/integration.md):

- Implementar `HardwareEcgSource` y `Esp32BleTransport`.
- Integrar TFLM.
- Elegir el entorno de compilación.
- Medir memoria y latencia.
