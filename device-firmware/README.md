# device-firmware/

Lógica del firmware para el ESP32, escrita en C++17 portable y **separada de la configuración física**.

> **Estado:** la lógica está **compilada y probada en PC** (g++ 16.2, C++17, `-Wall -Wextra -Werror`). Las pruebas de `test/host/` pasan: bytes idénticos a Python, filtro y detector equivalentes a Python y pipeline completo. **No se ha compilado para ESP32** y `inference_tflm.cpp` no se ha compilado para ESP32, porque falta la librería TFLM. **No hay port a ESP32 todavía:** `main.cpp` se detiene con `#error` si se compila para ESP32, para evitar desplegar por error un firmware incompleto.
>
> **Entorno decidido (E0):** PlatformIO + framework Arduino + NimBLE. `platformio.ini` está **preparado pero sin placa** (no hay `[env:...]` activo) y **no se ha descargado nada**. Ver [`docs/integration.md` §6](../docs/integration.md).

## Estructura

| Archivo | Responsabilidad | Depende del hardware |
|---|---|---|
| `include/hardware_config.h` | Placa, frecuencia de muestreo, ADC, pines. **Todo PENDIENTE** | **Sí** |
| `include/integration_config.h` | UUID, protocolo, detector y alertas | No |
| `include/ble_protocol.h`, `src/ble_protocol.cpp` | Paquetes binarios, JSON, parser de comandos, nombre a partir de la MAC (`deviceIdFromMac`, `buildDeviceName`), `IBleTransport` y `EcgPacketizer` | No |
| `include/ble_transport.h`, `src/ble_transport.cpp` | `ControlQueue` (cola SPSC sin memoria dinámica) y `QueuedBleTransport`: el callback BLE solo encola y `service()`, desde `loop()`, entrega los comandos al pipeline. Base del futuro `Esp32BleTransport` | No |
| `platformio.ini` | Proyecto PlatformIO (Arduino + NimBLE 2.1.0, versiones fijadas). **Sin placa**; excluye `main.cpp` | Sí (placa pendiente) |
| `include/signal_processing.h`, `src/signal_processing.cpp` | Biquads, detector de picos R, z-score, `RrFeatureTracker` y `quantizeInt8` (equivalentes a los de Python) | No |
| `include/ecg_source.h`, `src/ecg_source.cpp` | Interfaz `IEcgSource` y `TestPatternSource` **simulada** | Interfaz no; la fuente real sí |
| `include/inference.h`, `src/inference_tflm.cpp` | `IInferenceEngine`, *stub* y motor TFLM. El motor está **compilado y ejecutado en PC** con el modelo v1 (T1, no aprobada) y **no se ha probado en el ESP32** | Requiere TFLM. Ver `test/tflm/README.md` |
| `include/alert_manager.h`, `src/alert_manager.cpp` | Reglas de alerta (ventana, confianza, tiempo mínimo, clases) | No |
| `include/ecg_pipeline.h`, `src/ecg_pipeline.cpp` | Orquestación completa | No |
| `include/generated/*`, `src/generated/*` | **Generados** por `ai-model/src/export/export_model.py` | — |
| `src/main.cpp` | Punto de entrada: demo en PC o `#error` en ESP32. PlatformIO no lo compila | — |
| `src/esp32/` *(E2, no existe aún)* | `Esp32BleTransport : QueuedBleTransport` (NimBLE) y `main_esp32.cpp` (`setup()`/`loop()`) | Sí |
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

Comandos (CONTROL), sin ejecutar nada en el contexto de la pila BLE:
  callback onWrite (tarea BLE) ─► QueuedBleTransport::onControlWritten() ─► ControlQueue (SPSC, 4 × 180 B)
  loop(): ble.service() ─► EcgPipeline::onControlWrite() ─► cmd_ack por STATUS
          pipeline.poll(millis())
```

### Errores notificados en `STATUS`

`error_code` se calcula por prioridad y **solo se notifica al cambiar**:
`LEAD_OFF` > `BLE_TX_QUEUE_FULL` > `INFERENCE_FAILED` > error persistente (`MODEL_NOT_LOADED`, `CONFIG_MISMATCH`) > `OK`.

| Error | Se activa | Se despeja |
|---|---|---|
| `BLE_TX_QUEUE_FULL` (21) | Un paquete `ECG_DATA` o una `ALERT` no se pudo encolar (`notify()` = `false`) | El siguiente paquete o alerta encolado; también `stop_stream` o la desconexión |
| `INFERENCE_FAILED` (31) | `classify()` falla | La siguiente inferencia correcta |

Si el propio `STATUS` no se puede notificar (cola llena), se reintenta en el siguiente `poll()`. El valor de **lectura** de `STATUS` se refresca cada `ECG_STATUS_READ_REFRESH_MS` (100 ms), así que `uptime_ms` no queda congelado.

## Pruebas en PC

```sh
sh device-firmware/test/host/run_host_tests.sh   # requiere g++ (C++17) y el Python del proyecto
```

Comprueban:

- Bytes idénticos a los del códec Python.
- JSON y parser de comandos.
- Nombre `ECG-TINYML-XXXX` a partir de la MAC e id externo; un id inválido impide anunciar.
- Cola SPSC de `CONTROL`: orden FIFO, cola llena (se descarta la escritura nueva), escrituras de más de 180 B y vuelta de los índices.
- `CONTROL` diferido: el callback solo encola (sin cambio de estado ni notificaciones dentro de él) y `service()` entrega; `service(n)` respeta el máximo; errores `INVALID_COMMAND`/`UNSUPPORTED_PROTOCOL_VERSION`; escrituras sin conexión ignoradas; comando que llega antes de que `poll()` vea la conexión.
- **D3:** `BLE_TX_QUEUE_FULL` se notifica una vez, es legible por `STATUS`, se despeja al recuperarse y se olvida al desconectar.
- **D5:** `INFERENCE_FAILED` se notifica una vez y se despeja con la siguiente inferencia correcta.
- **D6:** `uptime_ms` de la lectura de `STATUS` se refresca sin generar notificaciones.
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

- Identificar la placa y activar `[env:cardioalert]` en `platformio.ini`.
- Implementar `Esp32BleTransport` (sobre `QueuedBleTransport`, NimBLE) y `src/esp32/main_esp32.cpp` (E2).
- Implementar `HardwareEcgSource`.
- Integrar TFLM.
- Medir memoria y latencia.
