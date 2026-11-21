# Integración software ↔ hardware

Qué está listo, qué falta y cómo conectar el software al ESP32 real.

## 1. Qué está implementado, simulado o pendiente

| Componente | Estado | Evidencia |
|---|---|---|
| Ruta MIT-BIH → modelo int8 → archivos C++ | ✅ Implementado y ejecutado | `ai-model/models/*_report.json`, `device-firmware/*/generated/` |
| Protocolo BLE v1 (Python) | ✅ Implementado y probado | `communication/tests/` |
| Simulador del dispositivo | ✅ Implementado y probado (**simulación**) | `mock_ecg_device.py selftest` |
| Lógica del firmware (C++) | ✅ Compilada y probada en PC; **sin compilar para ESP32** | `device-firmware/test/host/` |
| Fuente de señal del firmware | 🔶 Solo simulada (`TestPatternSource`) | — |
| Transporte BLE del ESP32 | 🔶 Base portable lista y probada en PC (`QueuedBleTransport`, cola SPSC de `CONTROL`); `Esp32BleTransport` (NimBLE) **pendiente (E2)** | `include/ble_transport.h`, `test/host/` |
| Proyecto de compilación ESP32 | 🔶 `platformio.ini` preparado (Arduino + NimBLE); **sin placa, sin descargar** | `device-firmware/platformio.ini`, §6 |
| Adquisición real (ADC/front-end) | ❌ Pendiente de especificaciones | interfaz `IEcgSource`, `hardware_config.h` |
| Inferencia TFLM en el ESP32 | 🔶 Compilada y ejecutada en PC (T1 no aprobada: ±1 LSB no se cumple; clases 1007/1007). **Sin compilar para ESP32** | `src/inference_tflm.cpp`, `test/tflm/` |
| App móvil | Contrato definido | `shared/protocol/communication_contract.md` |

### Entrada RR y memoria del firmware (modelo v2)

- El pipeline clasifica cada latido **cuando llega el siguiente** (o a los 3 s, por *timeout*), porque el modelo necesita el RR siguiente. La alerta llega, por tanto, **un latido más tarde** que con la v1.
- **RAM adicional:** cada latido pendiente guarda una copia de su ventana (`ECG_MAX_BEAT_WINDOW` = 256 muestras). Con 4 pendientes son **unos 4 KB**. Si el sensor muestrea a más de unos 365 Hz, hay que aumentar `ECG_MAX_BEAT_WINDOW`; si no, el dispositivo se niega a arrancar con `CONFIG_MISMATCH`.
- `TflmInferenceEngine` localiza las entradas por los índices exportados (`ECG_MODEL_BEAT_INPUT_INDEX` y `ECG_MODEL_RR_INPUT_INDEX`), nunca por posición.
- Si los parámetros RR con los que se entrenó el modelo no coinciden con los de `integration_config.h`, el firmware **no compila** (`static_assert`).

## 2. Información que hay que pedir al responsable de hardware

| # | Dato | Para qué se necesita | Dónde se configura |
|---|---|---|---|
| 1 | Modelo exacto de ESP32 y módulo (RAM, PSRAM, flash) | Tamaño de la arena TFLM, pila BLE, elección del SDK | `ECG_HW_MODEL`, `ECG_TFLM_ARENA_BYTES` |
| 2 | Sensor o front-end ECG (referencia y hoja de datos) | Driver de `HardwareEcgSource` | nuevo `src/hal/` |
| 3 | Interfaz con el sensor: ADC interno, SPI o I²C | Driver | `HardwareEcgSource` |
| 4 | **Frecuencia de muestreo** real y cómo se garantiza (timer, DRDY) | Filtros, detector y **reentrenamiento** del modelo | `ECG_HW_SAMPLE_RATE_HZ`, `train --fs` |
| 5 | Resolución efectiva, referencia y ganancia del ADC → mV por cuenta | Escala en la app; detección de saturación | `ECG_HW_ADC_BITS`, `ECG_HW_MV_PER_COUNT` |
| 6 | Número de derivaciones y su colocación (¿equivalente a MLII?) | El modelo se entrenó con MLII | `ECG_HW_LEADS`; posible reentrenamiento |
| 7 | Filtrado analógico del front-end (paso banda, notch) | Evitar filtrar dos veces y ajustar la banda digital | `PreprocessConfig` |
| 8 | Detección de electrodo suelto (lead-off) y sus pines | Flag `LEAD_OFF` y alertas | `ECG_HW_HAS_LEAD_OFF_DETECTION`, pines |
| 9 | Pines asignados (ADC, lead-off, LED…) | Driver | `hardware_config.h` |
| 10 | Alimentación (batería o USB) y su capacidad | Intervalo de conexión BLE, `LOW_BATTERY` | — |
| 11 | ~~Entorno de compilación preferido~~ **Decidido (E0): PlatformIO + Arduino + NimBLE** (§6). Falta confirmar que la placa es compatible | Proyecto de compilación y librerías BLE/TFLM | `platformio.ini` |
| 12 | Frecuencia de la red eléctrica donde se probará (50 o 60 Hz) | Notch | `--notch` |
| 13 | Una **grabación** de 5–10 min del sensor real (CSV o binario) | Validar el filtro y el detector, y reentrenar con datos reales antes de conectar nada | `ai-model/data/` |

## 3. Pasos para conectar con el dispositivo real

1. **Recibir las especificaciones** de la tabla anterior y completar `hardware_config.h`, con `ECG_HW_CONFIG_CONFIRMED 1`.
2. **Ejecutar las pruebas en PC** (`run_host_tests.sh` o la CI) tras cada cambio. Ahora mismo pasan.
3. **Si la frecuencia del sensor no es 360 Hz:**
   ```sh
   python -m src.training.train --fs <Hz>
   python -m src.evaluation.evaluate
   python -m src.export.export_model
   ```
   El firmware se niega a funcionar si las frecuencias no coinciden (`CONFIG_MISMATCH`).
4. **Activar el proyecto PlatformIO** (`device-firmware/platformio.ini`): descomentar `[env:cardioalert]` con el `board` real. Ya excluye `main.cpp`.
5. **Implementar `Esp32BleTransport : QueuedBleTransport`** con NimBLE-Arduino, en `src/esp32/` (ver §6.3):
   - 5 características según `ble_spec.md`;
   - `notify` debe devolver `false` si la cola está llena;
   - MTU negociado en `mtu()`;
   - el callback `onWrite` de `CONTROL` **solo** llama a `onControlWritten()`; `loop()` llama a `ble.service()` y después a `pipeline.poll()`.
6. **Validar el BLE con `TestPatternSource`** antes de tocar el sensor (pruebas de `bluetooth_documentation.md`).
7. **Implementar `HardwareEcgSource : IEcgSource`**, con muestreo por timer o DRDY e interrupción → *ring buffer* → `read()` no bloqueante. Señalizar desbordamientos con `SOURCE_OVERFLOW`.
8. **Integrar TFLM** (por ejemplo, `esp-tflite-micro`):
   - compilar con `-DECG_HAVE_TFLM=1`;
   - instanciar `TflmInferenceEngine`;
   - medir `arenaUsedBytes()` y ajustar `ECG_TFLM_ARENA_BYTES`.
9. **Medir** RAM libre, latencia por latido (debe ser mucho menor que el intervalo RR mínimo, unos 300 ms) y consumo.
10. **Validar de extremo a extremo** con la app de Eva.

## 4. Pruebas cuando recibamos el ESP32

| Prueba | Criterio de aceptación (propuesto) |
|---|---|
| Arranque y `DEVICE_INFO` | `simulated: false`, `hw_model` y fs correctos |
| Stream BLE de 10 min (patrón de prueba) | Pérdida < 0,1 % a 1 m; muestras contiguas según `seq` e índice |
| Fidelidad de la señal | Grabación del dispositivo comparada con una referencia, si existe un simulador de paciente o un ECG comercial |
| Filtro y detector en el dispositivo | Mismos picos R que Python sobre la misma grabación (±1 muestra) |
| Inferencia (T1) | Vectores de referencia (latido + RR int8 y salida int8 esperada de Python): misma clase en ≥ 99,5 % y salidas a ±1 LSB. Se puede adelantar compilando TFLM para PC. **Ejecutada en PC con el modelo v1 (2026-11-19): no aprobada.**
- La clase coincide en 1007/1007 latidos, también con el motor real `inference_tflm.cpp`, que ya compila.
- La diferencia máxima es de 3 LSB.
- La causa es el doble redondeo de TFLM en la recuantización de FULLY_CONNECTED por canal.

Ver `device-firmware/test/tflm/README.md` |
| Retraso de la clasificación | Un latido (o 3 s tras el último latido antes de una pausa) |
| Memoria | Arena TFLM medida y RAM libre > 20 % |
| Latencia | Fin del latido → alerta BLE < 100 ms (por definir) |
| Lead-off | Desconectar un electrodo genera el flag, un `STATUS` y una `ALERT` |
| Reconexión | La app recupera el stream tras un corte de 30 s |
| Autonomía | Horas con batería, si aplica |

## 5. Riesgos conocidos

- **Frecuencia del sensor distinta de 360 Hz:** hay que reentrenar. Está previsto (`--fs`).
- **Derivación distinta de MLII:** la morfología cambia y el rendimiento puede caer. Habrá que reentrenar o ajustar con datos propios.
- **Front-end ruidoso** (cables, movimiento): el detector de R puede degradarse. Validar con la grabación del punto 13.
- **Modelo débil en S y F** (ver `model_documentation.md`): por eso solo `V` puede generar alertas, de forma provisional.

## 6. Entorno de compilación: PlatformIO + Arduino + NimBLE (decisión E0)

### 6.1 Decisión

| Elemento | Elección | Motivo |
|---|---|---|
| Herramienta | **PlatformIO** | Versiones fijadas por proyecto, `build_src_filter` para excluir `main.cpp` sin duplicar fuentes, core y paquetes reubicables dentro del proyecto |
| Framework | **Arduino** (sobre ESP-IDF) | Entorno inicial más sencillo; las tres variantes (ESP32, S3, C3) están soportadas. Si más adelante hace falta, el código portable pasa a ESP-IDF sin cambios |
| Pila BLE | **NimBLE** (NimBLE-Arduino) | Menos RAM y flash que Bluedroid (importante junto con TFLM); disponible en ESP32, S3 y C3; en la versión 2.x `notify()` devuelve `bool` |
| Placa | **Sin fijar** | No se conoce el modelo físico; no se inventa un board ID ni pines |

### 6.2 Versiones fijadas (sin descargar)

| Dependencia | Versión | Dónde |
|---|---|---|
| PlatformIO Core | 6.1.x (fijar la exacta al instalarla, p. ej. `platformio==6.1.16`) | se instala en `.tools/` |
| Plataforma `espressif32` | **6.9.0** (Arduino-ESP32 2.0.17, ESP-IDF 4.4.7) | `platformio.ini` |
| `h2zero/NimBLE-Arduino` | **2.1.0** | `platformio.ini` (`lib_deps`) |
| Estándar C++ | `-std=gnu++17` (el core 2.x usa `gnu++11` por defecto) | `platformio.ini` |

> **No verificado todavía:** en E0 no se descargó nada, así que las versiones no se han resuelto contra el registro. En E2, la primera compilación debe confirmar que resuelven, que `NimBLECharacteristic::notify()` devuelve `bool` y que compila con `gnu++17`. Si alguna falla, se cambia **solo** la versión en `platformio.ini` y se documenta aquí.

**Disco:** todo queda dentro del proyecto, nunca en `C:`.
- `platformio.ini` fija `core_dir = ../.tools/platformio` (toolchains, plataformas, librerías y caché).
- La salida de compilación va a `device-firmware/.pio/`, ignorada por git.
- PlatformIO Core se instala en un entorno virtual dentro de `.tools/`, con `pip --no-cache-dir` y `TMP`/`TEMP` apuntando a `.tools/tmp` (`.tools/env.sh`).

### 6.3 Estructura

```
device-firmware/
├── platformio.ini          # sin [env] activo hasta conocer la placa
├── include/, src/          # código portable (lo compilan PC y PlatformIO; main.cpp excluido)
│   ├── ble_transport.h/.cpp   # ControlQueue + QueuedBleTransport (E1, probado en PC)
│   └── main.cpp               # demo de PC; #error con ARDUINO (no lo compila PlatformIO)
├── src/esp32/              # (E2) Esp32BleTransport + main_esp32.cpp (setup/loop)
└── test/host, test/tflm    # pruebas de PC con g++ (test_dir de PlatformIO: test/pio)
```

`Esp32BleTransport` (E2) hereda de `QueuedBleTransport` y solo implementa la parte que depende de NimBLE: `begin()` (servicio, 5 características, nombre en la respuesta al escaneo), `isConnected()`, `mtu()`, `notify()` y `setReadValue()`, y llama a `onControlWritten()` desde el callback `onWrite` de `CONTROL`. El nombre se obtiene con `deviceIdFromMac()` (cuidado: NimBLE entrega los bytes de la dirección al revés).

```cpp
void loop() {
  ble.service();               // comandos CONTROL, en el contexto de loop()
  pipeline.poll(millis());
}
```
