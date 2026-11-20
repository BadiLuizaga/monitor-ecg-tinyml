# Especificación del protocolo BLE — v1

> **Documento normativo.** Si otro documento contradice a este, manda este.
> Los valores concretos (UUID, códigos, tamaños) están en
> [`../protocol/protocol_config.json`](../protocol/protocol_config.json), que es la fuente única de verdad.
> La implementación de referencia es [`../protocol/ecg_protocol.py`](../protocol/ecg_protocol.py).
>
> **Estado:** implementado y probado en Python y en el simulador. El códec C++ del firmware (`device-firmware/src/ble_protocol.cpp`) está compilado y probado en PC: produce los mismos bytes que Python y su JSON valida con el códec. **No se ha probado sobre radio BLE real.**

## 1. Roles

| Rol | Quién | Función |
|---|---|---|
| Periférico / servidor GATT | ESP32 (o el simulador) | Anuncia, transmite ECG y alertas |
| Central / cliente GATT | App Flutter | Escanea, conecta, se suscribe y envía comandos |

## 2. Identificación del dispositivo

- **Nombre anunciado:** `ECG-TINYML-XXXX`, donde `XXXX` son los 4 últimos dígitos hexadecimales de la MAC. *(PROVISIONAL)*
- **Anuncio:** incluye el **Service UUID** para que la app filtre el escaneo por servicio y no por nombre.
- Tras conectar, la app **debe** leer `DEVICE_INFO` y usar sus valores: frecuencia de muestreo, escala y si los datos son simulados. **Nunca debe suponerlos.**

## 3. Servicio y características

UUID **PROVISIONALES**, generados aleatoriamente para desarrollo; ver [`uuid_config.md`](uuid_config.md).

| Característica | UUID | Propiedades | Codificación | Contenido |
|---|---|---|---|---|
| *Servicio* | `61860001-2b45-4f52-a512-0ba5fd3bf0a4` | — | — | — |
| `ECG_DATA` | `61860002-…` | notify | **binario** | Muestras ECG (§4) |
| `ALERT` | `61860003-…` | notify, indicate | JSON | Alertas y eventos (§5.3) |
| `STATUS` | `61860004-…` | read, notify | JSON | Estado, errores y respuesta a comandos (§5.2) |
| `DEVICE_INFO` | `61860005-…` | read | JSON | Identificación y parámetros del stream (§5.1) |
| `CONTROL` | `61860006-…` | write | JSON | Comandos de la app (§5.4) |

## 4. Paquete binario `ECG_DATA`

Enviar cada muestra como un objeto JSON multiplicaría el tráfico por unas 10. Por eso se usa un paquete binario compacto, en **little-endian**:

| Offset | Campo | Tipo | Descripción |
|---|---|---|---|
| 0 | `protocol_version` | u8 | `1` |
| 1 | `packet_type` | u8 | `1` = `ECG_SAMPLES` |
| 2 | `seq` | u16 | +1 por paquete (módulo 65536) |
| 4 | `first_sample_index` | u32 | Índice absoluto de la primera muestra (módulo 2³²) |
| 8 | `sample_count` | u8 | Número de muestras (1–255) |
| 9 | `flags` | u8 | Bit 0 `LEAD_OFF`, bit 1 `SOURCE_OVERFLOW`, bit 2 `SIMULATED`, bit 3 `SATURATED` |
| 10 | `samples` | int16[n] | Cuentas del ADC con signo, **sin convertir** |

- **Conversión a mV:** `mV = muestra × mv_per_count`, con `mv_per_count` tomado de `DEVICE_INFO`. Si vale `null` (sin calibrar), la app muestra la señal en unidades arbitrarias.
- **Muestras por paquete:** `⌊(MTU − 3 − 10) / 2⌋`, es decir, 5 con MTU 23, 86 con MTU 185 y 117 con MTU 247.
- **Ejemplo:** `010102010d0c0b0a03040100feff2c01` corresponde a seq = 258, índice = 0x0A0B0C0D, 3 muestras `[1, −2, 300]` y flag `SIMULATED`.
- **Caudal:** a 360 Hz × 2 bytes son unos 0,75 kB/s, más 10 bytes de cabecera por paquete. Muy por debajo de la capacidad de BLE 4.2/5.
- Si el ADC real tiene más de 16 bits, habrá que escalarlo a int16 en el firmware o definir un `sample_format` nuevo en una v2 del protocolo. **Pendiente de conocer el ADC.**

### Reglas del receptor (app)

1. Descartar el paquete si `len ≠ 10 + 2·sample_count`, si la versión no es 1 o si el tipo es desconocido.
2. Un salto en `seq` indica paquetes perdidos. El número exacto de muestras perdidas sale de `first_sample_index`.
3. Un `seq` anterior al esperado indica un paquete duplicado o viejo, y se descarta.
4. Al reconectar, reiniciar el receptor, porque el dispositivo puede reiniciar sus contadores.

El receptor de referencia es `EcgStreamReceiver` en `ecg_protocol.py`.

## 5. Mensajes JSON

- JSON compacto, en UTF-8.
- **Notificaciones y escrituras** (`ALERT`, `STATUS`, `CONTROL`): como máximo **180 bytes**, porque deben caber en un único paquete. Esto **exige MTU ≥ 185**.
  - En Android, la app debe pedirlo con `requestMtu(247)`.
  - En iOS se negocia automáticamente (185 o más).
- **`DEVICE_INFO`** (lectura): como máximo **512 bytes**, usando *long read*.
- El esquema completo está en [`../protocol/message_schema.json`](../protocol/message_schema.json) y hay ejemplos válidos en [`../examples/sample_messages.json`](../examples/sample_messages.json).

### 5.1 `device_info`
```json
{"v":1,"type":"device_info","device_id":"1A2B","fw_version":"0.1.0","hw_model":"TBD",
 "sample_rate_hz":360,"sample_format":"int16","adc_bits":null,"mv_per_count":null,"leads":1,"simulated":false}
```

### 5.2 `status` y `cmd_ack` (característica `STATUS`)
```json
{"v":1,"type":"status","state":"streaming","error_code":0,"error":"OK","uptime_ms":1520}
{"v":1,"type":"cmd_ack","cmd_id":1,"ok":true,"error_code":0}
```
Estados posibles: `booting` → `advertising` → `connected` ⇄ `streaming`; además, `error`.

### 5.3 `alert`
```json
{"v":1,"type":"alert","event_id":42,"event":"arrhythmia_suspected","beat_class":"V",
 "confidence":0.91,"sample_index":123456,"simulated":false}
```

| `event` | Cuándo se envía |
|---|---|
| `arrhythmia_suspected` | El modelo clasificó un latido de una clase que puede generar alertas |
| `lead_off` | Electrodo desconectado, si el hardware lo detecta |
| `signal_quality` | Reservado: señal saturada o demasiado ruidosa |
| `device_error` | Reservado |

- `event_id` es creciente y permite descartar duplicados.
- `sample_index` permite situar la alerta sobre el trazado. Apunta al pico R del latido clasificado.
- `"simulated": true` indica que la alerta no procede del modelo real.
- **Momento de envío (modelo v2, con RR):**
  - un latido se clasifica cuando llega el siguiente, o 3 s después si no llega ninguno, así que `arrhythmia_suspected` llega con **un latido de retraso**;
  - no se envía durante el calentamiento RR (unos 30 s tras conectar, tras un electrodo suelto o tras una pausa de más de 3 s).

  El formato del mensaje no cambia.

### 5.4 Comandos (`CONTROL`)
```json
{"v":1,"type":"cmd","cmd":"start_stream","cmd_id":1}
{"v":1,"type":"cmd","cmd":"ack_alert","cmd_id":2,"event_id":42}
```
Comandos disponibles: `start_stream`, `stop_stream`, `ping`, `get_status` y `ack_alert`.

Cada comando recibe un `cmd_ack` por `STATUS`. Si el comando no pudo interpretarse, el acuse lleva `cmd_id = -1`.

## 6. Códigos de error

| Código | Nombre | Significado |
|---|---|---|
| 0 | `OK` | |
| 10 | `LEAD_OFF` | Electrodo desconectado |
| 11 | `SIGNAL_SATURATED` | Señal saturada |
| 12 | `SIGNAL_NOISY` | Señal con demasiado ruido |
| 20 | `SAMPLE_BUFFER_OVERFLOW` | Se perdieron muestras en la adquisición |
| 21 | `BLE_TX_QUEUE_FULL` | La cola de transmisión BLE está llena |
| 30 | `MODEL_NOT_LOADED` | El ECG se transmite, pero **sin clasificación** |
| 31 | `INFERENCE_FAILED` | Falló la inferencia |
| 32 | `CONFIG_MISMATCH` | La frecuencia del sensor no coincide con la de los filtros o el modelo |
| 40 | `INVALID_COMMAND` | Comando inválido |
| 41 | `UNSUPPORTED_PROTOCOL_VERSION` | Versión del protocolo no soportada |
| 42 | `MTU_TOO_SMALL` | Se rechaza `start_stream` porque el MTU es menor que 185 |
| 50 | `LOW_BATTERY` | Batería baja, solo si el hardware lo permite |

## 7. Conexión, desconexión y reconexión

```
App: escanear (filtro por Service UUID) → conectar → requestMtu(247) → descubrir servicios
     → leer DEVICE_INFO → suscribirse a ECG_DATA, ALERT y STATUS → escribir start_stream
     → recibir datos … → stop_stream / desconectar
```

- **Pérdida de enlace:** la app la declara si pasan **3 s sin datos** mientras hay stream (`LinkMonitor`), o cuando la pila BLE notifica la desconexión.
- **Reconexión:** con *backoff* exponencial de 1, 2, 4, 8 y 16 s, hasta un máximo de 30 s por espera y 10 intentos (`ReconnectPolicy`). Después, aviso al usuario. *(PROVISIONAL)*
- **Del lado del dispositivo:** al perder la conexión deja de transmitir y vuelve a `advertising`. La adquisición y la detección siguen funcionando.
- Tras reconectar, la app repite el flujo completo: leer `DEVICE_INFO`, suscribirse y enviar `start_stream`.

## 8. Versionado

Cualquier cambio incompatible incrementa `protocol_version`. La app debe rechazar versiones desconocidas, y el dispositivo responde `UNSUPPORTED_PROTOCOL_VERSION`.
