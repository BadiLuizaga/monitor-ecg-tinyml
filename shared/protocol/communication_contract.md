# Contrato de integración: dispositivo ECG ↔ app móvil

> Dirigido a la app Flutter (`mobile-app/`, responsabilidad de Eva). Resume **qué debe hacer la app** para consumir el protocolo v1. El detalle normativo está en [`communication/ble/ble_spec.md`](../../communication/ble/ble_spec.md).
>
> Este contrato **no modifica** `mobile-app/`. Es la interfaz acordada para que ambos lados puedan avanzar en paralelo.

## Lo que la app recibe

| Característica | Tipo | La app debe… |
|---|---|---|
| `DEVICE_INFO` (read) | JSON | Leerla al conectar. Usar `sample_rate_hz` para el eje de tiempo, `mv_per_count` para la escala (si es `null`, mostrar unidades arbitrarias) y `simulated` para mostrar un aviso de datos simulados |
| `ECG_DATA` (notify) | **Binario** | Decodificar la cabecera de 10 bytes y las muestras int16 LE. Detectar huecos con `seq` y `first_sample_index` |
| `ALERT` (notify) | JSON | Mostrar la alerta, descartar los `event_id` repetidos y responder con `ack_alert` |
| `STATUS` (read/notify) | JSON | Mostrar el estado (`streaming`, `LEAD_OFF`…) y emparejar cada `cmd_ack` con su `cmd_id` |

## Lo que la app envía (`CONTROL`, write)

```json
{"v":1,"type":"cmd","cmd":"start_stream","cmd_id":1}
```
Comandos: `start_stream`, `stop_stream`, `ping`, `get_status` y `ack_alert` (este último con `event_id`). `cmd_id` lo genera la app y debe ser creciente.

## Requisitos obligatorios para la app

1. **Filtrar el escaneo por Service UUID** `61860001-2b45-4f52-a512-0ba5fd3bf0a4` (PROVISIONAL).
2. **Negociar MTU ≥ 185** antes de `start_stream`. En Android, con `requestMtu(247)`. Si no se negocia, el dispositivo responde `MTU_TOO_SMALL` (42).
3. **No suponer** frecuencia de muestreo ni escala: leerlas siempre de `DEVICE_INFO`.
4. **Reconexión:** declarar el enlace perdido tras 3 s sin datos y reintentar con *backoff* de 1, 2, 4, 8 y 16 s (máximo 30 s, 10 intentos). Al reconectar, reiniciar el receptor y repetir el flujo de conexión.
5. **Mostrar con claridad** que las alertas son **orientativas** y que el sistema **no es un dispositivo médico**.
6. Si `simulated` es `true` (en `DEVICE_INFO` o en una alerta), indicar que los datos son simulados.

## Decodificación de `ECG_DATA` (pseudocódigo Dart)

```dart
// bytes: Uint8List recibido en la notificación
final bd = ByteData.sublistView(bytes);
if (bytes.length < 10) return;                     // inválido
final version = bd.getUint8(0);                    // == 1
final type    = bd.getUint8(1);                    // == 1
final seq     = bd.getUint16(2, Endian.little);
final first   = bd.getUint32(4, Endian.little);
final count   = bd.getUint8(8);
final flags   = bd.getUint8(9);
if (version != 1 || type != 1 || bytes.length != 10 + 2 * count) return;
final samples = List<int>.generate(count, (i) => bd.getInt16(10 + 2 * i, Endian.little));
final leadOff   = flags & 0x01 != 0;
final simulated = flags & 0x04 != 0;
```

## Cómo probar la app sin hardware

- `communication/examples/sample_messages.json` contiene mensajes y paquetes válidos, con su representación hexadecimal. Sirven para pruebas unitarias del decodificador en Dart.
- `communication/simulator/mock_ecg_device.py record` genera capturas binarias reproducibles con alertas, pérdidas y lead-off.
- El modo `serve-tcp` del simulador emite el mismo flujo por TCP local. **No es BLE:** para usarlo desde la app haría falta un adaptador de depuración, que es decisión de Eva.
- La prueba con radio BLE real queda pendiente del ESP32.

## Cuándo llegan las alertas (modelo v2, con intervalos RR)

El formato de los mensajes **no cambia** (sigue siendo la versión 1 del protocolo). Cambia el momento en que llegan:

- Cada latido se clasifica **cuando llega el siguiente**, porque el modelo necesita su RR siguiente. Una alerta llega, por tanto, aproximadamente **un latido más tarde** (unos 0,4–1,5 s). Si después viene una pausa, llega a los **3 s**.
- `sample_index` sigue apuntando al pico R del latido que originó la alerta. La app puede situarla en el trazado sin hacer ningún cálculo.
- **No hay alertas de arritmia durante unos 30 s** tras conectar, tras un electrodo suelto o tras una pausa de más de 3 s (calentamiento RR). El ECG sí se transmite durante ese tiempo. Se recomienda que la app lo indique, por ejemplo con un mensaje de "analizando ritmo".

## Valores PROVISIONALES que pueden cambiar

UUID, prefijo del nombre (`ECG-TINYML`), frecuencia de muestreo, `mv_per_count`, `adc_bits`, parámetros de reconexión y clases que pueden generar alertas (de momento solo `V`). Los cambios se anunciarán incrementando la versión del protocolo cuando sean incompatibles.
