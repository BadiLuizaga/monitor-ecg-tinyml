# Documentación Bluetooth Low Energy

Resumen del protocolo. **La especificación normativa es [`communication/ble/ble_spec.md`](../communication/ble/ble_spec.md)**; este documento explica las decisiones tomadas y su estado.

## Decisiones

| Decisión | Motivo |
|---|---|
| Muestras en **binario** (cabecera de 10 B + int16 LE) | Un JSON por muestra (`{"v":123}`) ocupa unas 10 veces más y consume CPU del ESP32 serializando |
| JSON para metadatos, estado, alertas y comandos | Mensajes poco frecuentes; legibles y fáciles de depurar y extender |
| `seq` + `first_sample_index` | Detectar pérdidas y calcular exactamente cuántas muestras faltan, aunque cambie el MTU |
| Muestras en **cuentas del ADC** sin convertir | La escala real se desconoce; `DEVICE_INFO.mv_per_count` permitirá convertir cuando se calibre |
| MTU ≥ 185 obligatorio para el stream | Cada alerta JSON (≤ 180 B) cabe en una sola notificación, sin fragmentación |
| `DEVICE_INFO` como lectura | Admite hasta 512 B con *long read* y solo se lee una vez |
| Comandos con `cmd_id` y acuse (`cmd_ack`) | La app sabe si cada orden se aplicó y por qué falló |
| Backoff exponencial en la reconexión | No saturar la radio ni la batería si el dispositivo está lejos o apagado |
| **NimBLE** como pila BLE, con PlatformIO + Arduino (E0) | Menos RAM que Bluedroid; disponible en ESP32, S3 y C3; ver `docs/integration.md` §6 |
| Comandos `CONTROL` por una **cola SPSC** procesada en `loop()` (E1) | La pila BLE entrega las escrituras en su propia tarea y el pipeline no es thread-safe: el callback solo copia y `service()` las entrega |

## Estado

| Elemento | Estado |
|---|---|
| Especificación v1 | ✅ Completa |
| UUID | ⚠️ **PROVISIONALES** (aleatorios) |
| Códec Python, receptor y reconexión | ✅ Probados (pytest) |
| Simulador (memoria, archivo, TCP) | ✅ Probado. **No es BLE real** |
| Códec C++ del firmware | ✅ Probado en PC (bytes y JSON idénticos a Python) |
| Base del transporte (`QueuedBleTransport`, cola de `CONTROL`) | ✅ Probada en PC |
| Transporte BLE del ESP32 (`Esp32BleTransport`, NimBLE) | ❌ Pendiente (E2) |
| Prueba sobre radio real (ESP32 ↔ teléfono) | ❌ Pendiente del hardware |

## Pruebas cuando llegue el ESP32

1. **Descubrimiento:** con nRF Connect, comprobar el nombre `ECG-TINYML-XXXX`, el Service UUID y las 5 características con sus propiedades.
2. **MTU:** comprobar el MTU negociado con Android y con iOS. Con MTU 23 debe responder `MTU_TOO_SMALL`.
3. **Integridad:** stream de 10 min con la `TestPatternSource` del firmware. Pérdida de paquetes según `seq` menor al 0,1 % a 1 m, y medir también a 5 m y 10 m.
4. **Caudal y latencia:** tiempo desde que se toma la muestra hasta que la app la recibe, y desde el latido hasta la alerta.
5. **Reconexión:** apagar y encender el dispositivo y alejarse hasta perder el enlace. La app debe reconectar con backoff y reiniciar el receptor.
6. **Comandos:** todos los comandos, más comandos inválidos y una versión de protocolo distinta.
7. **Consumo:** corriente con stream activo y en reposo, si el dispositivo funciona con batería.
