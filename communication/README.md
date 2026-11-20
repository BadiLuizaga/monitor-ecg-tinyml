# communication/

Protocolo BLE entre el dispositivo ECG (ESP32) y la app móvil, junto con un simulador para probar el software sin hardware.

| Ruta | Contenido | Estado |
|---|---|---|
| `ble/ble_spec.md` | **Especificación normativa** del protocolo v1 | Completa (UUID provisionales) |
| `ble/uuid_config.md` | UUID y cómo cambiarlos | Provisionales |
| `protocol/protocol_config.json` | Fuente única de valores (UUID, cabecera, códigos) | — |
| `protocol/ecg_protocol.py` | Códec de referencia, receptor y política de reconexión | Probado |
| `protocol/message_schema.json` | JSON Schema de los mensajes | Verificado por las pruebas |
| `protocol/ecg_message.json` | Formato del paquete binario `ECG_DATA` | — |
| `protocol/alert_message.json` | Ejemplo de alerta | — |
| `examples/sample_messages.json` | Ejemplos válidos de todos los mensajes | Generados con el códec |
| `simulator/mock_ecg_device.py` | **Simulador** del dispositivo | Probado |
| `tests/` | Pruebas pytest | 89 pruebas en total en el repositorio |

## Simulador

> ⚠️ **Es una simulación.** No hay radio BLE ni ESP32. Produce los mismos bytes que definirá el firmware, pero los entrega por transportes de prueba: memoria, archivo o TCP en `127.0.0.1`. Las alertas salen de las etiquetas de la señal sintética, no de un modelo de IA, y llevan `"simulated": true`.

```sh
# Prueba completa en memoria (60 s simulados, con 5 % de paquetes perdidos)
python communication/simulator/mock_ecg_device.py selftest --duration 60 --drop 0.05

# Grabar una captura binaria reproducible
python communication/simulator/mock_ecg_device.py record --duration 30 --out communication/simulator/captures/demo.bin

# Servidor TCP local y cliente central de prueba (con reconexión por backoff)
python communication/simulator/mock_ecg_device.py serve-tcp --port 8765
python communication/simulator/mock_ecg_device.py client-tcp --port 8765 --duration 20

# Reproducir un registro MIT-BIH en lugar de la señal sintética
python communication/simulator/mock_ecg_device.py selftest --record ai-model/data/raw/mitdb/100
```

**Opciones principales:**

| Opción | Qué controla |
|---|---|
| `--seed` | Reproducibilidad |
| `--fs` | Frecuencia de muestreo **simulada** |
| `--hr` | Frecuencia cardiaca |
| `--pvc` | Probabilidad de latido ventricular prematuro |
| `--noise` | Ruido de la señal |
| `--mtu` | MTU negociado |
| `--drop`, `--corrupt` | Paquetes perdidos o corruptos |

Desde código, `FaultConfig` permite además simular desconexiones (`disconnect_at_s`) y tramos sin electrodo (`lead_off_intervals_s`).

**Formato de trama del transporte de prueba** (no forma parte del protocolo BLE): `u8 char_id | u16 longitud LE | payload`, con `char_id` 1 = `ECG_DATA`, 2 = `ALERT`, 3 = `STATUS`, 4 = `DEVICE_INFO` y 5 = `CONTROL`.

## Pruebas

```sh
python -m pytest communication/tests
```
