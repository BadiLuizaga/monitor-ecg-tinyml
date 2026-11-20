# Configuración de UUID

> ⚠️ **PROVISIONALES.** Se generaron aleatoriamente (UUID v4) solo para desarrollo. Hay que sustituirlos por los definitivos antes de la entrega.

| Elemento | UUID |
|---|---|
| Servicio ECG | `61860001-2b45-4f52-a512-0ba5fd3bf0a4` |
| `ECG_DATA` | `61860002-2b45-4f52-a512-0ba5fd3bf0a4` |
| `ALERT` | `61860003-2b45-4f52-a512-0ba5fd3bf0a4` |
| `STATUS` | `61860004-2b45-4f52-a512-0ba5fd3bf0a4` |
| `DEVICE_INFO` | `61860005-2b45-4f52-a512-0ba5fd3bf0a4` |
| `CONTROL` | `61860006-2b45-4f52-a512-0ba5fd3bf0a4` |

Todos comparten una base y solo cambia el segundo bloque de 4 dígitos (`0001`–`0006`). Así es fácil reconocerlos en herramientas como nRF Connect.

## Cómo cambiarlos

Los UUID aparecen en dos sitios, y una prueba automática falla si no coinciden:

1. `communication/protocol/protocol_config.json`, que es la **fuente única**.
2. `device-firmware/include/integration_config.h`, con los `#define ECG_*_UUID`.

La app de Eva debería leerlos de una constante generada a partir del JSON, no copiarlos a mano.

Para comprobarlo:

```sh
python -m pytest communication/tests/test_protocol.py -k firmware_header
```
