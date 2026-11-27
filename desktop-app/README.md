# Interfaz de laptop (Python) — CardioAlert

Receptor BLE v1 con GUI (PySide6 + pyqtgraph). Reutiliza `communication/protocol/ecg_protocol.py` sin modificarlo.

```sh
# desde la raíz del repo; el entorno vive en .tools/desktop-venv (E:)
cd desktop-app
..\.tools\desktop-venv\Scripts\python.exe -m ecg_desktop            # GUI
..\.tools\desktop-venv\Scripts\python.exe tools\ble_probe.py        # sonda por consola contra el ESP32
..\.tools\desktop-venv\Scripts\python.exe -m pytest tests           # pruebas del análisis (BPM)
..\.tools\desktop-venv\Scripts\python.exe -m ecg_desktop --selftest-ui captura.png   # render con datos SINTÉTICOS locales
```

Requisito: Bluetooth de Windows **encendido**.

## Qué muestra y de dónde sale cada dato
| Elemento | Origen |
|---|---|
| ECG en tiempo real | `ECG_DATA` (paquetes binarios) |
| BPM | **Calculado en el PC** desde la señal (el protocolo v1 no trae FC) |
| Clasificación N/S/V/F | Solo **alertas** `arrhythmia_suspected` (`beat_class`). Sin modelo (`MODEL_NOT_LOADED`) no hay clasificación y la interfaz lo dice |
| Simulado / real | `DEVICE_INFO.simulated` **o** flag `SIMULATED` de los paquetes (basta uno para marcar «simulado») |
| Estado / errores | `STATUS` |

Reconexión con *backoff* (1, 2, 4, 8, 16 s; 10 intentos) y aviso de enlace perdido tras 3 s sin datos, según `ble_spec.md` §7.
