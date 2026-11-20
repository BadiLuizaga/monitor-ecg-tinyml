# Requisitos

> Uso académico. El sistema **no** es un dispositivo médico, no está validado clínicamente y no debe usarse para tomar decisiones de salud.

## Requisitos funcionales

| ID | Requisito | Estado |
|---|---|---|
| RF-01 | Adquirir ECG de una derivación a una frecuencia fija | ❌ Pendiente del hardware (simulado) |
| RF-02 | Filtrar la señal (deriva de línea base y ruido de alta frecuencia; red eléctrica opcional) | ✅ Python · ✅ C++ (pruebas en PC) |
| RF-03 | Detectar picos R en tiempo real | ✅ Python (Se 99,2 %, VPP 99,1 % en DS2) · ✅ C++ equivalente (pruebas en PC) |
| RF-04 | Clasificar cada latido en el dispositivo (N/S/V/F) | 🔶 v1 exportado (S y F no fiables); v2 con RR implementado y probado (P1–P5, C1–C3), **pendiente de entrenar**; TFLM sin integrar |
| RF-05 | Transmitir el ECG por BLE a la app | ✅ Protocolo y simulador · ❌ radio real |
| RF-06 | Enviar alertas de latidos anómalos y de electrodo suelto | ✅ Protocolo y simulador · ✍️ firmware |
| RF-07 | Recibir comandos desde la app (iniciar o detener el stream, confirmar alertas) | ✅ Protocolo · ✍️ firmware |
| RF-08 | Informar estado y errores | ✅ |
| RF-09 | Reconectar automáticamente tras una pérdida de enlace | ✅ Política definida · ❌ prueba real |
| RF-10 | Mostrar el ECG y las alertas en la app | Responsabilidad de Eva |

## Requisitos no funcionales

| ID | Requisito | Valor objetivo | Estado |
|---|---|---|---|
| RNF-01 | Inferencia local (sin nube) | TFLM en el ESP32 | 🔶 Pendiente de integrar |
| RNF-02 | Tamaño del modelo | < 100 KB de flash | ✅ 20,8 KB (int8) |
| RNF-03 | RAM de inferencia (arena) | Por definir según el modelo de ESP32 | ❌ **Sin medir** |
| RNF-04 | Latencia por latido | ≪ 300 ms (RR mínimo) | ❌ **Sin medir** |
| RNF-05 | Pérdida de paquetes BLE | < 0,1 % a 1 m | ❌ Sin medir (el simulador la detecta exactamente) |
| RNF-06 | Detección de pérdidas | Exacta, por número de muestra | ✅ |
| RNF-07 | Reproducibilidad | Semillas fijas; datos de prueba deterministas | ✅ |
| RNF-08 | Trazabilidad entrenamiento ↔ firmware | Coeficientes y parámetros generados, no copiados | ✅ |
| RNF-09 | Separación lógica ↔ hardware | Configuración física aislada en `hardware_config.h` | ✅ |
| RNF-10 | Datos simulados identificables | Flag `SIMULATED` y `simulated: true` | ✅ |

## Requisitos del entorno de desarrollo

| Herramienta | Versión | Uso |
|---|---|---|
| Python | 3.11 | Modelo, simulador y pruebas |
| numpy, scipy, pandas, wfdb | ver `ai-model/requirements.txt` | Señal y datos |
| tensorflow-cpu | ≥ 2.16 (probado con 2.21) | Entrenamiento y conversión TFLite |
| pytest | ≥ 8 | Pruebas |
| g++/clang++ C++17 | probado con g++ 16.2 (w64devkit 2.11.0 en `.tools/`) | Pruebas en PC del firmware |
| Toolchain ESP32 | Por decidir | Firmware |
| Flutter / Dart | 3.47 / 3.13 | App |

Instalación: ver [installation.md](installation.md).
