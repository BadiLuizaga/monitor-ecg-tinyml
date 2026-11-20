 <div align="center">

# ❤️ Monitor ECG & Arritmias

**Monitor de electrocardiograma y detección de arritmias con inteligencia artificial local (TinyML) y alertas por Bluetooth.**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Edge Impulse](https://img.shields.io/badge/Edge_Impulse-3B82F6?style=for-the-badge&logoColor=white)
![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)
![Flutter](https://img.shields.io/badge/Flutter-02569B?style=for-the-badge&logo=flutter&logoColor=white)
![Dart](https://img.shields.io/badge/Dart-0175C2?style=for-the-badge&logo=dart&logoColor=white)
![Bluetooth](https://img.shields.io/badge/Bluetooth_LE-0082FC?style=for-the-badge&logo=bluetooth&logoColor=white)

</div>

---

## 🧰 Stack

| Inteligencia Artificial | Procesamiento de datos | TinyML / Firmware | Comunicación | Aplicación móvil | Herramientas |
|:---:|:---:|:---:|:---:|:---:|:---:|
| TensorFlow + Keras | Python + NumPy + Pandas + SciPy | TensorFlow Lite Micro + C++ | Bluetooth Low Energy (BLE) + JSON | Flutter + Dart | VS Code + Git + GitHub |
| Edge Impulse | WFDB + MIT-BIH | Modelos `.tflite` | Protocolo BLE | FlutterBluePlus + Flutter TTS | Python 3.11 |

---


## 📂 Estructura y estado

| Carpeta | Contenido | Estado |
|---|---|---|
| `ai-model/` | Preprocesamiento, entrenamiento, evaluación y exportación TinyML | ✅ Ruta completa ejecutada con MIT-BIH (modelo experimental) |
| `communication/` | Protocolo BLE v1 y simulador del dispositivo | ✅ Probado (simulación, sin radio real) |
| `device-firmware/` | Lógica C++ para el ESP32 | ✅ Lógica probada en PC; port a ESP32 pendiente |
| `mobile-app/` | App Flutter | Responsabilidad de Eva (contrato en `shared/`) |
| `shared/` | Contrato app ↔ dispositivo y constantes | ✅ |
| `docs/` | Arquitectura, integración, BLE, modelo, requisitos | ✅ |

Empieza por [`docs/architecture.md`](docs/architecture.md) y [`docs/integration.md`](docs/integration.md).

```sh
source .tools/env.sh      # entorno local (ver docs/installation.md)
python -m pytest          # pruebas del modelo, protocolo y simulador
python communication/simulator/mock_ecg_device.py selftest
```

> ⚠️ **Proyecto académico.** No es un dispositivo médico ni una herramienta de diagnóstico clínico. Las clasificaciones y alertas son experimentales.
