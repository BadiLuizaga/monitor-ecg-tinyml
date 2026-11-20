# Instalación del entorno (Python / modelo / simulador)

Todo se instala **dentro de la carpeta del proyecto**, sin tocar el disco del sistema.

## Windows (como está configurado ahora)

- Python 3.11 portable: `.tools/python311/`, extraído del paquete oficial de python.org en formato NuGet, sin instalador ni registro.
- Entorno virtual: `ai-model/.venv/` (unos 2 GB, sobre todo TensorFlow).
- Temporales y cachés: `.tools/tmp/`.
- `.tools/env.sh` dirige `TEMP`, `TMP`, `PIP_CACHE_DIR` y `KERAS_HOME` a `.tools/` y desactiva la caché de pip.

```sh
# Git Bash, desde la raíz del repositorio
source .tools/env.sh            # activa el venv y redirige temporales/cachés a .tools/
python -m pytest                # todas las pruebas
```

**Recrear desde cero:**

```sh
.tools/python311/python.exe -m venv ai-model/.venv
source .tools/env.sh
python -m pip install --no-cache-dir -r ai-model/requirements.txt
```

> **Red con inspección TLS** (antivirus o proxy): si pip falla con `CERTIFICATE_VERIFY_FAILED`, usar pip ≥ 24.2, que valida con los certificados de Windows (*truststore*). **No desactivar la verificación** con `--trusted-host`. La descarga de MIT-BIH usa `truststore` por el mismo motivo.

## Linux / macOS / CI

```sh
python3.11 -m venv ai-model/.venv && source ai-model/.venv/bin/activate
pip install -r ai-model/requirements.txt
python -m pytest
```

## Espacio en disco

| Elemento | Tamaño aprox. | Se puede borrar |
|---|---|---|
| `ai-model/.venv` | 2 GB | Sí; se recrea con `pip install` |
| `.tools/python311` | 50 MB | Solo si se deja de usar Python |
| `ai-model/data/raw/mitdb` | 90 MB | Sí; se vuelve a descargar con `dataset_loader --download` |
| `ai-model/models` | < 1 MB | Sí; se regenera entrenando |
| `.tools/tmp` | variable | Sí, siempre |
