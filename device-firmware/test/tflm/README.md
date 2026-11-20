# Prueba T1: TensorFlow Lite Micro en PC (modelo v1)

> Prueba de compatibilidad del modelo con TFLM en un PC. **No es una prueba en el ESP32 ni una validación clínica.**

## Qué hace
`run_tflm_test.sh` ejecuta, en este orden:
1. `build_tflm_lib.sh` compila TFLM como librería estática, `.tools/tflm/build/libtflm.a`.
   - Incluye el núcleo y los kernels de referencia: las mismas 148 fuentes que lista su CMake, sin kernels optimizados, sin `signal` y sin pruebas.
   - Se compila con g++ de w64devkit. El propio CMake no hace falta.
2. `make_vectors.py` genera los vectores de referencia en `vectors/`:
   - 1000 latidos de DS1 (250 por clase) y 7 casos límite. No se usa DS2.
   - Para cada uno: la entrada int8 cuantizada en Python y las salidas del intérprete TFLite de Python, con kernels de referencia (`BUILTIN_REF`) y con los kernels por defecto.
   - Comprueba que `src/generated/model_data.cc` contiene exactamente los bytes de `ai-model/models/ecg_model_int8.tflite`.
3. `test_tflm.cpp` (ejecutable en `bin/`) comprueba:
   - entradas, salidas, cuantización y operadores;
   - que la arena prevista (`ECG_TFLM_ARENA_BYTES`) es suficiente;
   - la cuantización con `ecg::quantizeInt8` del firmware;
   - que la salida int8 coincide con la de Python;
   - el motor real del firmware (`ecg::TflmInferenceEngine`, `src/inference_tflm.cpp`) con la arena de 16 KB, comparado con el intérprete de la prueba y con Python.

**Diagnóstico capa a capa**, que no forma parte de la prueba:
- `diag_tflm.cpp` (en `bin/diag_tflm`) vuelca la salida de cada nodo con `preserve_all_tensors`.
- `diag_layers.py` la compara con TFLite (`experimental_preserve_all_tensors`), usando la misma entrada int8.
- Resultados en `vectors/layer_diag*.json`.

Uso, desde la raíz y con w64devkit en el `PATH`:
```sh
PYTHON=ai-model/.venv/Scripts/python.exe sh device-firmware/test/tflm/run_tflm_test.sh
```

## Dependencias, todas dentro de `.tools/tflm/` (fuera de git)

| Dependencia | Fuente | Licencia | Integridad |
|---|---|---|---|
| TFLM, commit `b8d75cbf6b8009762bc1e3ee4817bc33e5dcee47` (2026-11-11, CI en verde) | `codeload.github.com/tensorflow/tflite-micro/zip/<commit>` | Apache-2.0 | SHA256 del ZIP: `0c6b032dab6a940f134cc40c32ccc8c47ff49f8af23468ad6837f722294afe11`. Comentario del ZIP = commit. Los 1809 archivos coinciden con los hashes git del árbol del commit (API de GitHub) |
| flatbuffers v25.9.23 (solo cabeceras) | `github.com/google/flatbuffers/archive/refs/tags/v25.9.23.zip`, la versión que fija TFLM | Apache-2.0 | MD5 `023eca1e211d64007124420cd6be29c7`, igual al que fija TFLM. SHA256 `083a424984986ba7fcd1635d54417d4523a3c729e137575db2421e996a1fb816` |

No se descargaron kissfft (solo la usa `signal`) ni eyalroz_printf (TFLM usa `vfprintf` en PC).

## Criterio y resultado (2026-11-19)

**Criterio**, fijado antes de ejecutar y coherente con `docs/integration.md`:
- 0 fallos de inferencia;
- entrada int8 idéntica a la de Python;
- salida a ±1 LSB de TFLite REF;
- misma clase en el 100 % de los latidos.

**Resultado:**

| Comprobación | Resultado |
|---|---|
| Compilación de TFLM con g++ 16.2 (w64devkit) | ✅ 148 fuentes, 0 avisos, 4,5 min |
| Operadores | ✅ CONV_2D, EXPAND_DIMS, FULLY_CONNECTED, MAX_POOL_2D, MEAN, RESHAPE y SOFTMAX (22 nodos), todos registrados |
| Entrada / salida | ✅ int8 [1,252,1], escala 0,0461284779, zp −18 / int8 [1,4], escala 1/256, zp −128. Coinciden con `model_config.h` |
| Arena | ✅ 12 928 bytes usados; los 16 KB previstos bastan, aunque en el ESP32 puede variar |
| Inferencias | ✅ 1007, ninguna fallida |
| Entrada int8 C++ = Python | ✅ 1007/1007 |
| Misma clase que TFLite REF | ✅ 1007/1007 (3 empates) |
| Salida idéntica a TFLite REF | 995/1007 |
| Diferencia máxima con TFLite REF | ❌ **3 LSB**, en una única salida (latido 536, clase V), por encima del criterio de 1 LSB. Las otras 3 salidas de ese latido difieren en 1 LSB. En los otros 11 latidos afectados, todas las diferencias son de 1 LSB |
| Distribución \|TFLM − REF\| (4028 salidas = 1007 latidos × 4 clases; fuente: `vectors/tflm_outputs.csv`) | **4004 salidas idénticas (0 LSB)**, 23 salidas con 1 LSB de diferencia y 1 salida con 3 LSB. Ninguna con 2 LSB ni con más de 3 |
| Dentro / fuera de ±1 LSB | 4027 salidas dentro, **1 salida fuera** (latido 536). Por latidos: 1006/1007 tienen las 4 salidas dentro de ±1 LSB |
| Latidos afectados | 12 latidos con alguna salida distinta; solo 1 (el 536) supera ±1 LSB. Ninguno cambia de clase |
| Comparación con TFLite por defecto | Idéntica en 357/1007, máximo 7 LSB, misma clase en 1006/1007. Estos kernels no son la referencia |
| `inference_tflm.cpp` | ✅ Compila con `-Wall -Wextra -Werror`, tras las correcciones de compatibilidad de la sección siguiente |
| Motor real (`TflmInferenceEngine`) | ✅ `begin()` correcto. 12 928 de 16 384 bytes de arena. 1007 inferencias, ninguna fallida. Clase y confianza idénticas al intérprete de la prueba en 1007/1007. Misma clase que REF en 1007/1007 |

**Veredicto: NO APROBADA** según el criterio fijado (±1 LSB). La clase predicha coincide en el 100 % de los latidos.

## Diagnóstico de las discrepancias (2026-11-19)

Se comparó cada nodo con la **misma entrada int8**, de modo que la cuantización de la entrada queda descartada como causa.

1. **Del nodo 0 al 18 son idénticos bit a bit** a TFLite REF en los 1007 latidos. Eso incluye CONV_2D, MAX_POOL_2D y **MEAN**.
2. **El primer operador que difiere es FULLY_CONNECTED**: el nodo 19 en 34 latidos y el nodo 20 en 6, siempre con 1 LSB de diferencia.
   - Sus pesos son int8 cuantizados por canal (zp 0) y su bias es int32.
   - Las escalas, los puntos cero y los multiplicadores por canal son los mismos en los dos intérpretes.
3. **SOFTMAX no introduce error.** Amplifica la diferencia de 1 LSB a su entrada hasta 3 LSB a su salida, en el latido 536.
4. **Todos los elementos que difieren tienen un valor exacto (acumulador × multiplicador) a menos de 0,002 del empate**, entre x,4980 y x,4998.
   - TFLite REF redondea correctamente: coincide al 100 % con el redondeo exacto.
   - TFLM sube 1 LSB en los 35 casos del nodo 19.
5. **Emulación de la recuantización de FULLY_CONNECTED:**
   - TFLM coincide al 100 % con el `MultiplyByQuantizedMultiplier` de **doble redondeo** de gemmlowp (`SaturatingRoundingDoublingHighMul` + `RoundingDivideByPOT`). Es lo que se usa cuando `TFLITE_SINGLE_ROUNDING` no está definido, el valor por defecto en TFLM y en su CMake.
   - TFLite REF coincide al 100 % con el redondeo simple, y también con la recuantización en float.
6. **Variante de diagnóstico con `-DTFLITE_SINGLE_ROUNDING=1`** (`.tools/tflm/build_single_rounding`, no adoptada): empeora el resultado.
   - Difieren CONV_2D y MEAN; la diferencia llega a 7 LSB y la clase cambia en 3 latidos.
   - Es decir, TFLite REF usa doble redondeo en CONV y MEAN, igual que TFLM, pero no en FULLY_CONNECTED por canal.

**Causa:** la diferencia está en el **kernel**: la recuantización de FULLY_CONNECTED por canal de TFLM usa doble redondeo, y la de TFLite REF no. No se debe a la cuantización del modelo ni de la entrada. Tampoco es un error de configuración del intérprete: no hay ninguna opción global que iguale los dos.

## Decisión final sobre T1 (2026-11-19)
1. **T1 queda NO APROBADA** con el criterio original de ±1 LSB, que no se modifica.
2. **La concordancia de clases de 1007/1007 no equivale a aprobar la concordancia numérica.** Se informa por separado.
3. **No se parchea FULLY_CONNECTED** ni ningún otro kernel de TFLM o de bibliotecas de terceros.
4. Se conservan:
   - el diagnóstico numérico: `diag_tflm.cpp`, `diag_layers.py` y `vectors/layer_diag*.json`;
   - los resultados de las 1007 inferencias: `vectors/tflm_outputs*.csv`;
   - la causa: el redondeo de FULLY_CONNECTED, amplificado por SOFTMAX.
5. Se mantienen las correcciones de compatibilidad, que están verificadas:
   - `inference_tflm.cpp` compila sin avisos y el motor real reproduce al intérprete de la prueba en 1007/1007;
   - las cabeceras v1 regeneradas conservan los mismos bytes del modelo;
   - las pruebas host del firmware y las de Python pasan.
6. **Siguiente validación (pendiente de autorización y del hardware):** ejecutar el modelo v1 en el ESP32 real con estos mismos vectores de referencia, y registrar:
   - las diferencias de salida en LSB frente a TFLite REF y frente a TFLM en PC;
   - la concordancia de clases;
   - la latencia por inferencia;
   - la memoria (`arena_used_bytes()`, flash y RAM).

   Los kernels del ESP32 (por ejemplo, ESP-NN en esp-tflite-micro) pueden redondear de otra forma, así que el resultado en PC no se traslada.
7. Los volcados de `vectors/` (unos 85 MB, ignorados por git) **se conservan** hasta que se autorice limpiarlos.

## Correcciones de compatibilidad aplicadas (autorizadas)
1. **Ruta del esquema** en `src/inference_tflm.cpp`: usa `tensorflow/lite/micro/schema/schema_generated.h` si existe y, si no, la ruta antigua (`__has_include`).
2. **Cabeceras v1 regeneradas** con las funciones del exportador actual a partir de los bytes del `.tflite` v1 existente, **sin reconvertir el modelo**.
   - `model_data.cc` tiene los mismos bytes.
   - `model_config.h` solo añade `ECG_MODEL_BEAT_INPUT_INDEX 0`, que es la posición real de la entrada `beat` identificada por su nombre.
   - En los demás archivos solo cambia la fecha.
   - Copia de las cabeceras anteriores en `.tools/tmp/t1diag/generated_before/`.
3. No se cambiaron ni los pesos, ni la arquitectura, ni la lógica del firmware. Las pruebas host del firmware y las 123 de Python siguen pasando.
