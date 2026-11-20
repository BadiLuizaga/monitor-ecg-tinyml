# Documentación del modelo de IA

> **Modelo experimental, de uso académico.** No es un dispositivo médico, no está validado clínicamente y **no está listo para producción**. La memoria (arena TFLM) y la latencia en el ESP32 **no se han medido**.

> **Estado de las versiones:**
> - **v1 (secciones 1–7):** solo la morfología del latido. Es el modelo entrenado y exportado actualmente al firmware.
> - **v2 (sección 8):** latido + intervalos RR. Las características, el pipeline y las pruebas están implementados y verificados. La validación cruzada **no cumplió los criterios**, ni en la iteración 1 (sección 8.8) ni en la iteración 2, que fue la última (sección 8.10). Por eso no hay modelo final v2 ni evaluación de DS2, y el firmware sigue con la v1.

## 1. Datos

- **MIT-BIH Arrhythmia Database**: 48 registros de 30 min, 360 Hz y 11 bits; se usa la derivación MLII.
- **Clases AAMI EC57**:
  - N: normal, bloqueos de rama, escapes;
  - S: supraventricular;
  - V: ventricular;
  - F: fusión.

  Q queda excluida porque casi solo aparece en los registros con marcapasos.
- **División inter-paciente** (de Chazal, 2004): DS1 para entrenar (22 registros) y DS2 para la prueba (22 registros de **otros pacientes**). Se excluyen los registros con marcapasos (102, 104, 107 y 217).
- **Validación para la parada temprana:** 4 registros completos de DS1, elegidos con semilla 42: 122, 208, 209 y 230.

| Conjunto | N | S | V | F |
|---|---|---|---|---|
| Entrenamiento (18 registros de DS1; N limitado a 15 000) | 36 915 | 559 | 2 794 | **42** |
| Validación (4 registros de DS1) | 8 934 | 385 | 994 | 372 |
| **Prueba DS2** | 44 240 | 1 837 | 3 220 | 388 |

## 2. Preprocesamiento (idéntico en Python y en el firmware)

1. Remuestreo a `fs_hz` (360 Hz; **provisional hasta conocer el sensor**).
2. Butterworth paso banda de 0,5–40 Hz, orden 2, **causal** (el mismo filtro que ejecuta el ESP32). Notch de 50/60 Hz opcional.
3. Pico R:
   - en entrenamiento, la anotación ajustada al máximo de la señal filtrada (±50 ms);
   - en el dispositivo, el detector Pan-Tompkins simplificado.
4. Ventana de 0,25 s antes y 0,45 s después del R, es decir, **252 muestras**.
5. z-score por latido, que hace al modelo **independiente de la ganancia del ADC**.

## 3. Arquitectura

CNN 1D con **7 252 parámetros**:

- 3 bloques de Conv1D (16, 32 y 32 filtros; kernels de 7, 5 y 3) + BatchNorm + ReLU + MaxPool(2);
- después, GlobalAveragePooling, Dense(32) y Softmax(4).

Se entrena con Adam (1e-3), pesos de clase balanceados (máximo 50), lotes de 128 y parada temprana con paciencia 8 sobre la pérdida de validación. Semilla 42: el resultado es reproducible.

## 4. Exportación TinyML

| Métrica | Valor |
|---|---|
| Cuantización | **int8 completa** (pesos, activaciones, entrada y salida) |
| Tamaño del `.tflite` int8 | **20 848 bytes** (float: 37 744) |
| Operadores | CONV_2D, EXPAND_DIMS, FULLY_CONNECTED, MAX_POOL_2D, MEAN, RESHAPE, SOFTMAX |
| Compatibilidad TFLM | ✅ Todos los operadores están en el `MicroMutableOpResolver` de TFLM |
| Entrada | int8 [1, 252, 1], escala 0,0461 y punto cero −18 |
| Concordancia float/int8 en DS2 | 98,7 % de los latidos con la misma clase |
| **Arena TFLM** | ❌ **Sin medir**: hay que medirla en el ESP32 con `arena_used_bytes()` |
| **Latencia por latido** | ❌ **Sin medir** en el ESP32 |

Los archivos generados son `device-firmware/include/generated/{model_data.h, model_config.h, preprocessing_coeffs.h}` y `device-firmware/src/generated/model_data.cc`.

## 5. Resultados en DS2 (pacientes no vistos)

**Modelo int8, el que se despliega:**

| Clase | n | Sensibilidad | VPP | Especificidad | F1 |
|---|---|---|---|---|---|
| N | 44 240 | 0,952 | 0,934 | 0,457 | 0,943 |
| S | 1 837 | **0,005** | 0,008 | 0,978 | 0,006 |
| V | 3 220 | **0,720** | **0,881** | 0,993 | 0,792 |
| F | 388 | **0,000** | 0,000 | 0,982 | 0,000 |

Exactitud global 0,895 y F1 macro 0,435. La exactitud no es representativa, porque el 90 % de los latidos son N. Los resultados en float son prácticamente iguales (V: Se 0,748, VPP 0,885).

**Detector de picos R** (tolerancia ±150 ms, 22 registros de DS2): **sensibilidad 99,2 %, VPP 99,1 %** (49 259 aciertos, 467 falsos positivos, 398 no detectados).

### Interpretación

- **V (ventricular):** es la única clase con un rendimiento útil como base. Por eso es la **única que puede generar alertas** (`ECG_ALERT_ELIGIBLE_CLASSES "V"`).
- **S: no se detecta.** Los latidos supraventriculares tienen una morfología casi normal y se distinguen sobre todo por llegar **antes de tiempo** (intervalo RR corto). Un modelo que solo ve la forma del latido no puede detectarlos. Es una limitación conocida en la literatura.
- **F: no se detecta.** Solo hay 42 latidos F en entrenamiento; casi todos los de DS1 están en el registro 208, que cayó en validación. Además, el modelo confunde unos 900 latidos N con F, que serían falsas alertas si F pudiera alertar.

### Experimento descartado (registrado para no repetirlo)

Se probó a validar con el último 15 % de cada registro de DS1 (`--val-mode tail`), para dejar todos los F en entrenamiento. En DS2 **empeoró** el resultado:

- la sensibilidad de N bajó a 0,68;
- unos 9 400 latidos N pasaron a clasificarse como F;
- la VPP de V bajó a 0,72.

Se mantiene la validación por registros. **DS2 se ha consultado 3 veces** (modelo inicial con menos capacidad, este experimento y el modelo final). Cualquier ajuste posterior debería hacerse con validación cruzada dentro de DS1, dejando DS2 solo para el informe final.

## 6. Próximos pasos del modelo (por prioridad)

1. **Añadir intervalos RR como entrada**: RR previo, RR posterior y la razón con el RR medio local. Ya se calculan en el detector, cuestan muy poco en el ESP32 y son la vía estándar para detectar S.
2. Validación cruzada por paciente dentro de DS1 para elegir hiperparámetros sin tocar DS2.
3. Aumento de datos para F y S (desplazamiento, escalado, ruido), o fusionar F con V para las alertas.
4. **Reentrenar a la frecuencia real del sensor** (`train --fs <Hz>`) cuando se conozca.
5. Ajustar o reentrenar con grabaciones del sensor real, porque la derivación puede no ser MLII.
6. Medir en el ESP32 la arena, la latencia y la coincidencia con `predict_tflite` en Python.

## 7. Reproducir

```sh
cd ai-model
python -m src.training.dataset_loader --download
python -m src.training.train
python -m src.evaluation.evaluate --detector
python -m src.export.export_model
```

Artefactos generados: `ai-model/models/{model_info,eval_report,export_report,train_history}.json`.


## 8. Versión 2: latido + intervalos RR (implementada, pendiente de entrenar)

### 8.1 Motivo
El diagnóstico de la v1 mostró que el modelo solo ve la forma del latido, y los latidos S se distinguen sobre todo por llegar **antes de tiempo**.

Con RR calculados con todos los latidos anotados, el porcentaje de latidos con `pre/local` < 0,85 es:

| | S | N |
|---|---|---|
| DS2 | 85 % | 4,4 % |
| Registro 232 (el 75 % de los S de DS2) | 84 % | 0,5 % |

Se descartó la mediana de 5 minutos como referencia de largo plazo, porque en el registro 232 da un 0 %: allí los S son tan frecuentes que la mediana acaba siendo su propio RR.

### 8.2 Arquitectura de entradas
```
"beat" int8 [1,252,1] ─► Conv1D 16/32/32 + BN + ReLU + MaxPool ─► GAP (32) ┐
"rr"   int8 [1,8]     ─► Dense 16 ReLU ──────────────────────────────────────┴► Concatenate ─► Dense 32 ─► Dropout 0,2* ─► Softmax (N,S,V,F)
* el dropout solo actúa en el entrenamiento
```
- Unos 7 900 parámetros y unos 22,6 KB en int8 (prototipo).
- Operadores: CONV_2D, MAX_POOL_2D, MEAN, FULLY_CONNECTED, **CONCATENATION**, RESHAPE, EXPAND_DIMS y SOFTMAX, todos soportados por TFLM.
- El conversor TFLite **reordena las entradas**: en las pruebas, `rr` quedó como entrada 0. El exportador las identifica por nombre y escribe sus índices en `model_config.h` (`ECG_MODEL_BEAT_INPUT_INDEX` y `ECG_MODEL_RR_INPUT_INDEX`).

### 8.3 Cálculo y normalización de los RR
Implementado en `ai-model/src/preprocessing/rr_features.py`, con su espejo en C++ (`ecg::RrFeatureTracker`). Para el latido *k*:

| Variable | Definición |
|---|---|
| `pre` | t_k − t_{k−1} |
| `post` | min(t_{k+1} − t_k, 3 s). Obliga a **esperar un latido** |
| `local` | Media de los 10 RR anteriores, sin incluir `pre` |
| `long` | Media exponencial (constante de 300 s) previa a incluir `pre` |

- **Características:** `log2(pre/local)`, `log2(post/local)`, `log2(post/pre)`, `log2(pre/long)`, `log2(local/long)`, `log2(pre/0,8 s)`, `log2(local/0,8 s)` y `log2(long/0,8 s)`, todas recortadas a ±2.
- Son **transformaciones fijas**, sin parámetros ajustados a los datos, así que no pueden filtrar información entre pacientes.
- **Picos usados:** en entrenamiento, todas las anotaciones de latido, incluidas las que no se clasifican. En el dispositivo, los picos del detector.
- **Artefactos:** un RR mayor de 3 s reinicia el historial local y el calentamiento. Un electrodo suelto reinicia el rastreador.
- **Calentamiento:** hacen falta al menos 3 RR en el historial y 30 s desde el inicio o el último reinicio. Los latidos en calentamiento **no se usan para entrenar ni para evaluar**, y en el dispositivo se clasifican pero **no generan alerta**.

### 8.4 Cuantización int8
- **Calibración estratificada:** hasta 150 latidos de **cada** clase (`train.stratified_representative`).
  - Motivo: con una muestra que no incluía latidos prematuros, la entrada RR solo representaba valores en [−0,34, 0,41], los RR de los latidos S y V se saturaban y la concordancia int8/float cayó al 61,5 %.
  - Con una calibración de todas las clases, la concordancia fue del 100 % en la prueba.
- **Comprobación de saturación en el exportador:** menos del 0,5 % de los valores de cada entrada pueden quedar fuera del rango int8, o la exportación falla.
- **Escalas de CONCATENATION:** se comprueba que todas sus entradas y su salida tengan la misma escala, como exige TFLM.
- **Cuantización idéntica en Python y en C++:** redondeo de la mitad lejos de cero, verificado bit a bit (C2).

### 8.5 Validación y prevención de fugas
- **Validación cruzada de 5 grupos por registro dentro de DS1**, con validación anidada para la parada temprana. El modelo final se entrena con todo DS1. Las comprobaciones automáticas están en `python -m src.training.cv --verify`.

| Grupo | Registros | N | S | V | F |
|---|---|---|---|---|---|
| 1 | 101, 106, 208, 220 | 6 734 | 99 | 1 487 | **360** |
| 2 | 108, 109, 209, 215, 122 | 12 299 | 390 | 213 | 4 |
| 3 | 112, 114, 201, 203 | 8 323 | 141 | 679 | 7 |
| 4 | 115, 116, 207, 223, 124 | 9 157 | 211 | 781 | 19 |
| 5 | 118, 119, 205, 230 | 8 383 | 99 | 523 | 11 |

  (Solo latidos calientes y con RR siguiente.)

- **Limitación de F:** el 90 % de los latidos F de DS1 están en el grupo 1, casi todos del registro 208. Cuando se evalúe el grupo 1, el modelo se habrá entrenado con solo 41 latidos F, así que **la estimación de F en la validación cruzada será poco fiable y muy dependiente de un único paciente**. En DS2 ocurre lo mismo: el 93 % de los F son del registro 213. **F se evaluará y reportará por separado, sin presentarla como funcionalidad validada.**
- **Fuga conocida:** los registros 201 (DS1) y 202 (DS2) son del mismo paciente. DS2 se reportará también sin el 202.
- **DS2 es una evaluación exploratoria:** ya se consultó 3 veces con la v1. Se evaluará **una sola vez**, al final, sin ajustar nada después.

### 8.6 Criterios de aceptación (fijados antes de entrenar)
Se evalúan sobre las predicciones de la validación cruzada (cada latido, con el modelo que no lo vio). DS2 es secundario y exploratorio. **Estos umbrales no representan validación clínica.**

| Criterio | Umbral |
|---|---|
| V | Se ≥ 0,80 y VPP ≥ 0,85 |
| S | Se ≥ 0,50, como mínimo exploratorio. **Solo genera alertas experimentales si Se ≥ 0,50 y VPP ≥ 0,50**; si no, se clasifica y se transmite pero no alerta |
| F | Se reportan Se y VPP por separado, sin umbral |
| Falsas alertas por hora | Se reportan |
| Técnicos | Concordancia int8/float ≥ 98 %, saturación < 0,5 %, modelo ≤ 64 KB, operadores TFLM y pruebas de equivalencia superadas |

### 8.7 Pruebas realizadas (todas superadas)

| Prueba | Qué verifica | Resultado |
|---|---|---|
| P1 | Implementación por lotes frente a la latido a latido (sintético y registros 201, 208 y 209 de DS1) | Diferencia < 1e-9 |
| P2 | Valores calculados a mano, recorte, pausas, reinicio, calentamiento y *timeout* | ✅ |
| P3 / P3b | Grupos válidos (22 registros, una vez cada uno), rechazo de grupos inválidos, pares del mismo paciente, causalidad | ✅ |
| P4 | Conversión int8, operadores TFLM, escala de CONCATENATION, concordancia ≥ 98 % (100 %), saturación < 0,5 %, y que la comprobación detecta una calibración defectuosa | ✅ |
| P5 | `model_config.h` e `integration_config.h` coherentes con Python | ✅ |
| C1 | `RrFeatureTracker` C++ frente a Python (697 latidos, con pausas y reinicio) | R y calentamiento idénticos; error 6,5e-7 |
| C2 | Cuantización int8 C++ frente a Python | 276/276 idénticos |
| C3 | Pipeline C++ frente a `simulate_device` (110 s, con electrodo suelto y una pausa de 8 s) | 112/112 latidos en la misma muestra; error de 1,6e-4 en el latido y 3,9e-7 en RR; 1 latido por *timeout* con ventana fuera del buffer; 12 alertas y ninguna en calentamiento |
| Pruebas de la réplica del dispositivo | El latido anterior a una pausa larga se clasifica; no se pierde ningún latido; calentamiento; constantes iguales al firmware | ✅ |

**Fallo de diseño detectado y corregido:** con una pausa larga, el latido anterior se emitía por *timeout* cuando su ventana ya no estaba en el buffer circular, y se descartaba sin avisar. Ahora la ventana se **copia en la entrada del latido pendiente** en cuanto está completa (unos 4 KB de RAM en el firmware), y C3 comprueba esta ruta de forma obligatoria.

**Prueba T1 (2026-11-19, TFLM en PC, modelo v1): NO APROBADA.**
- La clase coincide en 1007/1007 latidos, pero la diferencia máxima de salida es de 3 LSB y el criterio es ±1 LSB.
- La causa es el doble redondeo de TFLM en FULLY_CONNECTED por canal, que SOFTMAX amplifica.
- Ver `device-firmware/test/tflm/README.md`.

### 8.8 Resultados de la validación cruzada (5 grupos por paciente, DS1)

Cada latido se predice con el modelo que no lo vio. El informe completo está en `ai-model/models/cv_v2_report.json`. **Ninguna de estas cifras es una validación clínica.**

**Esquema elegido: `balanced`.** Cumple los mismos criterios que `sqrt` y tiene mayor F1 macro de N/S/V; además, `sqrt` no supera el umbral de concordancia int8.

| Clase (int8) | n | Se | VPP | Esp | F1 | Criterio |
|---|---|---|---|---|---|---|
| N | 44 896 | 0,811 | 0,975 | 0,815 | 0,885 | — |
| S | 940 | **0,490** | 0,337 | 0,982 | 0,400 | Se ≥ 0,50 ❌ (por muy poco). Alertas S ❌ (VPP < 0,50) |
| V | 3 683 | **0,737** | **0,659** | 0,970 | 0,696 | Se ≥ 0,80 ❌, VPP ≥ 0,85 ❌ |
| F | 401 | 0,007 | 0,000 | 0,857 | 0,001 | Solo se reporta. **No detecta F** |

**Datos técnicos:**
- Concordancia int8/float: 98,65 % ✅.
- Saturación máxima: 0,26 % ✅.
- Tamaño: 23,4 KB ✅.
- Operadores TFLM y escala de CONCATENATION ✅.

**Falsas alertas:**
- Solo con V: 39,1 por hora (431 de 1 824 alertas).
- Si S también alertara: 62,4 por hora.

**RR calculados desde el detector** (lo realista en el dispositivo), modelo float:
- S: Se 0,434, VPP 0,299.
- V: Se 0,734, VPP 0,609.
- La degradación es pequeña frente a usar las anotaciones.

**Comparación con la v1:** la sensibilidad de S pasa de prácticamente 0 a 0,49. Sin embargo, la VPP de V baja, y **unos 7 000 latidos N (el 15,5 %) se clasifican como F**, concentrados en los registros 109, 215, 122, 203 y 207. Las falsas alertas de V se concentran en los registros 203 (475/h), 201 (251/h) y 207 (108/h).

**Decisión, según las reglas fijadas:** la validación **no es satisfactoria**, porque no se cumple el criterio de V. **No se ha entrenado el modelo final ni se ha evaluado DS2 con la v2.** El firmware sigue con el modelo v1 y con alertas solo de V.

### 8.9 Iteración 2 (última iteración de desarrollo) — reglas fijadas antes de ejecutarla

Autorizada tras la sección 8.8. Se diseñó **a la vista de los resultados de la iteración 1**, así que la validación cruzada ya es en parte datos de desarrollo: sus cifras son algo optimistas.

- **Un único esquema, sin elección posterior:** `balanced`, con el **peso de F limitado a 5**. En la iteración 1 F llegaba al tope de 50 en los grupos sin el registro 208, unas 170 veces el peso de N. Con 5 queda por debajo del de S (~7–10). **F sigue siendo una clase independiente**; no se fusiona con V.
- **8 épocas fijas, sin parada temprana.** Es la mediana de las mejores épocas de la iteración 1 con `balanced` (17, 2, 4, 14 y 8), que eran muy inestables entre grupos.
- **Mismos 5 grupos por paciente.** Al no haber parada temprana no hace falta grupo de validación, y se entrena con los 4 grupos que no son de prueba (antes, 3). Ese cambio también puede influir en el resultado.
- **Criterios de aceptación sin cambios.** DS2 no se consulta. El modelo final solo se entrenará con una nueva autorización.
- Si no se alcanzan los criterios, se documentan las limitaciones y el firmware sigue con la v1.

Comando: `python -m src.training.rr_experiment cv2`. Informe: `ai-model/models/cv_v2_iter2_report.json`.

### 8.10 Resultados de la iteración 2 y decisión

Las cifras son predicciones int8 fuera de grupo y se comparan con la iteración 1 (sección 8.8). Informe completo: `ai-model/models/cv_v2_iter2_report.json`. **Ninguna de estas cifras es una validación clínica.**

| Clase | Se it1 → it2 | VPP it1 → it2 | F1 it1 → it2 |
|---|---|---|---|
| N | 0,811 → **0,899** | 0,975 → 0,983 | 0,885 → 0,939 |
| S | 0,490 → **0,527** ✅ | 0,337 → 0,277 | 0,400 → 0,363 |
| V | 0,737 → **0,808** ✅ | 0,659 → **0,535** ❌ | 0,696 → 0,644 |
| F | 0,007 → 0,002 | 0,000 → 0,001 | 0,001 → 0,001 |

- **Exactitud:** 0,793 → 0,878.
- **Falsas alertas por hora:** solo V, 39,1 → **62,4**; solo S, 26,1 → 34,2; V y S, 62,4 → 92,3.
- **Matriz de confusión de la iteración 2 (int8)** (filas: clase real N, S, V, F; columnas: clase predicha en el mismo orden):
  ```
  N [40363,  876, 2175, 1482]
  S [  122,  495,  322,    1]
  V [  256,  410, 2976,   41]
  F [  306,    4,   90,    1]
  ```
- **Float y RR del detector:**
  - float: V 0,824 / 0,528, S 0,532 / 0,265;
  - RR del detector: V 0,806 / 0,491, S 0,534 / 0,237.
- **Datos técnicos, todos ✅:**
  - concordancia int8/float: 99,03 %;
  - saturación máxima: 0,46 %, por debajo del 0,5 % pero cerca del límite (grupo 3, entrada RR);
  - tamaño: 22,7–22,8 KB;
  - operadores TFLM y escala de CONCATENATION correctos.

**Qué cambió:**
- Limitar el peso de F eliminó casi todo el error N→F (6 966 → 1 482 latidos). La sensibilidad de N subió en 109 (0,07 → 1,00), 215 (0,10 → 0,99) y 122 (0,59 → 1,00).
- Esos errores se desplazaron a N→V (1 094 → 2 175), concentrados en el registro 203 (1 680 latidos) y en 118 y 207. N→F aumentó en 114, 201 y 203.
- V gana sensibilidad, pero su VPP cae por debajo del criterio.
- Aparece confusión V→S en los registros 106, 109 y 215: latidos V prematuros con morfología poco frecuente se clasifican por su RR como S.
- F sigue sin detectarse. Su único paciente con muchos F es 208, que solo está en entrenamiento en 4 de los 5 grupos.

**Decisión, según las reglas fijadas:**
- La validación **no es satisfactoria**: se alcanza la sensibilidad de V, pero no su VPP (0,535 < 0,85).
- Las alertas S **no se permiten** (VPP 0,277 < 0,50).
- Esta era la última iteración de desarrollo autorizada: **no se entrena el modelo final de la v2, no se evalúa DS2 con ella y el firmware sigue con el modelo v1.**

**Limitaciones de la v2:**
- Con solo 22 pacientes de entrenamiento, el modelo no generaliza a morfologías que no ha visto (registros 203, 201, 207 y 118).
- Las RR ayudan a S, pero introducen confusión V↔S.
- F no se puede aprender con los datos de DS1: son unos 400 latidos, el 90 % de un solo paciente.
