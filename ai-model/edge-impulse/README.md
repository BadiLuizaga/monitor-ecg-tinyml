# edge-impulse/

**Estado: no implementado.** La ruta principal del proyecto es TensorFlow/Keras → TFLite int8 → TensorFlow Lite Micro (ver `src/export/`), que no depende de servicios externos.

Edge Impulse puede usarse como **alternativa** o para comparar resultados:

1. Exportar los latidos preprocesados (por ejemplo, a CSV con una columna por muestra y la etiqueta AAMI) usando `dataset_loader.build_beat_dataset`.
2. Subirlos al proyecto de Edge Impulse. Requiere una cuenta y aceptar sus condiciones.
3. Comparar las métricas con `docs/model_documentation.md`, **usando la misma división DS1/DS2**.

Se documentará aquí si el equipo decide usarlo.
