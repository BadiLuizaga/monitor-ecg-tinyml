// inference_tflm.cpp - Clasificador con TensorFlow Lite Micro.
//
// ESTADO: compilado y ejecutado en PC con TFLM (commit b8d75cbf6b80) y el
// modelo v1 (prueba T1, device-firmware/test/tflm). NO probado en el ESP32
// (p. ej. con el componente esp-tflite-micro). Requiere los archivos generados
// por `python -m src.export.export_model`. Se activa con -DECG_HAVE_TFLM=1.
// La API puede variar entre versiones de TFLM.
//
// Las entradas se localizan con los índices exportados en model_config.h
// (ECG_MODEL_BEAT_INPUT_INDEX / ECG_MODEL_RR_INPUT_INDEX): el conversor TFLite
// puede reordenarlas respecto al modelo Keras.
#if defined(ECG_HAVE_TFLM) && ECG_HAVE_TFLM

#include "generated/model_config.h"
#include "generated/model_data.h"
#include "inference.h"
#include "integration_config.h"
#include "signal_processing.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
// El esquema se movió a tensorflow/lite/micro/schema/ en TFLM reciente
// (probado en PC con el commit b8d75cbf6b80); versiones anteriores lo tienen
// en tensorflow/lite/schema/.
#if __has_include("tensorflow/lite/micro/schema/schema_generated.h")
#include "tensorflow/lite/micro/schema/schema_generated.h"
#else
#include "tensorflow/lite/schema/schema_generated.h"
#endif

namespace ecg {
namespace {
alignas(16) uint8_t g_arena[ECG_TFLM_ARENA_BYTES];
tflite::MicroInterpreter* g_interpreter = nullptr;
}  // namespace

bool TflmInferenceEngine::begin() {
  const tflite::Model* model = tflite::GetModel(g_ecg_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) return false;
  // Operadores que usa el modelo (ver ai-model/models/export_report.json).
  static tflite::MicroMutableOpResolver<9> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddExpandDims();
  resolver.AddConcatenation();
  static tflite::MicroInterpreter interpreter(model, resolver, g_arena, sizeof g_arena);
  if (interpreter.AllocateTensors() != kTfLiteOk) return false;  // arena insuficiente u op faltante
  g_interpreter = &interpreter;
  const TfLiteTensor* beat = interpreter.input(ECG_MODEL_BEAT_INPUT_INDEX);
  ready_ = beat->type == kTfLiteInt8 && beat->bytes == ECG_MODEL_INPUT_LENGTH;
#ifdef ECG_MODEL_HAS_RR_INPUT
  const TfLiteTensor* rr = interpreter.input(ECG_MODEL_RR_INPUT_INDEX);
  ready_ = ready_ && rr->type == kTfLiteInt8 && rr->bytes == ECG_MODEL_RR_FEATURES;
#endif
  return ready_;
}

size_t TflmInferenceEngine::inputLength() const { return ECG_MODEL_INPUT_LENGTH; }

size_t TflmInferenceEngine::rrFeatures() const {
#ifdef ECG_MODEL_HAS_RR_INPUT
  return ECG_MODEL_RR_FEATURES;
#else
  return 0;
#endif
}

size_t TflmInferenceEngine::arenaUsedBytes() const {
  return g_interpreter ? g_interpreter->arena_used_bytes() : 0;
}

bool TflmInferenceEngine::classify(const float* beat, size_t length, const float* rr, size_t rrCount,
                                   InferenceResult* out) {
  if (!ready_ || length != ECG_MODEL_INPUT_LENGTH || out == nullptr) return false;
  TfLiteTensor* in = g_interpreter->input(ECG_MODEL_BEAT_INPUT_INDEX);
  for (size_t i = 0; i < length; ++i)
    in->data.int8[i] = quantizeInt8(beat[i], ECG_MODEL_INPUT_SCALE, ECG_MODEL_INPUT_ZERO_POINT);
#ifdef ECG_MODEL_HAS_RR_INPUT
  if (rr == nullptr || rrCount != ECG_MODEL_RR_FEATURES) return false;
  TfLiteTensor* rin = g_interpreter->input(ECG_MODEL_RR_INPUT_INDEX);
  for (size_t i = 0; i < rrCount; ++i)
    rin->data.int8[i] = quantizeInt8(rr[i], ECG_MODEL_RR_SCALE, ECG_MODEL_RR_ZERO_POINT);
#else
  (void)rr;
  (void)rrCount;
#endif
  if (g_interpreter->Invoke() != kTfLiteOk) return false;
  const TfLiteTensor* o = g_interpreter->output(0);
  int best = 0;
  for (int k = 1; k < ECG_MODEL_NUM_CLASSES; ++k)
    if (o->data.int8[k] > o->data.int8[best]) best = k;
  out->classIndex = best;
  out->beatClass = ECG_MODEL_CLASSES[best];
  out->confidence = (o->data.int8[best] - ECG_MODEL_OUTPUT_ZERO_POINT) * ECG_MODEL_OUTPUT_SCALE;
  return true;
}

}  // namespace ecg

#endif  // ECG_HAVE_TFLM
