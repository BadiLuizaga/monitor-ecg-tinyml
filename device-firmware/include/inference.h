// inference.h - Interfaz del clasificador de latidos.
//
// * StubInferenceEngine: siempre "no disponible" (MODEL_NOT_LOADED). Se usa
//   mientras no esté integrada TFLM. NUNCA inventa predicciones.
// * TflmInferenceEngine: implementación con TensorFlow Lite Micro
//   (src/inference_tflm.cpp), compilada solo si se define ECG_HAVE_TFLM=1 y
//   existen los archivos generados por ai-model/src/export/export_model.py.
#pragma once

#include <cstddef>
#include <cstdint>

namespace ecg {

struct InferenceResult {
  int classIndex = -1;
  char beatClass = '-';
  float confidence = 0.0f;
};

class IInferenceEngine {
 public:
  virtual ~IInferenceEngine() = default;
  virtual bool begin() = 0;
  virtual bool ready() const = 0;
  // `beat`: latido filtrado y normalizado (z-score), longitud = inputLength().
  // `rr`: características RR en el orden ECG_RR_FEATURE_ORDER; `rrCount` debe
  // ser rrFeatures() (0 si el modelo no usa RR, en cuyo caso se ignora).
  virtual bool classify(const float* beat, size_t length, const float* rr, size_t rrCount,
                        InferenceResult* out) = 0;
  virtual size_t inputLength() const = 0;
  virtual size_t rrFeatures() const { return 0; }
  virtual size_t arenaUsedBytes() const { return 0; }
};

class StubInferenceEngine : public IInferenceEngine {
 public:
  explicit StubInferenceEngine(size_t inputLength, size_t rrFeatures = 0) : len_(inputLength), rr_(rrFeatures) {}
  bool begin() override { return false; }
  bool ready() const override { return false; }
  bool classify(const float*, size_t, const float*, size_t, InferenceResult*) override { return false; }
  size_t inputLength() const override { return len_; }
  size_t rrFeatures() const override { return rr_; }

 private:
  size_t len_;
  size_t rr_;
};

#if defined(ECG_HAVE_TFLM) && ECG_HAVE_TFLM
class TflmInferenceEngine : public IInferenceEngine {
 public:
  bool begin() override;
  bool ready() const override { return ready_; }
  bool classify(const float* beat, size_t length, const float* rr, size_t rrCount, InferenceResult* out) override;
  size_t inputLength() const override;
  size_t rrFeatures() const override;
  size_t arenaUsedBytes() const override;

 private:
  bool ready_ = false;
};
#endif

}  // namespace ecg
