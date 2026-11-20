// signal_processing.h - Filtro IIR (biquads) y detector de picos R en tiempo real.
//
// Equivalentes a ai-model/src/preprocessing/filters.py (sosfilt causal) y
// segmentation.py (StreamingRDetector). Los coeficientes del filtro se
// generan desde Python en include/generated/preprocessing_coeffs.h.
#pragma once

#include <cstddef>
#include <cstdint>

#include "integration_config.h"

namespace ecg {

// Cascada de biquads en forma directa II transpuesta (igual que scipy.sosfilt).
class BiquadCascade {
 public:
  static constexpr size_t kMaxSections = 8;
  // sos[i] = {b0, b1, b2, a1, a2}
  bool init(const float (*sos)[5], size_t sections);
  float process(float x);
  void reset();

 private:
  float c_[kMaxSections][5] = {};
  float z_[kMaxSections][2] = {};
  size_t n_ = 0;
};

// Detector Pan-Tompkins simplificado (ver segmentation.py para la descripción).
class RPeakDetector {
 public:
  bool init(float fs);
  // Devuelve true y el índice absoluto del pico R en *rIndex si detecta uno.
  bool process(float x, uint32_t* rIndex);
  uint32_t samplesProcessed() const { return static_cast<uint32_t>(n_); }

 private:
  float sq_[ECG_DET_MAX_INTEGRATION_SAMPLES] = {};
  float hist_[2 * ECG_DET_MAX_INTEGRATION_SAMPLES] = {};
  int w_ = 1, refractory_ = 0, learning_ = 0;
  int sqPos_ = 0, histPos_ = 0, histLen_ = 0;
  double integSum_ = 0.0;
  float prevX_ = 0.0f, prevI0_ = 0.0f, prevI1_ = 0.0f;
  float spk_ = 0.0f, npk_ = 0.0f, learnMax_ = 0.0f, qrsMax_ = 0.0f;
  int64_t n_ = 0, lastR_ = -1000000000LL, qrsStart_ = 0;
  bool inQrs_ = false;
};

// z-score de un latido in situ. Igual que normalization.zscore.
void zscoreInPlace(float* x, size_t n);

// Cuantización float -> int8 idéntica a rr_features.quantize_int8 (Python):
// redondeo "mitad lejos de cero" de x/scale, + zero_point, saturado a [-128, 127].
int8_t quantizeInt8(float x, float scale, int zeroPoint);

// Latido con sus características RR, listo para clasificar.
struct RrBeat {
  uint32_t rIndex = 0;
  bool warm = false;  // apto para alertar (ver rr_features.py)
  float features[ECG_RR_NUM_FEATURES] = {};
  float preS = 0.0f;
  float postS = 0.0f;
};

// Espejo exacto de ai-model/src/preprocessing/rr_features.py::RrFeatureTracker.
// El latido k se emite cuando llega el k+1 (onBeat) o cuando pasan
// ECG_RR_MAX_S sin latidos (onTimeout): retraso de un latido.
class RrFeatureTracker {
 public:
  void init(float fs);
  void reset();  // electrodo suelto: olvida historial local y pendiente
  bool onBeat(uint32_t rIndex, RrBeat* out);
  bool onTimeout(uint32_t nowIndex, RrBeat* out);
  // Índice R del latido pendiente (el primer latido tras un reinicio solo es
  // ancla y nunca queda pendiente). Igual que RrFeatureTracker.pending_index.
  bool pendingIndex(uint32_t* rIndex) const {
    if (hasPending_ && rIndex) *rIndex = pR_;
    return hasPending_;
  }

 private:
  void finalize(float postS, RrBeat* out);
  float fs_ = 360.0f;
  float hist_[ECG_RR_LOCAL_BEATS] = {};
  int histCount_ = 0, histPos_ = 0;
  bool hasEma_ = false, hasLast_ = false, hasWarmStart_ = false, hasPending_ = false;
  float ema_ = 0.0f;
  uint32_t last_ = 0, warmStart_ = 0;
  uint32_t pR_ = 0;
  float pPre_ = 0.0f, pLocal_ = 0.0f, pLong_ = 0.0f;
  bool pWarm_ = false;
};

}  // namespace ecg
