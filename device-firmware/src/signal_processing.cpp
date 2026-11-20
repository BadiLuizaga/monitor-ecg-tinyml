#include "signal_processing.h"

#include <cmath>

namespace ecg {

bool BiquadCascade::init(const float (*sos)[5], size_t sections) {
  if (sections == 0 || sections > kMaxSections) return false;
  for (size_t i = 0; i < sections; ++i)
    for (int k = 0; k < 5; ++k) c_[i][k] = sos[i][k];
  n_ = sections;
  reset();
  return true;
}

void BiquadCascade::reset() {
  for (auto& z : z_) z[0] = z[1] = 0.0f;
}

float BiquadCascade::process(float x) {
  for (size_t i = 0; i < n_; ++i) {
    const float* c = c_[i];
    const float y = c[0] * x + z_[i][0];
    z_[i][0] = c[1] * x - c[3] * y + z_[i][1];
    z_[i][1] = c[2] * x - c[4] * y;
    x = y;
  }
  return x;
}

bool RPeakDetector::init(float fs) {
  int w = static_cast<int>(std::lround(ECG_DET_INTEGRATION_S * fs));
  if (w < 1) w = 1;
  if (w > ECG_DET_MAX_INTEGRATION_SAMPLES) return false;
  *this = RPeakDetector{};  // estado limpio
  w_ = w;
  refractory_ = static_cast<int>(std::lround(ECG_DET_REFRACTORY_S * fs));
  learning_ = static_cast<int>(std::lround(ECG_DET_LEARNING_S * fs));
  return true;
}

bool RPeakDetector::process(float x, uint32_t* rIndex) {
  const int64_t n = n_++;
  const float d = n > 0 ? x - prevX_ : 0.0f;
  prevX_ = x;
  const float s = d * d;
  integSum_ += static_cast<double>(s) - sq_[sqPos_];
  sq_[sqPos_] = s;
  sqPos_ = (sqPos_ + 1) % w_;
  const float integ = static_cast<float>(integSum_ / w_);
  const int histCap = 2 * w_;
  hist_[histPos_] = x;
  histPos_ = (histPos_ + 1) % histCap;
  if (histLen_ < histCap) ++histLen_;

  if (n < learning_) {
    if (integ > learnMax_) learnMax_ = integ;
    if (n == learning_ - 1) {
      spk_ = 0.5f * learnMax_;
      npk_ = 0.1f * spk_;
    }
    prevI0_ = prevI1_;
    prevI1_ = integ;
    return false;
  }

  bool found = false;
  const float thr = npk_ + ECG_DET_THRESHOLD_RATIO * (spk_ - npk_);
  if (!inQrs_) {
    if (integ > thr && n - lastR_ > refractory_) {
      inQrs_ = true;
      qrsStart_ = n;
      qrsMax_ = integ;
    } else if (prevI1_ > prevI0_ && prevI1_ >= integ) {
      npk_ = 0.125f * prevI1_ + 0.875f * npk_;
    }
  } else {
    if (integ > qrsMax_) qrsMax_ = integ;
    if (integ < 0.5f * thr || n - qrsStart_ >= w_) {
      int span = static_cast<int>(n - qrsStart_) + w_;
      if (span > histLen_) span = histLen_;
      // Máximo de las últimas `span` muestras (la más antigua primero, como en Python).
      int best = 0;
      float bestV = -INFINITY;
      for (int k = 0; k < span; ++k) {
        const int pos = (histPos_ - span + k + 2 * histCap) % histCap;
        if (hist_[pos] > bestV) {
          bestV = hist_[pos];
          best = k;
        }
      }
      const int64_t r = n - span + 1 + best;
      if (r - lastR_ > refractory_) {
        lastR_ = r;
        if (rIndex) *rIndex = static_cast<uint32_t>(r);
        found = true;
      }
      spk_ = 0.125f * qrsMax_ + 0.875f * spk_;
      inQrs_ = false;
    }
  }
  prevI0_ = prevI1_;
  prevI1_ = integ;
  return found;
}

void zscoreInPlace(float* x, size_t n) {
  if (n == 0) return;
  double mean = 0.0;
  for (size_t i = 0; i < n; ++i) mean += x[i];
  mean /= static_cast<double>(n);
  double var = 0.0;
  for (size_t i = 0; i < n; ++i) var += (x[i] - mean) * (x[i] - mean);
  double sd = std::sqrt(var / static_cast<double>(n));
  if (sd < 1e-6) sd = 1e-6;
  for (size_t i = 0; i < n; ++i) x[i] = static_cast<float>((x[i] - mean) / sd);
}

int8_t quantizeInt8(float x, float scale, int zeroPoint) {
  // std::round redondea la mitad lejos de cero, como quantize_int8 en Python.
  const double q = std::round(static_cast<double>(x) / static_cast<double>(scale)) + zeroPoint;
  return static_cast<int8_t>(q < -128.0 ? -128 : (q > 127.0 ? 127 : static_cast<int>(q)));
}

// --- RrFeatureTracker ------------------------------------------------------

void RrFeatureTracker::init(float fs) {
  *this = RrFeatureTracker{};
  fs_ = fs;
}

void RrFeatureTracker::reset() {
  hasLast_ = false;
  hasWarmStart_ = false;
  hasPending_ = false;
  histCount_ = 0;
  histPos_ = 0;
}

static float clipLog2(double ratio) {
  double v = std::log2(ratio);
  if (v < -ECG_RR_CLIP) v = -ECG_RR_CLIP;
  if (v > ECG_RR_CLIP) v = ECG_RR_CLIP;
  return static_cast<float>(v);
}

void RrFeatureTracker::finalize(float postS, RrBeat* out) {
  const float post = postS < ECG_RR_MAX_S ? postS : ECG_RR_MAX_S;
  const double pre = pPre_, local = pLocal_, lng = pLong_;
  // Mismo orden que RR_FEATURE_NAMES (ECG_RR_FEATURE_ORDER).
  const double ratios[ECG_RR_NUM_FEATURES] = {pre / local, post / local, post / pre,  pre / lng,
                                              local / lng, pre / ECG_RR_REF_S, local / ECG_RR_REF_S,
                                              lng / ECG_RR_REF_S};
  out->rIndex = pR_;
  out->warm = pWarm_;
  out->preS = pPre_;
  out->postS = post;
  for (int i = 0; i < ECG_RR_NUM_FEATURES; ++i) out->features[i] = clipLog2(ratios[i]);
  hasPending_ = false;
}

bool RrFeatureTracker::onBeat(uint32_t r, RrBeat* out) {
  if (!hasLast_) {
    hasLast_ = true;
    last_ = r;
    if (!hasWarmStart_) {
      hasWarmStart_ = true;
      warmStart_ = r;
    }
    return false;
  }
  const float pre = static_cast<float>(r - last_) / fs_;
  bool emitted = false;
  if (hasPending_ && out) {
    finalize(pre, out);
    emitted = true;
  }
  const bool valid = pre > 0.0f && pre <= ECG_RR_MAX_S;
  const float preC = pre < ECG_RR_MAX_S ? pre : ECG_RR_MAX_S;
  float local = preC;
  if (histCount_ > 0) {
    float sum = 0.0f;  // del más antiguo al más reciente, como sum(deque) en Python
    for (int k = 0; k < histCount_; ++k)
      sum += hist_[(histPos_ - histCount_ + k + ECG_RR_LOCAL_BEATS) % ECG_RR_LOCAL_BEATS];
    local = sum / static_cast<float>(histCount_);
  }
  pR_ = r;
  pPre_ = preC;
  pLocal_ = local;
  pLong_ = hasEma_ ? ema_ : preC;
  pWarm_ = valid && histCount_ >= ECG_RR_WARMUP_MIN_RR &&
           static_cast<float>(r - warmStart_) / fs_ >= ECG_RR_WARMUP_S;
  hasPending_ = true;
  if (valid) {
    hist_[histPos_] = pre;
    histPos_ = (histPos_ + 1) % ECG_RR_LOCAL_BEATS;
    if (histCount_ < ECG_RR_LOCAL_BEATS) ++histCount_;
    if (!hasEma_) {
      ema_ = pre;
      hasEma_ = true;
    } else {
      ema_ += (1.0f - std::exp(-pre / ECG_RR_LONG_TAU_S)) * (pre - ema_);
    }
  } else {
    histCount_ = 0;
    histPos_ = 0;
    warmStart_ = r;
    hasWarmStart_ = true;
  }
  last_ = r;
  return emitted;
}

bool RrFeatureTracker::onTimeout(uint32_t now, RrBeat* out) {
  if (!hasPending_ || out == nullptr) return false;
  if (static_cast<float>(now - pR_) / fs_ < ECG_RR_MAX_S) return false;
  finalize(ECG_RR_MAX_S, out);
  return true;
}

}  // namespace ecg
