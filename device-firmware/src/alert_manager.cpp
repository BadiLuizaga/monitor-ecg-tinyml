#include "alert_manager.h"

#include <cstring>

namespace ecg {

void AlertManager::init(float fs) {
  cooldown_ = static_cast<int64_t>(ECG_ALERT_COOLDOWN_S * fs);
  reset();
}

void AlertManager::reset() {
  for (bool& b : window_) b = false;
  pos_ = 0;
  lastAlert_ = -1000000000LL;
}

AlertDecision AlertManager::onBeat(char beatClass, float confidence, uint32_t sampleIndex) {
  const bool eligible = beatClass != '\0' && std::strchr(ECG_ALERT_ELIGIBLE_CLASSES, beatClass) != nullptr;
  const bool abnormal = eligible && confidence >= ECG_ALERT_MIN_CONFIDENCE;
  window_[pos_] = abnormal;
  pos_ = (pos_ + 1) % ECG_ALERT_WINDOW_BEATS;
  int count = 0;
  for (bool b : window_) count += b ? 1 : 0;

  AlertDecision d;
  if (abnormal && count >= ECG_ALERT_MIN_ABNORMAL && static_cast<int64_t>(sampleIndex) - lastAlert_ >= cooldown_) {
    lastAlert_ = sampleIndex;
    d.raise = true;
    d.beatClass = beatClass;
    d.confidence = confidence;
    d.sampleIndex = sampleIndex;
  }
  return d;
}

}  // namespace ecg
