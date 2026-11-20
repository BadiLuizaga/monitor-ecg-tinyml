// alert_manager.h - Decide cuándo una clasificación se convierte en alerta.
//
// Regla (PROVISIONAL, ver integration_config.h): se alerta si en los últimos
// ECG_ALERT_WINDOW_BEATS latidos hay al menos ECG_ALERT_MIN_ABNORMAL latidos
// no normales con confianza >= ECG_ALERT_MIN_CONFIDENCE, respetando un
// periodo mínimo entre alertas (cooldown). Evita inundar la app de avisos.
#pragma once

#include <cstdint>

#include "integration_config.h"

namespace ecg {

struct AlertDecision {
  bool raise = false;
  char beatClass = '-';
  float confidence = 0.0f;
  uint32_t sampleIndex = 0;
};

class AlertManager {
 public:
  void init(float fs);
  // beatClass: 'N', 'S', 'V', 'F'... ; devuelve la decisión para este latido.
  AlertDecision onBeat(char beatClass, float confidence, uint32_t sampleIndex);
  void reset();

 private:
  bool window_[ECG_ALERT_WINDOW_BEATS] = {};
  int pos_ = 0;
  int64_t lastAlert_ = -1000000000LL;
  int64_t cooldown_ = 0;
};

}  // namespace ecg
