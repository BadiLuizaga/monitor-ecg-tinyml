#include "ecg_source.h"

namespace ecg {

bool TestPatternSource::read(EcgSample* out) {
  if (out == nullptr) return false;
  const uint32_t n = n_++;
  const uint32_t phase = n % period_;
  const uint32_t half = static_cast<uint32_t>(fs_ * 0.02f) + 1;  // QRS ≈ 40 ms
  const uint32_t center = period_ / 2;
  int32_t v = 0;
  if (phase + half >= center && phase <= center + half) {
    const int32_t dist = static_cast<int32_t>(phase) - static_cast<int32_t>(center);
    v = 1000 - (1000 * (dist < 0 ? -dist : dist)) / static_cast<int32_t>(half);
  }
  rng_ = rng_ * 1664525u + 1013904223u;  // LCG reproducible
  v += static_cast<int32_t>((rng_ >> 24) & 0x1F) - 16;  // ruido ±16 cuentas
  out->leadOff = n >= leadOffStart_ && n < leadOffEnd_;
  out->value = out->leadOff ? 0 : static_cast<int16_t>(v);
  out->saturated = false;
  return true;
}

}  // namespace ecg
