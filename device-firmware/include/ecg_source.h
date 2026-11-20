// ecg_source.h - Origen de las muestras ECG.
//
// La lógica de integración solo conoce esta interfaz. La adquisición real
// (ADC/front-end, temporización, lead-off) se implementará en una clase
// HardwareEcgSource cuando se confirmen las especificaciones (ver
// hardware_config.h y docs/integration.md). No se incluye aquí código de un
// sensor concreto porque aún no está definido.
#pragma once

#include <cstdint>

namespace ecg {

struct EcgSample {
  int16_t value = 0;     // cuentas del ADC con signo
  bool leadOff = false;  // electrodo desconectado (si el hardware lo detecta)
  bool saturated = false;
};

class IEcgSource {
 public:
  virtual ~IEcgSource() = default;
  virtual bool begin() = 0;
  // No bloqueante: true si había una muestra nueva.
  virtual bool read(EcgSample* out) = 0;
  virtual float sampleRateHz() const = 0;
  virtual bool isSimulated() const = 0;
};

// Fuente SIMULADA y determinista: tren de complejos tipo QRS (triangulares)
// con ruido pseudoaleatorio reproducible. Sirve para probar el firmware sin
// sensor; no representa una señal fisiológica real. Cada llamada a read()
// entrega la siguiente muestra (el ritmo lo marca quien llama).
class TestPatternSource : public IEcgSource {
 public:
  TestPatternSource(float fs, float heartRateBpm, uint32_t seed = 1)
      : fs_(fs), period_(static_cast<uint32_t>(fs * 60.0f / heartRateBpm)), rng_(seed ? seed : 1) {}
  bool begin() override { return true; }
  bool read(EcgSample* out) override;
  float sampleRateHz() const override { return fs_; }
  bool isSimulated() const override { return true; }
  // Fuerza "electrodo desconectado" durante [start, end) muestras (pruebas).
  void setLeadOffWindow(uint32_t start, uint32_t end) { leadOffStart_ = start; leadOffEnd_ = end; }

 private:
  float fs_;
  uint32_t period_;
  uint32_t rng_;
  uint32_t n_ = 0;
  uint32_t leadOffStart_ = 0, leadOffEnd_ = 0;
};

}  // namespace ecg
