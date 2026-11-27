// main_esp32.cpp - Punto de entrada del ESP32 (E2): setup()/loop() de Arduino.
//
// ALCANCE E2: transporte BLE real (NimBLE) + fuente SIMULADA (TestPatternSource).
// NO hay AD8232 ni TFLM todavía: DEVICE_INFO declara "simulated": true y el
// STATUS informa MODEL_NOT_LOADED (el modelo no está integrado; no se inventan
// clasificaciones). hardware_config.h sigue SIN confirmar a propósito.
//
// Contexto de ejecución (ble_transport.h): el callback de NimBLE solo encola;
// loop() entrega los comandos y después procesa muestras.
#if defined(ARDUINO)

#include <Arduino.h>
#include <esp_system.h>

#include "ble_transport.h"
#include "ecg_pipeline.h"
#include "esp32_ble_transport.h"
#include "hardware_config.h"
#include "inference.h"

namespace {

// Entrega las muestras de otra fuente al ritmo de fs (reloj de micros()).
// TestPatternSource no tiene reloj propio: "el ritmo lo marca quien llama".
class PacedSource : public ecg::IEcgSource {
 public:
  explicit PacedSource(ecg::IEcgSource& inner) : inner_(inner) {}
  bool begin() override {
    last_us_ = micros();
    elapsed_us_ = 0;
    delivered_ = 0;
    return inner_.begin();
  }
  bool read(ecg::EcgSample* out) override {
    // Muestras que deberían haberse entregado ya (aritmética en 64 bits).
    // elapsed_us es módulo 2^32 (micros() da la vuelta cada ~71 min): se
    // acumula en 64 bits mientras read() se llame al menos una vez por vuelta.
    const uint32_t now = micros();
    elapsed_us_ += static_cast<uint32_t>(now - last_us_);
    last_us_ = now;
    const uint64_t total_us = elapsed_us_;
    const uint64_t due = static_cast<uint64_t>(static_cast<double>(total_us) * inner_.sampleRateHz() / 1e6);
    if (delivered_ >= due) return false;
    ++delivered_;
    return inner_.read(out);
  }
  float sampleRateHz() const override { return inner_.sampleRateHz(); }
  bool isSimulated() const override { return inner_.isSimulated(); }

 private:
  ecg::IEcgSource& inner_;
  uint32_t last_us_ = 0;
  uint64_t elapsed_us_ = 0;
  uint64_t delivered_ = 0;
};

// Objetos estáticos: el pipeline guarda ~10 KB (buffer circular + latidos
// pendientes) y no debe vivir en la pila de loopTask.
ecg::TestPatternSource g_pattern(ECG_HW_SAMPLE_RATE_HZ, 75.0f, 7);
PacedSource g_source(g_pattern);
ecg::Esp32BleTransport g_ble;
ecg::StubInferenceEngine g_model(0);  // sin TFLM: no hay clasificación (E4)
ecg::EcgPipeline g_pipeline(g_source, g_ble, g_model);
char g_deviceId[ECG_DEVICE_ID_MAX_LEN + 1];

}  // namespace

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println();
  Serial.println("[cardioalert] E2: BLE NimBLE + fuente SIMULADA (sin AD8232, sin TFLM)");

  // MAC base de fábrica (eFuse), en orden de visualización: 80:F3:DA:60:3C:FC
  // -> id "3CFC" (ble_spec.md §2). OJO: la dirección BLE que ve un escáner es
  // la base + 2 (...:3C:FE); el nombre usa la MAC base documentada por esptool.
  uint8_t mac[6] = {};
  esp_efuse_mac_get_default(mac);
  if (!ecg::deviceIdFromMac(mac, g_deviceId, sizeof g_deviceId)) {
    Serial.println("[cardioalert] ERROR: no se pudo derivar el id del dispositivo");
    return;
  }
  Serial.printf("[cardioalert] MAC base %02X:%02X:%02X:%02X:%02X:%02X  id=%s\n", mac[0], mac[1], mac[2], mac[3],
                mac[4], mac[5], g_deviceId);

  const bool ok = g_pipeline.begin(g_deviceId);
  Serial.printf("[cardioalert] pipeline.begin=%s estado=%s error=%s\n", ok ? "OK" : "FALLO",
                ecg::stateName(g_pipeline.state()), ecg::errorName(g_pipeline.error()));
}

void loop() {
  g_ble.loopService();            // comandos CONTROL, en el contexto de loop()
  g_pipeline.poll(millis());      // muestras -> BLE
  delay(1);                       // cede CPU a las tareas BLE y al watchdog
}

#endif  // ARDUINO
