// main.cpp - Punto de entrada del firmware.
//
// * En ESP32 (ARDUINO o ESP_PLATFORM): solo orquesta. La fuente de señal real
//   y el transporte BLE concreto están PENDIENTES (ver docs/integration.md);
//   la compilación se detiene con #error hasta que se implementen, para que
//   nadie despliegue por error un firmware que solo transmite datos simulados.
//   PlatformIO EXCLUYE este archivo (build_src_filter en platformio.ini): el
//   punto de entrada del ESP32 será src/esp32/main_esp32.cpp (etapa E2).
// * En PC (host): demo con la fuente simulada y un transporte en memoria.
//   No hay BLE real: imprime lo que se habría notificado.

#include <cstdio>

#include "ble_transport.h"
#include "ecg_pipeline.h"
#include "hardware_config.h"

#if defined(ARDUINO) || defined(ESP_PLATFORM)

#ifndef ECG_ESP32_PORT_READY
#error "Falta el port a ESP32: implementar IBleTransport (NimBLE/Bluedroid) y HardwareEcgSource, y definir ECG_ESP32_PORT_READY. Ver docs/integration.md"
#endif

// TODO(integración): sustituir por las implementaciones reales.
//   static Esp32BleTransport ble;                       // QueuedBleTransport + NimBLE
//   static HardwareEcgSource source;                    // IEcgSource (cuando ECG_HW_CONFIG_CONFIRMED)
//   static ecg::TflmInferenceEngine model;              // con -DECG_HAVE_TFLM=1
//   static ecg::EcgPipeline pipeline(source, ble, model);
//   static char deviceId[5];
//   void setup() { ecg::deviceIdFromMac(<MAC BLE en orden de visualización>, deviceId, sizeof deviceId);
//                  pipeline.begin(deviceId); }
//   void loop()  { ble.service(); pipeline.poll(millis()); }  // comandos en el contexto de loop()

#else  // ---------------------------- demo en PC ----------------------------

#include <cstring>
#include <vector>

namespace {

class ConsoleTransport : public ecg::QueuedBleTransport {
 public:
  bool begin(const char* name) override {
    std::printf("[demo] anunciando como %s (SIMULADO, sin radio BLE)\n", name);
    return true;
  }
  bool isConnected() const override { return connected; }
  uint16_t mtu() const override { return mtuValue; }
  bool notify(ecg::Characteristic c, const uint8_t* d, size_t n) override {
    if (c == ecg::Characteristic::ECG_DATA) {
      ++ecgPackets;
    } else {
      std::printf("[demo] notify %-6s %.*s\n", c == ecg::Characteristic::ALERT ? "ALERT" : "STATUS",
                  static_cast<int>(n), reinterpret_cast<const char*>(d));
    }
    return true;
  }
  void setReadValue(ecg::Characteristic c, const uint8_t* d, size_t n) override {
    if (c == ecg::Characteristic::DEVICE_INFO)
      std::printf("[demo] DEVICE_INFO %.*s\n", static_cast<int>(n), reinterpret_cast<const char*>(d));
  }
  // Como el callback onWrite de la pila: solo encola; loop() llama a service().
  void write(const char* json) { onControlWritten(reinterpret_cast<const uint8_t*>(json), std::strlen(json)); }

  bool connected = false;
  uint16_t mtuValue = 247;
  unsigned ecgPackets = 0;
};

}  // namespace

int main() {
  ecg::TestPatternSource source(ECG_HW_SAMPLE_RATE_HZ, 75.0f, 7);
  ConsoleTransport ble;
  ecg::StubInferenceEngine model(0);  // sin TFLM en PC: no hay clasificación
  ecg::EcgPipeline pipeline(source, ble, model);
  pipeline.begin("DEMO");

  ble.connected = true;
  pipeline.poll(0);
  ble.write("{\"v\":1,\"type\":\"cmd\",\"cmd\":\"start_stream\",\"cmd_id\":1}");
  const int seconds = 10;
  const int per_poll = 64;  // EcgPipeline::poll procesa hasta 64 muestras por llamada
  const int polls = static_cast<int>(seconds * ECG_HW_SAMPLE_RATE_HZ) / per_poll;
  for (int i = 0; i < polls; ++i) {  // mismo orden que loop() en el ESP32
    ble.service();
    pipeline.poll(static_cast<uint32_t>(i * 100));
  }
  ble.connected = false;
  pipeline.poll(seconds * 1000);

  const ecg::PipelineStats& s = pipeline.stats();
  std::printf("[demo] muestras=%u latidos=%u clasificados=%u paquetes ECG=%u error=%s\n", s.samples,
              s.beatsDetected, s.beatsClassified, ble.ecgPackets, ecg::errorName(pipeline.error()));
  return 0;
}

#endif
