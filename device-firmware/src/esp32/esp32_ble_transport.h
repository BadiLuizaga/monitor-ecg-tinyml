// esp32_ble_transport.h - Transporte BLE del ESP32 con NimBLE-Arduino 2.x (E2).
//
// Hereda de QueuedBleTransport (ble_transport.h): solo implementa lo que
// depende de la pila. Contrato (ble_protocol.h, IBleTransport):
//  * notify() devuelve false si no se pudo encolar (mbuf agotados / cola llena).
//  * Los comandos de CONTROL NUNCA se ejecutan en el callback de NimBLE: el
//    callback onWrite solo llama a onControlWritten() (copia en la cola SPSC) y
//    loop() los procesa con loopService() antes de EcgPipeline::poll().
// UUID y protocolo: integration_config.h / ble_spec.md (sin cambios, v1).
//
// Solo se compila con el framework Arduino (PlatformIO); en PC no existe.
#pragma once

#if defined(ARDUINO)

#include <atomic>

#include <NimBLEDevice.h>

#include "ble_transport.h"

namespace ecg {

class Esp32BleTransport : public QueuedBleTransport {
 public:
  bool begin(const char* deviceName) override;
  bool isConnected() const override { return connected_.load(std::memory_order_acquire); }
  uint16_t mtu() const override { return mtu_.load(std::memory_order_acquire); }
  bool notify(Characteristic c, const uint8_t* data, size_t len) override;
  void setReadValue(Characteristic c, const uint8_t* data, size_t len) override;

  // Llamar desde loop(), antes de EcgPipeline::poll(): descarta los comandos
  // pendientes si hubo una desconexión y entrega los de CONTROL al pipeline.
  size_t loopService();

 private:
  struct ServerCallbacks : NimBLEServerCallbacks {
    explicit ServerCallbacks(Esp32BleTransport& o) : owner(o) {}
    void onConnect(NimBLEServer*, NimBLEConnInfo&) override;
    void onDisconnect(NimBLEServer*, NimBLEConnInfo&, int reason) override;
    void onMTUChange(uint16_t mtu, NimBLEConnInfo&) override;
    Esp32BleTransport& owner;
  };
  struct CharCallbacks : NimBLECharacteristicCallbacks {
    explicit CharCallbacks(Esp32BleTransport& o) : owner(o) {}
    void onWrite(NimBLECharacteristic*, NimBLEConnInfo&) override;
    void onSubscribe(NimBLECharacteristic*, NimBLEConnInfo&, uint16_t subValue) override;
    Esp32BleTransport& owner;
  };

  NimBLECharacteristic* charOf(Characteristic c) const;
  std::atomic<bool>* subscribedOf(Characteristic c);

  ServerCallbacks serverCb_{*this};
  CharCallbacks charCb_{*this};
  NimBLECharacteristic* ecg_ = nullptr;
  NimBLECharacteristic* alert_ = nullptr;
  NimBLECharacteristic* status_ = nullptr;
  NimBLECharacteristic* info_ = nullptr;
  NimBLECharacteristic* control_ = nullptr;

  // Escritos por la tarea de NimBLE (un solo escritor), leídos por loop():
  // solo load/store, sin operaciones RMW (misma política que ControlQueue).
  std::atomic<bool> connected_{false};
  std::atomic<uint32_t> disconnects_{0};
  uint32_t disconnectsSeen_ = 0;  // solo loop()
  std::atomic<uint16_t> mtu_{ECG_DEFAULT_MTU};
  std::atomic<bool> subEcg_{false}, subAlert_{false}, subStatus_{false};
  bool started_ = false;
};

}  // namespace ecg

#endif  // ARDUINO
