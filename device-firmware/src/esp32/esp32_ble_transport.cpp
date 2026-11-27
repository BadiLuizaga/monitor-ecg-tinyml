#if defined(ARDUINO)

#include "esp32_ble_transport.h"

namespace ecg {

void Esp32BleTransport::ServerCallbacks::onConnect(NimBLEServer*, NimBLEConnInfo&) {
  owner.mtu_.store(ECG_DEFAULT_MTU, std::memory_order_release);  // hasta que el central negocie
  owner.connected_.store(true, std::memory_order_release);
}

void Esp32BleTransport::ServerCallbacks::onDisconnect(NimBLEServer*, NimBLEConnInfo&, int) {
  owner.connected_.store(false, std::memory_order_release);
  owner.subEcg_.store(false, std::memory_order_relaxed);
  owner.subAlert_.store(false, std::memory_order_relaxed);
  owner.subStatus_.store(false, std::memory_order_relaxed);
  owner.mtu_.store(ECG_DEFAULT_MTU, std::memory_order_release);
  // La cola SPSC solo puede vaciarla el consumidor (loop()): se le avisa con un
  // contador (un solo escritor: load + store). Ver loopService().
  owner.disconnects_.store(owner.disconnects_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
  // El re-anuncio lo hace NimBLE (advertiseOnDisconnect, activado en begin()).
}

void Esp32BleTransport::ServerCallbacks::onMTUChange(uint16_t mtu, NimBLEConnInfo&) {
  owner.mtu_.store(mtu, std::memory_order_release);
}

// Solo copia: nunca ejecuta el ControlHandler en la tarea de NimBLE.
void Esp32BleTransport::CharCallbacks::onWrite(NimBLECharacteristic* c, NimBLEConnInfo&) {
  const NimBLEAttValue v = c->getValue();
  owner.onControlWritten(v.data(), v.size());
}

void Esp32BleTransport::CharCallbacks::onSubscribe(NimBLECharacteristic* c, NimBLEConnInfo&, uint16_t subValue) {
  std::atomic<bool>* flag = nullptr;
  if (c == owner.ecg_) flag = &owner.subEcg_;
  else if (c == owner.alert_) flag = &owner.subAlert_;
  else if (c == owner.status_) flag = &owner.subStatus_;
  if (flag) flag->store(subValue != 0, std::memory_order_release);
}

bool Esp32BleTransport::begin(const char* deviceName) {
  if (started_) return true;
  if (!NimBLEDevice::init(deviceName)) return false;
  NimBLEDevice::setMTU(247);  // MTU preferido; el central decide el acordado (onMTUChange)

  NimBLEServer* server = NimBLEDevice::createServer();
  server->setCallbacks(&serverCb_);
  server->advertiseOnDisconnect(true);

  NimBLEService* svc = server->createService(ECG_SERVICE_UUID);
  ecg_ = svc->createCharacteristic(ECG_CHAR_ECG_DATA_UUID, NIMBLE_PROPERTY::NOTIFY);
  alert_ = svc->createCharacteristic(ECG_CHAR_ALERT_UUID, NIMBLE_PROPERTY::NOTIFY | NIMBLE_PROPERTY::INDICATE);
  status_ = svc->createCharacteristic(ECG_CHAR_STATUS_UUID, NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY,
                                      ECG_JSON_MAX_READ_BYTES);
  info_ = svc->createCharacteristic(ECG_CHAR_DEVICE_INFO_UUID, NIMBLE_PROPERTY::READ, ECG_JSON_MAX_READ_BYTES);
  control_ = svc->createCharacteristic(ECG_CHAR_CONTROL_UUID, NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::WRITE_NR,
                                       ECG_JSON_MAX_BYTES);
  if (!ecg_ || !alert_ || !status_ || !info_ || !control_) return false;
  control_->setCallbacks(&charCb_);
  ecg_->setCallbacks(&charCb_);
  alert_->setCallbacks(&charCb_);
  status_->setCallbacks(&charCb_);
  if (!svc->start()) return false;

  // Anuncio: flags + Service UUID de 128 bits (la app filtra por servicio).
  // El nombre va en la respuesta al escaneo (ble_spec.md §2).
  NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
  adv->addServiceUUID(ECG_SERVICE_UUID);
  NimBLEAdvertisementData scan;
  scan.setName(deviceName);
  adv->setScanResponseData(scan);
  adv->enableScanResponse(true);
  if (!adv->start()) return false;
  started_ = true;
  return true;
}

NimBLECharacteristic* Esp32BleTransport::charOf(Characteristic c) const {
  switch (c) {
    case Characteristic::ECG_DATA: return ecg_;
    case Characteristic::ALERT: return alert_;
    case Characteristic::STATUS: return status_;
    case Characteristic::DEVICE_INFO: return info_;
    case Characteristic::CONTROL: return control_;
  }
  return nullptr;
}

std::atomic<bool>* Esp32BleTransport::subscribedOf(Characteristic c) {
  switch (c) {
    case Characteristic::ECG_DATA: return &subEcg_;
    case Characteristic::ALERT: return &subAlert_;
    case Characteristic::STATUS: return &subStatus_;
    default: return nullptr;
  }
}

bool Esp32BleTransport::notify(Characteristic c, const uint8_t* data, size_t len) {
  NimBLECharacteristic* ch = charOf(c);
  std::atomic<bool>* sub = subscribedOf(c);
  if (!ch || !sub || data == nullptr || len == 0) return false;
  if (!isConnected()) return false;
  // Sin suscripción no hay a quién notificar: no es un fallo de la cola.
  if (!sub->load(std::memory_order_acquire)) return true;
  if (len + ECG_ATT_OVERHEAD > mtu()) return false;
  return ch->notify(data, len);  // false: sin mbuf (cola de transmisión llena) o error
}

void Esp32BleTransport::setReadValue(Characteristic c, const uint8_t* data, size_t len) {
  NimBLECharacteristic* ch = charOf(c);
  if (ch && data) ch->setValue(data, len);
}

size_t Esp32BleTransport::loopService() {
  const uint32_t d = disconnects_.load(std::memory_order_acquire);
  if (d != disconnectsSeen_) {
    disconnectsSeen_ = d;
    discardPendingControl();  // comandos de la conexión anterior
  }
  return service();
}

}  // namespace ecg

#endif  // ARDUINO
