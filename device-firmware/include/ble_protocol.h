// ble_protocol.h - Codificación del protocolo BLE v1 (portable, sin dependencias).
//
// Implementación de referencia: communication/protocol/ecg_protocol.py.
// No usa memoria dinámica ni printf de coma flotante.
#pragma once

#include <cstddef>
#include <cstdint>

#include "integration_config.h"

namespace ecg {

// Deben coincidir con error_codes de protocol_config.json (lo verifica una prueba).
enum class ErrorCode : uint8_t {
  OK = 0,
  LEAD_OFF = 10,
  SIGNAL_SATURATED = 11,
  SIGNAL_NOISY = 12,
  SAMPLE_BUFFER_OVERFLOW = 20,
  BLE_TX_QUEUE_FULL = 21,
  MODEL_NOT_LOADED = 30,
  INFERENCE_FAILED = 31,
  CONFIG_MISMATCH = 32,  // fs del sensor distinta de la de filtros/modelo
  INVALID_COMMAND = 40,
  UNSUPPORTED_PROTOCOL_VERSION = 41,
  MTU_TOO_SMALL = 42,
  LOW_BATTERY = 50,
};

enum class DeviceState : uint8_t { BOOTING, ADVERTISING, CONNECTED, STREAMING, ERROR };

enum class Characteristic : uint8_t { ECG_DATA = 1, ALERT = 2, STATUS = 3, DEVICE_INFO = 4, CONTROL = 5 };

enum class Command : uint8_t { UNKNOWN, START_STREAM, STOP_STREAM, PING, GET_STATUS, ACK_ALERT };

const char* errorName(ErrorCode code);
const char* stateName(DeviceState state);

// Muestras int16 que caben en una notificación con el MTU dado (0 si no cabe ninguna).
size_t maxSamplesPerPacket(uint16_t mtu);

// Escribe un paquete ECG_DATA en `out`. Devuelve los bytes escritos o 0 si error.
size_t encodeEcgPacket(uint8_t* out, size_t capacity, uint16_t seq, uint32_t firstSampleIndex,
                       const int16_t* samples, uint8_t count, uint8_t flags);

// Mensajes JSON compactos. Devuelven la longitud escrita (sin '\0') o 0 si no cabe.
size_t buildStatusJson(char* out, size_t capacity, DeviceState state, ErrorCode error, uint32_t uptimeMs);
size_t buildAlertJson(char* out, size_t capacity, uint32_t eventId, const char* event, char beatClass,
                      float confidence, uint32_t sampleIndex, bool simulated);
size_t buildCmdAckJson(char* out, size_t capacity, int32_t cmdId, bool ok, ErrorCode error);

struct DeviceInfo {
  const char* deviceId;
  const char* fwVersion;
  const char* hwModel;
  float sampleRateHz;
  int adcBits;        // 0 = desconocido -> null
  float mvPerCount;   // 0 = sin calibrar -> null
  int leads;
  bool simulated;
};
size_t buildDeviceInfoJson(char* out, size_t capacity, const DeviceInfo& info);

struct ParsedCommand {
  Command cmd = Command::UNKNOWN;
  int32_t cmdId = -1;
  int32_t eventId = -1;
  bool versionOk = false;
};

// Parser mínimo para los comandos de CONTROL (no es un parser JSON general:
// solo extrae "v", "cmd", "cmd_id" y "event_id"). Devuelve false si el
// mensaje no es un comando válido.
bool parseCommand(const char* json, size_t len, ParsedCommand* out);

// Interfaz hacia la pila BLE concreta (NimBLE/Bluedroid en ESP32, mock en pruebas).
class IBleTransport {
 public:
  using ControlHandler = void (*)(const uint8_t* data, size_t len, void* ctx);
  virtual ~IBleTransport() = default;
  virtual bool begin(const char* deviceName) = 0;
  virtual bool isConnected() const = 0;
  virtual uint16_t mtu() const = 0;
  virtual bool notify(Characteristic c, const uint8_t* data, size_t len) = 0;
  // Valor servido en lecturas de DEVICE_INFO / STATUS.
  virtual void setReadValue(Characteristic c, const uint8_t* data, size_t len) = 0;
  virtual void setControlHandler(ControlHandler handler, void* ctx) = 0;
};

// Agrupa muestras en paquetes ECG_DATA según el MTU y los envía.
class EcgPacketizer {
 public:
  void reset();
  void setMtu(uint16_t mtu);
  // Añade una muestra; envía el paquete cuando se llena. false si el envío falló.
  bool push(IBleTransport& tx, uint32_t sampleIndex, int16_t sample, uint8_t flags);
  bool flush(IBleTransport& tx);
  uint16_t seq() const { return seq_; }
  uint32_t droppedPackets() const { return dropped_; }

 private:
  int16_t buf_[ECG_MAX_SAMPLES_PER_PACKET] = {};
  uint8_t count_ = 0;
  uint8_t capacity_ = 5;
  uint8_t flags_ = 0;
  uint16_t seq_ = 0;
  uint32_t firstIndex_ = 0;
  uint32_t dropped_ = 0;
};

}  // namespace ecg
