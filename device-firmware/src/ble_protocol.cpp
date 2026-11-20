#include "ble_protocol.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ecg {

const char* errorName(ErrorCode code) {
  switch (code) {
    case ErrorCode::OK: return "OK";
    case ErrorCode::LEAD_OFF: return "LEAD_OFF";
    case ErrorCode::SIGNAL_SATURATED: return "SIGNAL_SATURATED";
    case ErrorCode::SIGNAL_NOISY: return "SIGNAL_NOISY";
    case ErrorCode::SAMPLE_BUFFER_OVERFLOW: return "SAMPLE_BUFFER_OVERFLOW";
    case ErrorCode::BLE_TX_QUEUE_FULL: return "BLE_TX_QUEUE_FULL";
    case ErrorCode::MODEL_NOT_LOADED: return "MODEL_NOT_LOADED";
    case ErrorCode::INFERENCE_FAILED: return "INFERENCE_FAILED";
    case ErrorCode::CONFIG_MISMATCH: return "CONFIG_MISMATCH";
    case ErrorCode::INVALID_COMMAND: return "INVALID_COMMAND";
    case ErrorCode::UNSUPPORTED_PROTOCOL_VERSION: return "UNSUPPORTED_PROTOCOL_VERSION";
    case ErrorCode::MTU_TOO_SMALL: return "MTU_TOO_SMALL";
    case ErrorCode::LOW_BATTERY: return "LOW_BATTERY";
  }
  return "UNKNOWN";
}

const char* stateName(DeviceState state) {
  switch (state) {
    case DeviceState::BOOTING: return "booting";
    case DeviceState::ADVERTISING: return "advertising";
    case DeviceState::CONNECTED: return "connected";
    case DeviceState::STREAMING: return "streaming";
    case DeviceState::ERROR: return "error";
  }
  return "error";
}

size_t maxSamplesPerPacket(uint16_t mtu) {
  const int payload = static_cast<int>(mtu) - ECG_ATT_OVERHEAD - ECG_PACKET_HEADER_SIZE;
  if (payload < 2) return 0;
  const size_t n = static_cast<size_t>(payload) / 2;
  return n > ECG_MAX_SAMPLES_PER_PACKET ? ECG_MAX_SAMPLES_PER_PACKET : n;
}

static void putU16(uint8_t* p, uint16_t v) {
  p[0] = static_cast<uint8_t>(v & 0xFF);
  p[1] = static_cast<uint8_t>(v >> 8);
}

static void putU32(uint8_t* p, uint32_t v) {
  for (int i = 0; i < 4; ++i) p[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xFF);
}

size_t encodeEcgPacket(uint8_t* out, size_t capacity, uint16_t seq, uint32_t firstSampleIndex,
                       const int16_t* samples, uint8_t count, uint8_t flags) {
  const size_t total = ECG_PACKET_HEADER_SIZE + 2u * count;
  if (count == 0 || out == nullptr || capacity < total) return 0;
  out[0] = ECG_PROTOCOL_VERSION;
  out[1] = 1;  // ECG_SAMPLES
  putU16(out + 2, seq);
  putU32(out + 4, firstSampleIndex);
  out[8] = count;
  out[9] = flags;
  for (uint8_t i = 0; i < count; ++i) putU16(out + ECG_PACKET_HEADER_SIZE + 2 * i, static_cast<uint16_t>(samples[i]));
  return total;
}

// snprintf con comprobación de truncado.
template <typename... Args>
static size_t fmt(char* out, size_t capacity, const char* f, Args... args) {
  const int n = std::snprintf(out, capacity, f, args...);
  return (n < 0 || static_cast<size_t>(n) >= capacity) ? 0 : static_cast<size_t>(n);
}

// Confianza con 3 decimales sin printf de float (newlib-nano no lo soporta).
static void formatUnit(char* buf, size_t cap, float v) {
  if (!(v > 0.0f)) v = 0.0f;
  if (v > 1.0f) v = 1.0f;
  const int milli = static_cast<int>(v * 1000.0f + 0.5f);
  if (milli >= 1000) std::snprintf(buf, cap, "1.000");
  else std::snprintf(buf, cap, "0.%03d", milli);
}

size_t buildStatusJson(char* out, size_t capacity, DeviceState state, ErrorCode error, uint32_t uptimeMs) {
  return fmt(out, capacity, "{\"v\":%d,\"type\":\"status\",\"state\":\"%s\",\"error_code\":%d,\"error\":\"%s\",\"uptime_ms\":%lu}",
             ECG_PROTOCOL_VERSION, stateName(state), static_cast<int>(error), errorName(error),
             static_cast<unsigned long>(uptimeMs));
}

size_t buildAlertJson(char* out, size_t capacity, uint32_t eventId, const char* event, char beatClass,
                      float confidence, uint32_t sampleIndex, bool simulated) {
  char conf[8];
  formatUnit(conf, sizeof conf, confidence);
  return fmt(out, capacity,
             "{\"v\":%d,\"type\":\"alert\",\"event_id\":%lu,\"event\":\"%s\",\"beat_class\":\"%c\","
             "\"confidence\":%s,\"sample_index\":%lu,\"simulated\":%s}",
             ECG_PROTOCOL_VERSION, static_cast<unsigned long>(eventId), event, beatClass ? beatClass : '-', conf,
             static_cast<unsigned long>(sampleIndex), simulated ? "true" : "false");
}

size_t buildCmdAckJson(char* out, size_t capacity, int32_t cmdId, bool ok, ErrorCode error) {
  return fmt(out, capacity, "{\"v\":%d,\"type\":\"cmd_ack\",\"cmd_id\":%ld,\"ok\":%s,\"error_code\":%d}",
             ECG_PROTOCOL_VERSION, static_cast<long>(cmdId), ok ? "true" : "false", static_cast<int>(error));
}

size_t buildDeviceInfoJson(char* out, size_t capacity, const DeviceInfo& info) {
  char adc[12];
  if (info.adcBits > 0) std::snprintf(adc, sizeof adc, "%d", info.adcBits);
  else std::snprintf(adc, sizeof adc, "null");
  // mv_per_count en notación científica entera: mantisa * 1e-9 (sin printf de float).
  char mv[24];
  if (info.mvPerCount > 0.0f) {
    std::snprintf(mv, sizeof mv, "%lue-9", static_cast<unsigned long>(info.mvPerCount * 1e9f + 0.5f));
  } else {
    std::snprintf(mv, sizeof mv, "null");
  }
  const unsigned long fsMilli = static_cast<unsigned long>(info.sampleRateHz * 1000.0f + 0.5f);
  return fmt(out, capacity,
             "{\"v\":%d,\"type\":\"device_info\",\"device_id\":\"%s\",\"fw_version\":\"%s\",\"hw_model\":\"%s\","
             "\"sample_rate_hz\":%lu.%03lu,\"sample_format\":\"int16\",\"adc_bits\":%s,\"mv_per_count\":%s,"
             "\"leads\":%d,\"simulated\":%s}",
             ECG_PROTOCOL_VERSION, info.deviceId, info.fwVersion, info.hwModel, fsMilli / 1000, fsMilli % 1000, adc,
             mv, info.leads, info.simulated ? "true" : "false");
}

// Busca "key": y devuelve el puntero al valor (tras espacios) o nullptr.
static const char* findValue(const char* json, size_t len, const char* key) {
  char pattern[24];
  const int plen = std::snprintf(pattern, sizeof pattern, "\"%s\"", key);
  if (plen <= 0) return nullptr;
  for (size_t i = 0; i + static_cast<size_t>(plen) <= len; ++i) {
    if (std::memcmp(json + i, pattern, static_cast<size_t>(plen)) != 0) continue;
    size_t j = i + static_cast<size_t>(plen);
    while (j < len && (json[j] == ' ' || json[j] == '\t')) ++j;
    if (j >= len || json[j] != ':') continue;
    ++j;
    while (j < len && (json[j] == ' ' || json[j] == '\t')) ++j;
    return j < len ? json + j : nullptr;
  }
  return nullptr;
}

static bool readInt(const char* p, const char* end, int32_t* out) {
  if (p == nullptr) return false;
  char buf[16];
  size_t n = 0;
  while (p < end && n < sizeof buf - 1 && (*p == '-' || (*p >= '0' && *p <= '9'))) buf[n++] = *p++;
  if (n == 0 || (n == 1 && buf[0] == '-')) return false;
  buf[n] = '\0';
  *out = static_cast<int32_t>(std::strtol(buf, nullptr, 10));
  return true;
}

bool parseCommand(const char* json, size_t len, ParsedCommand* out) {
  if (json == nullptr || out == nullptr) return false;
  *out = ParsedCommand{};
  const char* end = json + len;
  int32_t v = 0;
  out->versionOk = readInt(findValue(json, len, "v"), end, &v) && v == ECG_PROTOCOL_VERSION;
  const char* type = findValue(json, len, "type");
  if (type == nullptr || static_cast<size_t>(end - type) < 5 || std::memcmp(type, "\"cmd\"", 5) != 0) return false;
  if (!readInt(findValue(json, len, "cmd_id"), end, &out->cmdId)) return false;
  readInt(findValue(json, len, "event_id"), end, &out->eventId);
  const char* c = findValue(json, len, "cmd");
  if (c == nullptr || *c != '"') return false;
  ++c;
  const char* q = static_cast<const char*>(std::memchr(c, '"', static_cast<size_t>(end - c)));
  if (q == nullptr) return false;
  const size_t n = static_cast<size_t>(q - c);
  struct { const char* name; Command cmd; } table[] = {
      {"start_stream", Command::START_STREAM}, {"stop_stream", Command::STOP_STREAM}, {"ping", Command::PING},
      {"get_status", Command::GET_STATUS},     {"ack_alert", Command::ACK_ALERT},
  };
  for (const auto& e : table) {
    if (std::strlen(e.name) == n && std::memcmp(e.name, c, n) == 0) out->cmd = e.cmd;
  }
  return out->versionOk && out->cmd != Command::UNKNOWN;
}

// --- EcgPacketizer ---------------------------------------------------------

void EcgPacketizer::reset() {
  count_ = 0;
  flags_ = 0;
  seq_ = 0;
  dropped_ = 0;
}

void EcgPacketizer::setMtu(uint16_t mtu) {
  const size_t n = maxSamplesPerPacket(mtu);
  capacity_ = static_cast<uint8_t>(n == 0 ? 1 : n);
}

bool EcgPacketizer::push(IBleTransport& tx, uint32_t sampleIndex, int16_t sample, uint8_t flags) {
  if (count_ == 0) firstIndex_ = sampleIndex;
  buf_[count_++] = sample;
  flags_ |= flags;
  return count_ >= capacity_ ? flush(tx) : true;
}

bool EcgPacketizer::flush(IBleTransport& tx) {
  if (count_ == 0) return true;
  uint8_t pkt[ECG_PACKET_HEADER_SIZE + 2 * ECG_MAX_SAMPLES_PER_PACKET];
  const size_t n = encodeEcgPacket(pkt, sizeof pkt, seq_, firstIndex_, buf_, count_, flags_);
  // seq avanza aunque falle el envío: la app detecta el hueco como pérdida.
  seq_ = static_cast<uint16_t>(seq_ + 1);
  count_ = 0;
  flags_ = 0;
  const bool ok = n > 0 && tx.notify(Characteristic::ECG_DATA, pkt, n);
  if (!ok) ++dropped_;
  return ok;
}

}  // namespace ecg
