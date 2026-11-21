// test_host.cpp - Pruebas de la lógica del firmware en PC (sin ESP32 ni BLE).
//
// Compilar y ejecutar con device-firmware/test/host/run_host_tests.sh
// (requiere g++ o clang++ con C++17).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ble_protocol.h"
#include "ble_transport.h"
#include "ecg_pipeline.h"
#include "hardware_config.h"
#include "signal_processing.h"
#include "test_vectors.h"

using namespace ecg;

static int g_failures = 0;
#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::printf("  FALLO %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
      ++g_failures;                                                     \
    }                                                                   \
  } while (0)

static std::string hex(const uint8_t* d, size_t n) {
  std::string s;
  char b[3];
  for (size_t i = 0; i < n; ++i) {
    std::snprintf(b, sizeof b, "%02x", d[i]);
    s += b;
  }
  return s;
}

// Transporte simulado sobre QueuedBleTransport: las escrituras de CONTROL
// pasan por la misma cola SPSC que usará Esp32BleTransport.
class MockTransport : public QueuedBleTransport {
 public:
  bool begin(const char* name) override {
    advertisedName = name;
    return true;
  }
  bool isConnected() const override { return connected; }
  uint16_t mtu() const override { return mtuValue; }
  // Solo se registra lo ENTREGADO; con failNotify la "cola de la pila" está llena.
  bool notify(Characteristic c, const uint8_t* d, size_t n) override {
    ++notifyAttempts;
    if (inStackCallback) ++notifiesInsideCallback;
    if (failNotify) return false;
    if (c == Characteristic::ECG_DATA) packets.emplace_back(d, d + n);
    else json.emplace_back(reinterpret_cast<const char*>(d), n);
    return true;
  }
  void setReadValue(Characteristic c, const uint8_t* d, size_t n) override {
    if (c == Characteristic::DEVICE_INFO) deviceInfo.assign(reinterpret_cast<const char*>(d), n);
    if (c == Characteristic::STATUS) statusRead.assign(reinterpret_cast<const char*>(d), n);
  }
  // Lo que haría el callback onWrite de la pila BLE: solo encolar.
  bool stackWrite(const char* s) { return stackWrite(reinterpret_cast<const uint8_t*>(s), std::strlen(s)); }
  bool stackWrite(const uint8_t* d, size_t n) {
    inStackCallback = true;
    const bool ok = onControlWritten(d, n);
    inStackCallback = false;
    return ok;
  }
  // Escritura + service() inmediato (lo que ocurre en un loop() completo).
  void write(const char* s) {
    stackWrite(s);
    service();
  }
  bool has(const char* needle) const {
    for (const auto& j : json)
      if (j.find(needle) != std::string::npos) return true;
    return false;
  }
  size_t count(const char* needle) const {
    size_t k = 0;
    for (const auto& j : json) k += j.find(needle) != std::string::npos;
    return k;
  }

  bool connected = false, failNotify = false, inStackCallback = false;
  uint16_t mtuValue = 247;
  unsigned notifyAttempts = 0, notifiesInsideCallback = 0;
  std::vector<std::vector<uint8_t>> packets;
  std::vector<std::string> json;
  std::string deviceInfo, statusRead, advertisedName;
};

// Igual que communication/tests/test_protocol.py::test_packet_golden_bytes
static void testGoldenPacket() {
  std::printf("paquete binario (vector de referencia Python)\n");
  const int16_t s[] = {1, -2, 300};
  uint8_t out[32];
  const size_t n = encodeEcgPacket(out, sizeof out, 0x0102, 0x0A0B0C0D, s, 3, 0x04);
  CHECK(n == 16);
  CHECK(hex(out, n) == "010102010d0c0b0a03040100feff2c01");
  CHECK(encodeEcgPacket(out, 10, 0, 0, s, 3, 0) == 0);  // sin espacio
  CHECK(encodeEcgPacket(out, sizeof out, 0, 0, s, 0, 0) == 0);
}

static void testMtu() {
  std::printf("muestras por MTU\n");
  CHECK(maxSamplesPerPacket(23) == 5);
  CHECK(maxSamplesPerPacket(185) == 86);
  CHECK(maxSamplesPerPacket(247) == 117);
  CHECK(maxSamplesPerPacket(517) == 252);
  CHECK(maxSamplesPerPacket(14) == 0);
}

static void testJson() {
  std::printf("mensajes JSON\n");
  char b[ECG_JSON_MAX_BYTES + 1];
  buildStatusJson(b, sizeof b, DeviceState::STREAMING, ErrorCode::LEAD_OFF, 60500);
  CHECK(std::string(b) ==
        "{\"v\":1,\"type\":\"status\",\"state\":\"streaming\",\"error_code\":10,\"error\":\"LEAD_OFF\",\"uptime_ms\":60500}");
  buildAlertJson(b, sizeof b, 42, "arrhythmia_suspected", 'V', 0.91f, 123456, false);
  CHECK(std::string(b) ==
        "{\"v\":1,\"type\":\"alert\",\"event_id\":42,\"event\":\"arrhythmia_suspected\",\"beat_class\":\"V\","
        "\"confidence\":0.910,\"sample_index\":123456,\"simulated\":false}");
  buildAlertJson(b, sizeof b, 1, "lead_off", 0, 1.7f, 0, true);
  CHECK(std::strstr(b, "\"beat_class\":\"-\"") && std::strstr(b, "\"confidence\":1.000"));
  buildCmdAckJson(b, sizeof b, -1, false, ErrorCode::INVALID_COMMAND);
  CHECK(std::string(b) == "{\"v\":1,\"type\":\"cmd_ack\",\"cmd_id\":-1,\"ok\":false,\"error_code\":40}");
  char small[20];
  CHECK(buildStatusJson(small, sizeof small, DeviceState::BOOTING, ErrorCode::OK, 0) == 0);  // truncado -> 0

  char info[ECG_JSON_MAX_READ_BYTES + 1];
  DeviceInfo di{"1A2B", "0.1.0", "TBD", 360.0f, 0, 0.0f, 1, true};
  const size_t n = buildDeviceInfoJson(info, sizeof info, di);
  CHECK(n > 0 && n <= ECG_JSON_MAX_READ_BYTES);
  CHECK(std::strstr(info, "\"adc_bits\":null,\"mv_per_count\":null") != nullptr);
  CHECK(std::strstr(info, "\"sample_rate_hz\":360.000") != nullptr);
}

static void testParseCommand() {
  std::printf("parser de comandos\n");
  ParsedCommand c;
  const char* ok = "{\"v\":1,\"type\":\"cmd\",\"cmd\":\"ack_alert\",\"cmd_id\":7,\"event_id\":42}";
  CHECK(parseCommand(ok, std::strlen(ok), &c) && c.cmd == Command::ACK_ALERT && c.cmdId == 7 && c.eventId == 42);
  const char* spaced = "{ \"v\" : 1, \"type\" : \"cmd\", \"cmd\" : \"ping\", \"cmd_id\" : 3 }";
  CHECK(parseCommand(spaced, std::strlen(spaced), &c) && c.cmd == Command::PING);
  const char* badv = "{\"v\":2,\"type\":\"cmd\",\"cmd\":\"ping\",\"cmd_id\":3}";
  CHECK(!parseCommand(badv, std::strlen(badv), &c) && c.hasVersion && !c.versionOk);
  const char* unknown = "{\"v\":1,\"type\":\"cmd\",\"cmd\":\"format\",\"cmd_id\":3}";
  CHECK(!parseCommand(unknown, std::strlen(unknown), &c) && c.versionOk);
  CHECK(!parseCommand("basura", 6, &c) && !c.hasVersion);
  CHECK(!parseCommand("", 0, &c) && !c.hasVersion && c.cmdId == -1);
}

static void testDeviceName() {
  std::printf("nombre del dispositivo (MAC / id externo)\n");
  const uint8_t mac[6] = {0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF};  // orden de visualización
  char id[5];
  CHECK(deviceIdFromMac(mac, id, sizeof id) && std::string(id) == "EEFF");
  const uint8_t mac2[6] = {0, 0, 0, 0, 0x0a, 0x1b};
  CHECK(deviceIdFromMac(mac2, id, sizeof id) && std::string(id) == "0A1B");
  char tiny[4];
  CHECK(!deviceIdFromMac(mac, tiny, sizeof tiny));
  char name[ECG_DEVICE_NAME_MAX_LEN + 1];
  CHECK(buildDeviceName(name, sizeof name, "EEFF") == 15 && std::string(name) == "ECG-TINYML-EEFF");
  CHECK(buildDeviceName(name, sizeof name, "DEMO") > 0);           // id externo (sin hardware)
  CHECK(buildDeviceName(name, sizeof name, "") == 0);
  CHECK(buildDeviceName(name, sizeof name, nullptr) == 0);
  CHECK(buildDeviceName(name, sizeof name, "123456789") == 0);     // > ECG_DEVICE_ID_MAX_LEN
  CHECK(buildDeviceName(name, 10, "EEFF") == 0);                   // no cabe

  TestPatternSource src(ECG_HW_SAMPLE_RATE_HZ, 75.0f, 1);
  MockTransport ble;
  StubInferenceEngine model(0);
  EcgPipeline p(src, ble, model);
  CHECK(p.begin(id));  // id derivado de una MAC (aquí, de prueba)
  CHECK(ble.advertisedName == "ECG-TINYML-0A1B");
  CHECK(ble.deviceInfo.find("\"device_id\":\"0A1B\"") != std::string::npos);

  MockTransport ble2;
  EcgPipeline p2(src, ble2, model);
  CHECK(!p2.begin("") && p2.state() == DeviceState::ERROR && ble2.advertisedName.empty());  // no se anuncia
}

static void testControlQueue() {
  std::printf("cola SPSC de CONTROL (%u huecos)\n", static_cast<unsigned>(ControlQueue::kSlots));
  ControlQueue q;
  uint8_t out[ControlQueue::kMaxLen];
  size_t len = 0;
  bool big = false;
  CHECK(q.empty() && !q.pop(out, &len, &big));
  for (size_t i = 0; i < ControlQueue::kSlots; ++i) {
    const uint8_t b[2] = {static_cast<uint8_t>(i), 0x55};
    CHECK(q.push(b, sizeof b));
  }
  const uint8_t extra[1] = {0xEE};
  CHECK(!q.push(extra, 1) && q.droppedFull() == 1);  // llena: se descarta la NUEVA
  CHECK(q.size() == ControlQueue::kSlots);
  for (size_t i = 0; i < ControlQueue::kSlots; ++i) {  // FIFO
    CHECK(q.pop(out, &len, &big) && len == 2 && !big && out[0] == i && out[1] == 0x55);
  }
  CHECK(q.empty());

  std::vector<uint8_t> huge(ControlQueue::kMaxLen + 1, '{');
  CHECK(q.push(huge.data(), huge.size()) && q.oversize() == 1);
  CHECK(q.pop(out, &len, &big) && big && len == 0);
  std::vector<uint8_t> exact(ControlQueue::kMaxLen, 'x');  // 180 B: se admite
  CHECK(q.push(exact.data(), exact.size()) && q.pop(out, &len, &big) && !big && len == ControlQueue::kMaxLen);

  // Vuelta completa de los índices y clear() desde el consumidor.
  bool fifo = true;
  for (uint32_t i = 0; i < 1000; ++i) {
    const uint8_t b[1] = {static_cast<uint8_t>(i)};
    q.push(b, 1);
    fifo = fifo && q.pop(out, &len, &big) && len == 1 && out[0] == static_cast<uint8_t>(i);
  }
  CHECK(fifo);
  q.push(extra, 1);
  q.push(extra, 1);
  q.clear();
  CHECK(q.empty() && !q.pop(out, &len, &big));
}

static const char* kStart = "{\"v\":1,\"type\":\"cmd\",\"cmd\":\"start_stream\",\"cmd_id\":1}";

static void testDeferredControl() {
  std::printf("CONTROL diferido: callback -> cola -> service() -> pipeline\n");
  TestPatternSource src(ECG_HW_SAMPLE_RATE_HZ, 75.0f, 5);
  MockTransport ble;
  StubInferenceEngine model(0);
  EcgPipeline p(src, ble, model);
  CHECK(p.begin("TEST"));
  ble.connected = true;
  p.poll(0);
  CHECK(p.state() == DeviceState::CONNECTED);

  // 1. El callback solo encola: ni cambia el estado ni responde nada.
  const size_t before = ble.json.size();
  const unsigned attempts = ble.notifyAttempts;
  CHECK(ble.stackWrite(kStart));
  CHECK(p.state() == DeviceState::CONNECTED);
  CHECK(ble.json.size() == before && ble.notifyAttempts == attempts && ble.notifiesInsideCallback == 0);
  CHECK(ble.controlQueue().size() == 1);
  // 2. service() (desde loop()) entrega el comando.
  CHECK(ble.service() == 1 && !ble.dispatching());
  CHECK(p.state() == DeviceState::STREAMING && ble.has("\"cmd_id\":1,\"ok\":true"));
  CHECK(ble.notifiesInsideCallback == 0);

  // 3. Cola llena: se procesan los 4 primeros en orden; el 5.º se pierde sin acuse.
  for (int i = 10; i < 10 + static_cast<int>(ControlQueue::kSlots) + 1; ++i) {
    char cmd[80];
    std::snprintf(cmd, sizeof cmd, "{\"v\":1,\"type\":\"cmd\",\"cmd\":\"ping\",\"cmd_id\":%d}", i);
    const bool queued = ble.stackWrite(cmd);
    CHECK(queued == (i < 10 + static_cast<int>(ControlQueue::kSlots)));
  }
  CHECK(ble.controlQueue().droppedFull() == 1);
  CHECK(ble.service(2) == 2);  // service() respeta el máximo por llamada
  CHECK(ble.service() == ControlQueue::kSlots - 2);
  size_t last = 0;
  bool ordered = true;
  for (int i = 10; i < 10 + static_cast<int>(ControlQueue::kSlots); ++i) {
    char ack[40];
    std::snprintf(ack, sizeof ack, "\"cmd_id\":%d,\"ok\":true", i);
    size_t at = 0;
    for (; at < ble.json.size() && ble.json[at].find(ack) == std::string::npos; ++at) {
    }
    ordered = ordered && at < ble.json.size() && at >= last;
    last = at;
  }
  CHECK(ordered);
  CHECK(!ble.has("\"cmd_id\":14"));

  // 4. Errores de interpretación coherentes con ble_spec.md (§5.4, §8).
  ble.write("basura");
  CHECK(ble.has("{\"v\":1,\"type\":\"cmd_ack\",\"cmd_id\":-1,\"ok\":false,\"error_code\":40}"));
  ble.json.clear();
  ble.write("{\"v\":2,\"type\":\"cmd\",\"cmd\":\"ping\",\"cmd_id\":5}");
  CHECK(ble.has("\"cmd_id\":5,\"ok\":false,\"error_code\":41"));
  ble.json.clear();
  std::string huge = "{\"v\":1,\"type\":\"cmd\",\"cmd\":\"ping\",\"cmd_id\":6,\"pad\":\"";
  huge += std::string(200, 'x') + "\"}";
  ble.stackWrite(reinterpret_cast<const uint8_t*>(huge.data()), huge.size());
  ble.service();
  CHECK(ble.has("\"cmd_id\":-1,\"ok\":false,\"error_code\":40") && !ble.has("\"cmd_id\":6"));

  // 5. Sin conexión, las escrituras se ignoran (sin acuse, sin cambio de estado).
  ble.connected = false;
  p.poll(1000);
  CHECK(p.state() == DeviceState::ADVERTISING);
  ble.json.clear();
  ble.write(kStart);
  CHECK(ble.json.empty() && p.state() == DeviceState::ADVERTISING);

  // 6. Un comando que llega antes de que poll() vea la conexión no se pierde.
  ble.connected = true;
  ble.write(kStart);
  CHECK(p.state() == DeviceState::STREAMING);
  p.poll(1100);
  CHECK(p.state() == DeviceState::STREAMING);
  CHECK(ble.notifiesInsideCallback == 0);
}

static void testTxQueueFull() {
  std::printf("D3 BLE_TX_QUEUE_FULL\n");
  TestPatternSource src(ECG_HW_SAMPLE_RATE_HZ, 75.0f, 9);
  MockTransport ble;
  StubInferenceEngine model(0);
  EcgPipeline p(src, ble, model);
  CHECK(p.begin("TEST"));
  ble.connected = true;
  p.poll(0);
  ble.write(kStart);
  CHECK(p.state() == DeviceState::STREAMING);
  for (int i = 0; i < 5; ++i) p.poll(10 + i);
  CHECK(p.error() == ErrorCode::MODEL_NOT_LOADED && !ble.has("\"error_code\":21"));

  ble.failNotify = true;  // la cola de la pila se llena
  for (int i = 0; i < 5; ++i) p.poll(20 + i);
  CHECK(p.error() == ErrorCode::BLE_TX_QUEUE_FULL);
  CHECK(p.stats().packetsDropped > 0);
  CHECK(ble.statusRead.find("\"error_code\":21") != std::string::npos);  // visible por lectura

  ble.failNotify = false;  // se libera: el STATUS pendiente se reintenta y luego se despeja
  for (int i = 0; i < 5; ++i) p.poll(30 + i);
  CHECK(p.error() == ErrorCode::MODEL_NOT_LOADED);
  size_t full = ble.json.size(), cleared = ble.json.size();
  for (size_t k = 0; k < ble.json.size(); ++k) {
    if (ble.json[k].find("\"error_code\":21,\"error\":\"BLE_TX_QUEUE_FULL\"") != std::string::npos && full == ble.json.size())
      full = k;
    if (full < k && ble.json[k].find("\"type\":\"status\"") != std::string::npos &&
        ble.json[k].find("\"error_code\":30") != std::string::npos)
      cleared = k;
  }
  CHECK(full < ble.json.size() && cleared < ble.json.size());  // se notificó y luego se despejó
  CHECK(ble.count("\"error_code\":21,\"error\"") == 1);          // una sola notificación entregada

  // Al desconectar, la congestión de la conexión anterior se olvida.
  ble.failNotify = true;
  for (int i = 0; i < 3; ++i) p.poll(40 + i);
  CHECK(p.error() == ErrorCode::BLE_TX_QUEUE_FULL);
  ble.failNotify = false;
  ble.connected = false;
  p.poll(50);
  CHECK(p.state() == DeviceState::ADVERTISING && p.error() == ErrorCode::MODEL_NOT_LOADED);
}

static void testStatusReadUptime() {
  std::printf("D6 uptime_ms en la lectura de STATUS\n");
  TestPatternSource src(ECG_HW_SAMPLE_RATE_HZ, 75.0f, 2);
  MockTransport ble;
  StubInferenceEngine model(0);
  EcgPipeline p(src, ble, model);
  CHECK(p.begin("TEST"));
  p.poll(0);
  CHECK(ble.statusRead.find("\"uptime_ms\":0") != std::string::npos);
  p.poll(ECG_STATUS_READ_REFRESH_MS - 1);  // todavía no toca refrescar
  CHECK(ble.statusRead.find("\"uptime_ms\":0") != std::string::npos);
  p.poll(250);
  CHECK(ble.statusRead.find("\"uptime_ms\":250") != std::string::npos);
  p.poll(60000);  // sin conexión ni cambios de estado, también se actualiza
  CHECK(ble.statusRead.find("\"uptime_ms\":60000") != std::string::npos);
  CHECK(ble.statusRead.find("\"state\":\"advertising\"") != std::string::npos);
  CHECK(ble.json.empty());  // refrescar la lectura no genera notificaciones
}

#include "generated/preprocessing_coeffs.h"

static void testFilterAndDetectorParity() {
  std::printf("paridad filtro/detector con Python (%u muestras)\n", TV_INPUT_LEN);
  BiquadCascade f;
  CHECK(f.init(ECG_FILTER_SOS, ECG_FILTER_NUM_SECTIONS));
  RPeakDetector det;
  CHECK(det.init(TV_FS));
  double maxErr = 0.0, maxRef = 0.0;
  std::vector<uint32_t> peaks;
  for (unsigned i = 0; i < TV_INPUT_LEN; ++i) {
    const float y = f.process(static_cast<float>(TV_INPUT[i]));
    maxErr = std::fmax(maxErr, std::fabs(y - TV_FILTERED[i]));
    maxRef = std::fmax(maxRef, std::fabs(TV_FILTERED[i]));
    uint32_t r;
    if (det.process(y, &r)) peaks.push_back(r);
  }
  std::printf("  error máx. filtro: %.3g (relativo %.2e)\n", maxErr, maxErr / maxRef);
  CHECK(maxErr / maxRef < 1e-3);
  CHECK(peaks.size() == TV_PEAKS_LEN);
  unsigned matched = 0;
  for (size_t k = 0; k < peaks.size() && k < TV_PEAKS_LEN; ++k)
    if (std::abs(static_cast<long>(peaks[k]) - static_cast<long>(TV_PEAKS[k])) <= 1) ++matched;
  std::printf("  picos R: C++=%zu Python=%u coincidentes(±1)=%u\n", peaks.size(), TV_PEAKS_LEN, matched);
  CHECK(matched == TV_PEAKS_LEN);
}

static void testPipeline() {
  std::printf("pipeline con transporte simulado\n");
  TestPatternSource src(ECG_HW_SAMPLE_RATE_HZ, 75.0f, 3);
  src.setLeadOffWindow(1800, 2160);
  MockTransport ble;
  StubInferenceEngine model(0);
  EcgPipeline p(src, ble, model);
  CHECK(p.begin("TEST"));
  CHECK(p.error() == ErrorCode::MODEL_NOT_LOADED);  // sin modelo: lo declara, no inventa
  CHECK(std::strstr(ble.deviceInfo.c_str(), "\"simulated\":true") != nullptr);

  ble.connected = true;
  ble.mtuValue = 23;
  p.poll(0);
  ble.write("{\"v\":1,\"type\":\"cmd\",\"cmd\":\"start_stream\",\"cmd_id\":1}");
  CHECK(p.state() == DeviceState::CONNECTED && ble.has("\"error_code\":42"));  // MTU_TOO_SMALL

  ble.mtuValue = 247;
  ble.connected = false;
  p.poll(1);
  ble.connected = true;
  p.poll(2);
  ble.write("{\"v\":1,\"type\":\"cmd\",\"cmd\":\"start_stream\",\"cmd_id\":2}");
  CHECK(p.state() == DeviceState::STREAMING);
  const uint32_t startIdx = p.stats().samples;
  for (int i = 0; i < 100; ++i) p.poll(10 + i);  // 6400 muestras
  ble.write("{\"v\":1,\"type\":\"cmd\",\"cmd\":\"stop_stream\",\"cmd_id\":3}");

  // Integridad: índices consecutivos y número total de muestras.
  uint32_t expected = startIdx, total = 0;
  bool contiguous = true, sawLeadOff = false;
  uint16_t seq = 0;
  for (size_t k = 0; k < ble.packets.size(); ++k) {
    const auto& pk = ble.packets[k];
    const uint32_t first = pk[4] | (pk[5] << 8) | (pk[6] << 16) | (static_cast<uint32_t>(pk[7]) << 24);
    const uint16_t s = static_cast<uint16_t>(pk[2] | (pk[3] << 8));
    if (first != expected || (k > 0 && s != static_cast<uint16_t>(seq + 1))) contiguous = false;
    seq = s;
    expected = first + pk[8];
    total += pk[8];
    CHECK(pk[9] & ECG_FLAG_SIMULATED);
    if (pk[9] & ECG_FLAG_LEAD_OFF) sawLeadOff = true;
    CHECK(pk.size() <= 247u - ECG_ATT_OVERHEAD);
  }
  CHECK(contiguous);
  CHECK(total == p.stats().samples - startIdx);
  CHECK(sawLeadOff && ble.has("\"event\":\"lead_off\"") && ble.has("\"error\":\"LEAD_OFF\""));
  std::printf("  paquetes=%zu muestras=%u latidos detectados=%u\n", ble.packets.size(), total,
              p.stats().beatsDetected);
  // 75 lpm durante ~18 s, menos 2 s de aprendizaje y 1 s sin electrodo: ~19 latidos.
  CHECK(p.stats().beatsDetected >= 14 && p.stats().beatsDetected <= 24);
  CHECK(p.stats().beatsClassified == 0);  // stub: ninguna clasificación inventada
}

// C1: RrFeatureTracker C++ frente a rr_features.RrFeatureTracker (Python).
static void testRrTrackerParity() {
  std::printf("C1 características RR C++ vs Python (%u eventos)\n", TV_C1_EVENTS_LEN);
  RrFeatureTracker t;
  t.init(TV_FS);
  std::vector<RrBeat> got;
  for (unsigned i = 0; i < TV_C1_EVENTS_LEN; ++i) {
    if (TV_C1_EVENTS[i] == TV_RESET) {
      t.reset();
      continue;
    }
    RrBeat b;
    if (t.onBeat(TV_C1_EVENTS[i], &b)) got.push_back(b);
  }
  CHECK(got.size() == TV_C1_R_LEN);
  double maxErr = 0.0;
  unsigned rOk = 0, warmOk = 0;
  for (size_t k = 0; k < got.size() && k < TV_C1_R_LEN; ++k) {
    rOk += got[k].rIndex == TV_C1_R[k];
    warmOk += got[k].warm == (TV_C1_WARM[k] != 0);
    for (int f = 0; f < ECG_RR_NUM_FEATURES; ++f)
      maxErr = std::fmax(maxErr, std::fabs(got[k].features[f] - TV_C1_FEAT[k * ECG_RR_NUM_FEATURES + f]));
  }
  std::printf("  latidos=%zu/%u  R iguales=%u  calentamiento igual=%u  error máx.=%.2e\n", got.size(), TV_C1_R_LEN,
              rOk, warmOk, maxErr);
  CHECK(rOk == TV_C1_R_LEN && warmOk == TV_C1_R_LEN);
  CHECK(maxErr < 1e-5);

  RrBeat b;  // timeout: sin latido en ECG_RR_MAX_S se emite el pendiente con post = máximo
  RrFeatureTracker t2;
  t2.init(360.0f);
  t2.onBeat(0, &b);
  t2.onBeat(360, &b);
  CHECK(!t2.onTimeout(360 + 1079, &b));
  CHECK(t2.onTimeout(360 + 1080, &b) && b.rIndex == 360 && std::fabs(b.postS - ECG_RR_MAX_S) < 1e-6f);
  CHECK(!t2.onTimeout(360 + 2000, &b));
}

// C2: cuantización int8 bit a bit igual que Python.
static void testQuantizationParity() {
  std::printf("C2 cuantización int8 C++ vs Python\n");
  unsigned same = 0, total = 0;
  for (unsigned s = 0; s < TV_C2_SCALE_LEN; ++s)
    for (unsigned i = 0; i < TV_C2_X_LEN; ++i, ++total)
      same += quantizeInt8(TV_C2_X[i], TV_C2_SCALE[s], TV_C2_ZP[s]) == TV_C2_Q[s * TV_C2_X_LEN + i];
  std::printf("  idénticos %u/%u\n", same, total);
  CHECK(same == total && total == TV_C2_Q_LEN);
}

// C3: el pipeline entrega al modelo exactamente lo mismo que simulate_device (Python).
class VectorSource : public IEcgSource {
 public:
  bool begin() override { return true; }
  bool read(EcgSample* s) override {
    if (pos >= TV_C3_INPUT_LEN) return false;
    s->value = TV_C3_INPUT[pos];
    s->leadOff = TV_C3_LEADOFF[pos] != 0;
    s->saturated = false;
    ++pos;
    return true;
  }
  float sampleRateHz() const override { return TV_FS; }
  bool isSimulated() const override { return true; }
  unsigned pos = 0;
};

class RecordingEngine : public IInferenceEngine {
 public:
  explicit RecordingEngine(const VectorSource& s) : src(s) {}
  bool begin() override { return true; }
  bool ready() const override { return true; }
  size_t inputLength() const override { return 252; }
  size_t rrFeatures() const override { return ECG_RR_NUM_FEATURES; }
  bool classify(const float* beat, size_t len, const float* rr, size_t nRr, InferenceResult* out) override {
    Call c;
    c.at = src.pos - 1;
    c.beat.assign(beat, beat + len);
    c.rr.assign(rr, rr + nRr);
    calls.push_back(c);
    out->classIndex = 2;  // siempre 'V' con confianza 1: para comprobar que el calentamiento bloquea alertas
    out->beatClass = 'V';
    out->confidence = 1.0f;
    return true;
  }
  struct Call {
    unsigned at = 0;
    std::vector<float> beat, rr;
  };
  std::vector<Call> calls;
  const VectorSource& src;
};

static void testPipelineRrParity() {
  std::printf("C3 pipeline C++ vs simulate_device Python (%u muestras)\n", TV_C3_INPUT_LEN);
  VectorSource src;
  MockTransport ble;
  RecordingEngine model(src);
  EcgPipeline p(src, ble, model);
  CHECK(p.begin("C3"));
  CHECK(p.error() == ErrorCode::OK);
  ble.connected = true;
  for (unsigned t = 0; src.pos < TV_C3_INPUT_LEN; ++t) p.poll(t);

  const auto& calls = model.calls;
  CHECK(calls.size() == TV_C3_R_LEN);
  unsigned atOk = 0;
  double maxBeat = 0.0, maxRr = 0.0;
  for (size_t k = 0; k < calls.size() && k < TV_C3_R_LEN; ++k) {
    atOk += calls[k].at == TV_C3_AT[k];
    for (size_t i = 0; i < calls[k].beat.size(); ++i)
      maxBeat = std::fmax(maxBeat, std::fabs(calls[k].beat[i] - TV_C3_BEAT[k * 252 + i]));
    for (size_t i = 0; i < calls[k].rr.size(); ++i)
      maxRr = std::fmax(maxRr, std::fabs(calls[k].rr[i] - TV_C3_RR[k * ECG_RR_NUM_FEATURES + i]));
  }
  unsigned warm = 0;
  for (unsigned k = 0; k < TV_C3_WARM_LEN; ++k) warm += TV_C3_WARM[k];
  // Alertas: solo latidos calientes, con 2 s mínimo entre alertas.
  unsigned arrhythmiaAlerts = 0;
  for (const auto& j : ble.json) arrhythmiaAlerts += j.find("arrhythmia_suspected") != std::string::npos;
  std::printf("  clasificados C++=%zu Python=%u  muestra de clasificación igual=%u  "
              "error máx. latido=%.2e  RR=%.2e  calientes=%u alertas=%u\n",
              calls.size(), TV_C3_R_LEN, atOk, maxBeat, maxRr, warm, arrhythmiaAlerts);
  CHECK(atOk == TV_C3_R_LEN);  // mismo instante: retraso de un latido idéntico
  CHECK(maxBeat < 1e-3);
  CHECK(maxRr < 1e-5);
  CHECK(arrhythmiaAlerts > 0 && arrhythmiaAlerts <= warm);
  CHECK(ble.has("\"event\":\"lead_off\""));

  // Obligatorio: al menos un latido clasificado por timeout (pausa de 8 s sin
  // latidos), los mismos que en Python, y con su ventana correcta aunque ya no
  // esté en el buffer circular.
  unsigned timeouts = 0, expectedTimeouts = 0, timeoutOutsideRing = 0;
  for (size_t k = 0; k < calls.size() && k < TV_C3_R_LEN; ++k) {
    const bool waited = calls[k].at - TV_C3_R[k] >= static_cast<unsigned>(ECG_RR_MAX_S * TV_FS);
    timeouts += waited;
    expectedTimeouts += TV_C3_TIMEOUT[k];
    CHECK(waited == (TV_C3_TIMEOUT[k] != 0));
    if (TV_C3_TIMEOUT[k] && calls[k].at + 1 - (TV_C3_R[k] - 90) > ECG_BEAT_RING_SIZE) ++timeoutOutsideRing;
  }
  CHECK(p.stats().beatsByTimeout == expectedTimeouts);
  CHECK(timeoutOutsideRing >= 1);  // la ventana salió de la copia, no del buffer
  // Ninguna alerta de arritmia durante el calentamiento: cada alerta debe
  // corresponder a un latido marcado como caliente.
  unsigned alertsChecked = 0, alertsInWarmup = 0;
  for (const auto& j : ble.json) {
    if (j.find("arrhythmia_suspected") == std::string::npos) continue;
    const size_t p0 = j.find("\"sample_index\":");
    CHECK(p0 != std::string::npos);
    if (p0 == std::string::npos) continue;
    const unsigned long si = std::strtoul(j.c_str() + p0 + 15, nullptr, 10);
    bool found = false;
    for (unsigned k = 0; k < TV_C3_R_LEN; ++k) {
      if (TV_C3_R[k] != si) continue;
      found = true;
      alertsInWarmup += TV_C3_WARM[k] ? 0 : 1;
    }
    CHECK(found);
    ++alertsChecked;
  }
  std::printf("  clasificados por timeout=%u (Python=%u, ventana fuera del buffer=%u)  alertas verificadas=%u  "
              "alertas en calentamiento=%u\n",
              timeouts, expectedTimeouts, timeoutOutsideRing, alertsChecked, alertsInWarmup);
  CHECK(timeouts >= 1);
  CHECK(alertsChecked == arrhythmiaAlerts && alertsInWarmup == 0);
}

// Motor que falla en las primeras inferencias y luego funciona.
class FlakyEngine : public IInferenceEngine {
 public:
  bool begin() override { return true; }
  bool ready() const override { return true; }
  size_t inputLength() const override { return 252; }
  size_t rrFeatures() const override { return ECG_RR_NUM_FEATURES; }
  bool classify(const float*, size_t, const float*, size_t, InferenceResult* out) override {
    if (++calls <= failFirst) return false;
    out->classIndex = 0;
    out->beatClass = 'N';
    out->confidence = 0.99f;
    return true;
  }
  unsigned calls = 0, failFirst = 3;
};

static void testInferenceFailedStatus() {
  std::printf("D5 INFERENCE_FAILED en STATUS\n");
  VectorSource src;
  MockTransport ble;
  FlakyEngine model;
  EcgPipeline p(src, ble, model);
  CHECK(p.begin("D5"));
  CHECK(p.error() == ErrorCode::OK);
  ble.connected = true;
  bool sawFailed = false;
  for (unsigned t = 0; src.pos < TV_C3_INPUT_LEN; ++t) {
    p.poll(t);
    sawFailed = sawFailed || p.error() == ErrorCode::INFERENCE_FAILED;
  }
  CHECK(model.calls > model.failFirst);
  CHECK(sawFailed);
  CHECK(ble.count("\"error_code\":31,\"error\":\"INFERENCE_FAILED\"") == 1);  // una notificación, no una por latido
  CHECK(p.error() == ErrorCode::OK);                                          // se despeja con la siguiente inferencia correcta
  CHECK(p.stats().beatsClassified == model.calls - model.failFirst);
  // Tras el fallo hay un STATUS con OK posterior.
  size_t failedAt = ble.json.size();
  bool okAfter = false;
  for (size_t k = 0; k < ble.json.size(); ++k) {
    if (ble.json[k].find("\"error_code\":31") != std::string::npos) failedAt = k;
    if (k > failedAt && ble.json[k].find("\"type\":\"status\"") != std::string::npos &&
        ble.json[k].find("\"error_code\":0") != std::string::npos)
      okAfter = true;
  }
  CHECK(okAfter);
}

int main() {
  testGoldenPacket();
  testMtu();
  testJson();
  testParseCommand();
  testDeviceName();
  testControlQueue();
  testFilterAndDetectorParity();
  testPipeline();
  testDeferredControl();
  testTxQueueFull();
  testStatusReadUptime();
  testRrTrackerParity();
  testQuantizationParity();
  testPipelineRrParity();
  testInferenceFailedStatus();
  std::printf(g_failures ? "\n%d FALLOS\n" : "\nTODAS LAS PRUEBAS HOST OK\n", g_failures);
  return g_failures ? 1 : 0;
}
