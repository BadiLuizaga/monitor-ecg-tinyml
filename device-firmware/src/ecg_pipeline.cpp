#include "ecg_pipeline.h"

#include <cmath>
#include <cstdio>
#include <cstring>

#include "generated/preprocessing_coeffs.h"
#include "hardware_config.h"

#if defined(__has_include)
#if __has_include("generated/model_config.h")
#include "generated/model_config.h"
#endif
#endif

namespace ecg {

namespace {
// Ventana del latido: la del modelo exportado o, si no existe, la de
// ai-model/src/preprocessing/segmentation.py (0.25 s antes, 0.45 s después).
size_t windowBefore(float fs) {
#ifdef ECG_MODEL_WINDOW_BEFORE
  (void)fs;
  return ECG_MODEL_WINDOW_BEFORE;
#else
  return static_cast<size_t>(std::lround(0.25f * fs));
#endif
}
size_t windowAfter(float fs) {
#ifdef ECG_MODEL_WINDOW_AFTER
  (void)fs;
  return ECG_MODEL_WINDOW_AFTER;
#else
  return static_cast<size_t>(std::lround(0.45f * fs));
#endif
}
}  // namespace

static_assert((ECG_BEAT_RING_SIZE & (ECG_BEAT_RING_SIZE - 1)) == 0, "ECG_BEAT_RING_SIZE debe ser potencia de 2");

#ifdef ECG_MODEL_HAS_RR_INPUT
// El modelo se entrenó con unos parámetros RR concretos: deben coincidir.
static_assert(ECG_MODEL_RR_FEATURES == ECG_RR_NUM_FEATURES, "RR: número de características");
static_assert(ECG_MODEL_RR_LOCAL_BEATS == ECG_RR_LOCAL_BEATS, "RR: local_beats");
static_assert(ECG_MODEL_RR_LONG_TAU_S == ECG_RR_LONG_TAU_S, "RR: long_tau_s");
static_assert(ECG_MODEL_RR_MAX_S == ECG_RR_MAX_S, "RR: max_s");
static_assert(ECG_MODEL_RR_REF_S == ECG_RR_REF_S, "RR: ref_s");
static_assert(ECG_MODEL_RR_CLIP == ECG_RR_CLIP, "RR: clip");
static_assert(ECG_MODEL_RR_WARMUP_MIN_RR == ECG_RR_WARMUP_MIN_RR, "RR: warmup_min_rr");
static_assert(ECG_MODEL_RR_WARMUP_S == ECG_RR_WARMUP_S, "RR: warmup_s");
#endif

bool EcgPipeline::begin(const char* deviceId) {
  deviceId_ = deviceId;
  const float fs = src_.sampleRateHz();
  before_ = windowBefore(fs);
  after_ = windowAfter(fs);
  bool ok = src_.begin() && filter_.init(ECG_FILTER_SOS, ECG_FILTER_NUM_SECTIONS) && detector_.init(fs);
  rr_.init(fs);
  alerts_.init(fs);
  packetizer_.reset();
  packetizer_.setMtu(ECG_DEFAULT_MTU);

  // Los filtros y el modelo se diseñaron para una fs concreta: si el sensor
  // usa otra, hay que regenerarlos (y reentrenar). No se corrige en silencio.
  if (std::fabs(fs - ECG_FILTER_FS_HZ) > 0.01f || before_ + after_ > ECG_BEAT_RING_SIZE / 2 ||
      before_ + after_ > ECG_MAX_BEAT_WINDOW) {
    error_ = ErrorCode::CONFIG_MISMATCH;
    ok = false;
  }
#ifdef ECG_MODEL_FS_HZ
  if (std::fabs(fs - ECG_MODEL_FS_HZ) > 0.01f) {
    error_ = ErrorCode::CONFIG_MISMATCH;
    ok = false;
  }
#endif
  if (ok && (!model_.begin() || model_.inputLength() != before_ + after_ ||
             (model_.rrFeatures() != 0 && model_.rrFeatures() != ECG_RR_NUM_FEATURES))) {
    // El dispositivo sigue transmitiendo ECG, pero sin clasificación.
    error_ = ErrorCode::MODEL_NOT_LOADED;
    baseError_ = error_;
  }

  char name[32];
  std::snprintf(name, sizeof name, "%s-%s", ECG_DEVICE_NAME_PREFIX, deviceId_);
  ok = ble_.begin(name) && ok;
  ble_.setControlHandler(
      [](const uint8_t* d, size_t n, void* ctx) { static_cast<EcgPipeline*>(ctx)->onControlWrite(d, n); }, this);

  char info[ECG_JSON_MAX_READ_BYTES + 1];
  DeviceInfo di{deviceId_, ECG_FW_VERSION, src_.isSimulated() ? "SIMULATOR" : ECG_HW_MODEL, fs,
                ECG_HW_ADC_BITS, ECG_HW_MV_PER_COUNT, ECG_HW_LEADS, src_.isSimulated()};
  const size_t n = buildDeviceInfoJson(info, sizeof info, di);
  ble_.setReadValue(Characteristic::DEVICE_INFO, reinterpret_cast<const uint8_t*>(info), n);
  setState(ok ? DeviceState::ADVERTISING : DeviceState::ERROR);
  return ok;
}

void EcgPipeline::poll(uint32_t nowMs) {
  nowMs_ = nowMs;
  const bool connected = ble_.isConnected();
  if (connected && !wasConnected_) {
    packetizer_.setMtu(ble_.mtu());
    if (state_ != DeviceState::ERROR) setState(DeviceState::CONNECTED);
  } else if (!connected && wasConnected_) {
    // Pérdida de conexión: se deja de transmitir y se vuelve a anunciar. La
    // app reconecta con backoff y reinicia su receptor.
    packetizer_.reset();
    if (state_ != DeviceState::ERROR) setState(DeviceState::ADVERTISING);
  }
  wasConnected_ = connected;
  if (state_ == DeviceState::ERROR) return;

  // Acota el trabajo por llamada para no bloquear la pila BLE.
  EcgSample s;
  for (int i = 0; i < 64 && src_.read(&s); ++i) processSample(s);
}

void EcgPipeline::processSample(const EcgSample& s) {
  const uint32_t idx = sampleIndex_++;
  ++stats_.samples;

  if (s.leadOff != leadOff_) {
    leadOff_ = s.leadOff;
    error_ = leadOff_ ? ErrorCode::LEAD_OFF : baseError_;
    sendStatus();
    if (leadOff_) sendAlert("lead_off", '-', 0.0f, idx);
    // Igual que simulate_device (Python): se descarta lo pendiente y se
    // reinician los RR en ambos flancos.
    pendingCount_ = 0;
    rr_.reset();
  }

  if (state_ == DeviceState::STREAMING) {
    uint8_t flags = 0;
    if (s.leadOff) flags |= ECG_FLAG_LEAD_OFF;
    if (s.saturated) flags |= ECG_FLAG_SATURATED;
    if (src_.isSimulated()) flags |= ECG_FLAG_SIMULATED;
    packetizer_.push(ble_, idx, s.value, flags);
    stats_.packetsDropped = packetizer_.droppedPackets();
  }

  // Cadena de clasificación (en cuentas de ADC: el z-score elimina la ganancia).
  const float y = filter_.process(static_cast<float>(s.value));
  ring_[idx & (ECG_BEAT_RING_SIZE - 1)] = y;
  // Mismo orden que simulate_device (Python). El latido k recibe su RR del
  // RrFeatureTracker al llegar el k+1 o tras ECG_RR_MAX_S (retraso de un
  // latido); su ventana se copia en cuanto está completa, sin depender del
  // buffer circular.
  uint32_t r = 0;
  RrBeat b;
  if (detector_.process(y, &r) && !leadOff_) {  // 2. detección
    ++stats_.beatsDetected;
    if (rr_.onBeat(r, &b)) attachRr(b, false);
    uint32_t pr = 0;
    if (rr_.pendingIndex(&pr) && pr == r) enqueue(r);
  }
  if (!leadOff_ && rr_.onTimeout(idx, &b)) attachRr(b, true);  // 3. timeout
  const size_t len = before_ + after_;
  for (size_t k = 0; k < pendingCount_;) {  // 4. copiar ventanas completas
    PendingBeat& e = pending_[k];
    if (!e.hasWindow && idx + 1 >= e.r + after_) {
      if (e.r < before_ || idx + 1 - (e.r - before_) > ECG_BEAT_RING_SIZE) {
        removeAt(k);  // ventana no disponible
        continue;
      }
      for (size_t i = 0; i < len; ++i) e.window[i] = ring_[(e.r - before_ + i) & (ECG_BEAT_RING_SIZE - 1)];
      e.hasWindow = true;
    }
    ++k;
  }
  while (pendingCount_ > 0 && pending_[0].hasWindow && pending_[0].hasRr) {  // 5. clasificar
    handleBeat(pending_[0]);
    removeAt(0);
  }
}

void EcgPipeline::enqueue(uint32_t r) {
  if (pendingCount_ == ECG_MAX_PENDING_BEATS) removeAt(0);  // no debería ocurrir: descarta el más antiguo
  PendingBeat& e = pending_[pendingCount_++];
  e.r = r;
  e.hasWindow = false;
  e.hasRr = false;
  e.timeout = false;
}

void EcgPipeline::attachRr(const RrBeat& b, bool timeout) {
  for (size_t k = 0; k < pendingCount_; ++k) {
    if (pending_[k].r != b.rIndex) continue;
    pending_[k].rr = b;
    pending_[k].hasRr = true;
    pending_[k].timeout = timeout;
    return;
  }
}

void EcgPipeline::removeAt(size_t k) {
  if (k >= pendingCount_) return;
  std::memmove(pending_ + k, pending_ + k + 1, (pendingCount_ - k - 1) * sizeof pending_[0]);
  --pendingCount_;
}

void EcgPipeline::handleBeat(PendingBeat& e) {
  const RrBeat& b = e.rr;
  const uint32_t rIndex = e.r;
  const size_t len = before_ + after_;
  zscoreInPlace(e.window, len);
  if (!model_.ready()) return;
  InferenceResult res;
  if (!model_.classify(e.window, len, b.features, model_.rrFeatures(), &res)) {
    error_ = ErrorCode::INFERENCE_FAILED;
    return;
  }
  ++stats_.beatsClassified;
  if (e.timeout) ++stats_.beatsByTimeout;
  if (!b.warm) return;  // calentamiento RR: se clasifica, pero no se alerta
  const AlertDecision d = alerts_.onBeat(res.beatClass, res.confidence, rIndex);
  if (d.raise) sendAlert("arrhythmia_suspected", d.beatClass, d.confidence, d.sampleIndex);
}

void EcgPipeline::onControlWrite(const uint8_t* data, size_t len) {
  ParsedCommand c;
  const bool valid = parseCommand(reinterpret_cast<const char*>(data), len, &c);
  if (!valid) {
    sendAck(c.cmdId, false, c.versionOk ? ErrorCode::INVALID_COMMAND : ErrorCode::UNSUPPORTED_PROTOCOL_VERSION);
    return;
  }
  switch (c.cmd) {
    case Command::START_STREAM:
      if (ble_.mtu() < ECG_MIN_MTU_FOR_JSON) {
        sendAck(c.cmdId, false, ErrorCode::MTU_TOO_SMALL);
        return;
      }
      packetizer_.setMtu(ble_.mtu());
      setState(DeviceState::STREAMING);
      break;
    case Command::STOP_STREAM:
      packetizer_.flush(ble_);
      setState(DeviceState::CONNECTED);
      break;
    case Command::GET_STATUS:
      sendStatus();
      break;
    case Command::PING:
    case Command::ACK_ALERT:  // la app confirma que mostró la alerta c.eventId
    case Command::UNKNOWN:
      break;
  }
  sendAck(c.cmdId, true, ErrorCode::OK);
}

void EcgPipeline::setState(DeviceState s) {
  state_ = s;
  sendStatus();
}

void EcgPipeline::sendStatus() {
  char buf[ECG_JSON_MAX_BYTES + 1];
  const size_t n = buildStatusJson(buf, sizeof buf, state_, error_, nowMs_);
  ble_.setReadValue(Characteristic::STATUS, reinterpret_cast<const uint8_t*>(buf), n);
  if (ble_.isConnected()) ble_.notify(Characteristic::STATUS, reinterpret_cast<const uint8_t*>(buf), n);
}

void EcgPipeline::sendAlert(const char* event, char beatClass, float conf, uint32_t index) {
  if (!ble_.isConnected()) return;
  char buf[ECG_JSON_MAX_BYTES + 1];
  const size_t n = buildAlertJson(buf, sizeof buf, ++eventId_, event, beatClass, conf, index, src_.isSimulated());
  if (n && ble_.notify(Characteristic::ALERT, reinterpret_cast<const uint8_t*>(buf), n)) ++stats_.alertsSent;
}

void EcgPipeline::sendAck(int32_t cmdId, bool ok, ErrorCode err) {
  char buf[ECG_JSON_MAX_BYTES + 1];
  const size_t n = buildCmdAckJson(buf, sizeof buf, cmdId, ok, err);
  if (n) ble_.notify(Characteristic::STATUS, reinterpret_cast<const uint8_t*>(buf), n);
}

}  // namespace ecg
