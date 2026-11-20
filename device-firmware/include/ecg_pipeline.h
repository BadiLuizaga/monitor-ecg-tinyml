// ecg_pipeline.h - Orquestación de la lógica del dispositivo.
//
//   IEcgSource -> [stream crudo] -> EcgPacketizer -> BLE ECG_DATA
//              -> BiquadCascade -> RPeakDetector -> RrFeatureTracker (+1 latido)
//              -> ventana -> z-score -> IInferenceEngine (latido + RR)
//              -> AlertManager (solo latidos "calientes") -> BLE ALERT
//
// No contiene nada específico del hardware: recibe la fuente, el transporte
// BLE y el motor de inferencia por inyección de dependencias.
#pragma once

#include <cstddef>
#include <cstdint>

#include "alert_manager.h"
#include "ble_protocol.h"
#include "ecg_source.h"
#include "inference.h"
#include "integration_config.h"
#include "signal_processing.h"

namespace ecg {

struct PipelineStats {
  uint32_t samples = 0;
  uint32_t beatsDetected = 0;
  uint32_t beatsClassified = 0;
  uint32_t beatsByTimeout = 0;  // clasificados con RR siguiente = ECG_RR_MAX_S (pausa)
  uint32_t alertsSent = 0;
  uint32_t packetsDropped = 0;
};

class EcgPipeline {
 public:
  EcgPipeline(IEcgSource& source, IBleTransport& ble, IInferenceEngine& model)
      : src_(source), ble_(ble), model_(model) {}

  // `deviceId` debe vivir mientras viva el pipeline.
  bool begin(const char* deviceId);
  // Procesa todas las muestras disponibles. Llamar desde loop()/tarea.
  void poll(uint32_t nowMs);
  void onControlWrite(const uint8_t* data, size_t len);

  DeviceState state() const { return state_; }
  ErrorCode error() const { return error_; }
  const PipelineStats& stats() const { return stats_; }

 private:
  void processSample(const EcgSample& s);
  // Latido a la espera de su ventana y/o de su RR siguiente.
  struct PendingBeat {
    uint32_t r = 0;
    bool hasWindow = false;
    bool hasRr = false;
    bool timeout = false;
    RrBeat rr;
    float window[ECG_MAX_BEAT_WINDOW] = {};
  };
  void handleBeat(PendingBeat& e);
  void enqueue(uint32_t r);
  void attachRr(const RrBeat& b, bool timeout);
  void removeAt(size_t k);
  void setState(DeviceState s);
  void sendStatus();
  void sendAlert(const char* event, char beatClass, float conf, uint32_t index);
  void sendAck(int32_t cmdId, bool ok, ErrorCode err);
  void updateReadValues();

  IEcgSource& src_;
  IBleTransport& ble_;
  IInferenceEngine& model_;
  BiquadCascade filter_;
  RPeakDetector detector_;
  RrFeatureTracker rr_;
  AlertManager alerts_;
  EcgPacketizer packetizer_;
  float ring_[ECG_BEAT_RING_SIZE] = {};
  PendingBeat pending_[ECG_MAX_PENDING_BEATS] = {};
  size_t pendingCount_ = 0;
  size_t before_ = 0, after_ = 0;
  uint32_t sampleIndex_ = 0;
  uint32_t eventId_ = 0;
  uint32_t nowMs_ = 0;
  bool wasConnected_ = false;
  bool leadOff_ = false;
  const char* deviceId_ = "";
  DeviceState state_ = DeviceState::BOOTING;
  ErrorCode error_ = ErrorCode::OK;
  ErrorCode baseError_ = ErrorCode::OK;  // error persistente (p. ej. MODEL_NOT_LOADED)
  PipelineStats stats_;
};

}  // namespace ecg
