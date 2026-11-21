// ble_transport.h - Recepción diferida de comandos CONTROL (portable).
//
// Problema: la pila BLE (NimBLE en el ESP32) entrega las escrituras de
// CONTROL en SU tarea, mientras EcgPipeline se ejecuta en loop(). EcgPipeline
// no es thread-safe, así que el ControlHandler NUNCA debe ejecutarse dentro
// del callback de la pila.
//
// Solución:
//   callback de la pila ──► onControlWritten() ──► ControlQueue (copia, SPSC)
//   loop() ──► service() ──► ControlHandler ──► EcgPipeline::onControlWrite()
//   loop() ──► EcgPipeline::poll()
//
// QueuedBleTransport implementa la parte de IBleTransport que no depende de la
// pila. El futuro Esp32BleTransport (E2) hereda de ella e implementa begin(),
// isConnected(), mtu(), notify() y setReadValue() con NimBLE, y llama a
// onControlWritten() desde su callback onWrite de CONTROL.
//
// Sin memoria dinámica. No usa mutex: es una cola de un solo productor (la
// tarea de la pila BLE) y un solo consumidor (loop()), sincronizada con
// std::atomic (solo load/store acquire/release, sin operaciones RMW, que en el
// ESP32-C3 —RISC-V sin extensión A— requerirían libatomic).
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "ble_protocol.h"
#include "integration_config.h"

namespace ecg {

class ControlQueue {
 public:
  static constexpr size_t kSlots = ECG_CONTROL_QUEUE_SLOTS;
  static constexpr size_t kMaxLen = ECG_JSON_MAX_BYTES;
  static_assert(kSlots >= 1 && (kSlots & (kSlots - 1)) == 0, "ECG_CONTROL_QUEUE_SLOTS debe ser potencia de 2");

  // --- Productor (callback de la pila BLE). No bloquea. ---------------------
  // Copia la escritura. Si `len` > kMaxLen (180 B, ble_spec.md §5) no se copia
  // el contenido: se encola una escritura marcada como "oversize" para que el
  // consumidor la rechace con cmd_ack INVALID_COMMAND. Si la cola está llena se
  // descarta la escritura NUEVA (las anteriores se procesan en orden) y se
  // devuelve false; la app no recibe acuse y debe reintentar por timeout.
  bool push(const uint8_t* data, size_t len);

  // --- Consumidor (loop()). ------------------------------------------------
  // Saca la escritura más antigua. `out` debe tener al menos kMaxLen bytes.
  // `*len` = 0 y `*oversize` = true si la escritura superaba kMaxLen.
  bool pop(uint8_t* out, size_t* len, bool* oversize);
  // Descarta todo lo pendiente (p. ej. tras una desconexión).
  void clear();

  // --- Observación (cualquier contexto; valores aproximados) ---------------
  size_t size() const;
  bool empty() const { return size() == 0; }
  uint32_t droppedFull() const { return droppedFull_.load(std::memory_order_relaxed); }
  uint32_t oversize() const { return oversize_.load(std::memory_order_relaxed); }

 private:
  struct Slot {
    uint16_t len = 0;
    bool oversize = false;
    uint8_t data[kMaxLen] = {};
  };
  Slot slots_[kSlots];
  // head_: solo lo escribe el productor; tail_: solo el consumidor. Índices
  // libres que dan la vuelta en 2^32 (kSlots es potencia de 2).
  std::atomic<uint32_t> head_{0};
  std::atomic<uint32_t> tail_{0};
  // Contadores escritos solo por el productor (load + store, sin RMW).
  std::atomic<uint32_t> droppedFull_{0};
  std::atomic<uint32_t> oversize_{0};
};

class QueuedBleTransport : public IBleTransport {
 public:
  void setControlHandler(ControlHandler handler, void* ctx) override;

  // Llamar SOLO desde el callback de escritura de CONTROL de la pila BLE.
  // Únicamente copia en la cola; nunca invoca el ControlHandler.
  bool onControlWritten(const uint8_t* data, size_t len) { return queue_.push(data, len); }

  // Llamar desde loop(), en el mismo contexto que EcgPipeline::poll() y antes
  // que él. Entrega al ControlHandler hasta `maxCommands` escrituras en orden
  // de llegada. Devuelve cuántas entregó.
  size_t service(size_t maxCommands = ControlQueue::kSlots);

  // true mientras service() está entregando una escritura (para pruebas y
  // comprobaciones de contexto).
  bool dispatching() const { return dispatching_; }

  const ControlQueue& controlQueue() const { return queue_; }

 protected:
  // Para la implementación concreta al desconectarse (llamar desde loop()).
  void discardPendingControl() { queue_.clear(); }

 private:
  ControlQueue queue_;
  ControlHandler handler_ = nullptr;
  void* handlerCtx_ = nullptr;
  bool dispatching_ = false;
};

}  // namespace ecg
