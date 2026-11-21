#include "ble_transport.h"

#include <cstring>

namespace ecg {

bool ControlQueue::push(const uint8_t* data, size_t len) {
  const uint32_t head = head_.load(std::memory_order_relaxed);  // propio: relaxed basta
  const uint32_t tail = tail_.load(std::memory_order_acquire);  // ¿liberó hueco el consumidor?
  if (head - tail >= kSlots) {
    droppedFull_.store(droppedFull_.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    return false;
  }
  Slot& s = slots_[head & (kSlots - 1)];
  if (data == nullptr) len = 0;
  if (len > kMaxLen) {
    s.len = 0;
    s.oversize = true;
    oversize_.store(oversize_.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
  } else {
    if (len) std::memcpy(s.data, data, len);
    s.len = static_cast<uint16_t>(len);
    s.oversize = false;
  }
  head_.store(head + 1, std::memory_order_release);  // publica el slot completo
  return true;
}

bool ControlQueue::pop(uint8_t* out, size_t* len, bool* oversize) {
  const uint32_t tail = tail_.load(std::memory_order_relaxed);
  const uint32_t head = head_.load(std::memory_order_acquire);  // ve el slot ya escrito
  if (head == tail) return false;
  const Slot& s = slots_[tail & (kSlots - 1)];
  if (s.len) std::memcpy(out, s.data, s.len);
  *len = s.len;
  *oversize = s.oversize;
  tail_.store(tail + 1, std::memory_order_release);  // devuelve el slot al productor
  return true;
}

void ControlQueue::clear() { tail_.store(head_.load(std::memory_order_acquire), std::memory_order_release); }

size_t ControlQueue::size() const {
  return static_cast<size_t>(head_.load(std::memory_order_acquire) - tail_.load(std::memory_order_acquire));
}

void QueuedBleTransport::setControlHandler(ControlHandler handler, void* ctx) {
  handler_ = handler;
  handlerCtx_ = ctx;
}

size_t QueuedBleTransport::service(size_t maxCommands) {
  uint8_t buf[ControlQueue::kMaxLen];
  size_t delivered = 0;
  while (delivered < maxCommands) {
    size_t len = 0;
    bool oversize = false;
    if (!queue_.pop(buf, &len, &oversize)) break;
    ++delivered;
    if (handler_ == nullptr) continue;  // sin consumidor registrado: se descarta
    dispatching_ = true;
    // Una escritura demasiado larga se entrega vacía: el parser la rechaza y el
    // pipeline responde cmd_ack(-1, INVALID_COMMAND).
    handler_(buf, oversize ? 0 : len, handlerCtx_);
    dispatching_ = false;
  }
  return delivered;
}

}  // namespace ecg
