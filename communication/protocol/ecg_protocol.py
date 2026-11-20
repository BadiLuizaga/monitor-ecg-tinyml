"""Códec de referencia del protocolo BLE del Monitor ECG TinyML.

Implementa, en Python puro (sin dependencias externas):

* Codificación/decodificación del paquete binario ``ECG_DATA``.
* Construcción y validación de mensajes JSON (``DEVICE_INFO``, ``STATUS``,
  ``ALERT``, ``CONTROL`` y su respuesta ``cmd_ack``).
* Un receptor de referencia (lado app / central) que reconstruye el flujo de
  muestras y detecta paquetes perdidos, duplicados o desordenados.
* La política de reconexión con *backoff* exponencial y el monitor de enlace.

Todos los valores (UUID, cabecera, códigos de error...) se leen de
``protocol_config.json``, que es la fuente única de verdad. Los valores
marcados como PROVISIONAL deben confirmarse con el hardware real.

Este módulo es la especificación ejecutable que deben respetar el firmware
(``device-firmware/src/ble_protocol.cpp``) y la app móvil.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

CONFIG_PATH = Path(__file__).with_name("protocol_config.json")


def load_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Carga la configuración del protocolo."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


CONFIG = load_config()

PROTOCOL_VERSION: int = CONFIG["protocol_version"]
SERVICE_UUID: str = CONFIG["gatt"]["service_uuid"]
CHARACTERISTICS: dict[str, dict[str, Any]] = CONFIG["gatt"]["characteristics"]
CHAR_ID_TO_NAME: dict[int, str] = {v["char_id"]: k for k, v in CHARACTERISTICS.items()}

_PKT = CONFIG["ecg_packet"]
HEADER_STRUCT = struct.Struct("<BBHIBB")  # version, type, seq, first_idx, count, flags
HEADER_SIZE: int = _PKT["header_size"]
assert HEADER_STRUCT.size == HEADER_SIZE, "protocol_config.json y HEADER_STRUCT no coinciden"
PACKET_TYPE_ECG_SAMPLES: int = _PKT["packet_types"]["ECG_SAMPLES"]
BYTES_PER_SAMPLE: int = _PKT["sample_formats"][_PKT["default_sample_format"]]
ATT_OVERHEAD: int = _PKT["att_overhead"]
DEFAULT_MTU: int = _PKT["default_mtu"]
MAX_SAMPLES_PER_PACKET: int = _PKT["max_samples_per_packet"]
FLAGS: dict[str, int] = _PKT["flags"]

JSON_MAX_BYTES: int = CONFIG["json_messages"]["max_bytes"]  # notify / write
JSON_MAX_READ_BYTES: int = CONFIG["json_messages"]["max_bytes_read"]  # DEVICE_INFO (long read)
READ_ONLY_MESSAGES = {"device_info"}
MIN_MTU_FOR_JSON: int = CONFIG["json_messages"]["min_mtu_for_json_notify"]
COMMANDS: tuple[str, ...] = tuple(CONFIG["commands"])
DEVICE_STATES: tuple[str, ...] = tuple(CONFIG["device_states"])
ERROR_CODES: dict[str, int] = CONFIG["error_codes"]
ERROR_NAMES: dict[int, str] = {v: k for k, v in ERROR_CODES.items()}
BEAT_CLASSES: tuple[str, ...] = tuple(k for k in CONFIG["beat_classes"] if not k.startswith("_"))

INT16_MIN, INT16_MAX = -32768, 32767
_SEQ_MOD = 1 << 16
_IDX_MOD = 1 << 32


class ProtocolError(ValueError):
    """Paquete o mensaje que no cumple el protocolo."""


# ---------------------------------------------------------------------------
# Paquete binario ECG_DATA
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EcgPacket:
    seq: int
    first_sample_index: int
    samples: tuple[int, ...]
    flags: int = 0
    version: int = PROTOCOL_VERSION

    def has_flag(self, name: str) -> bool:
        return bool(self.flags & FLAGS[name])


def max_samples_per_packet(mtu: int = DEFAULT_MTU) -> int:
    """Número máximo de muestras int16 que caben en una notificación."""
    payload = mtu - ATT_OVERHEAD - HEADER_SIZE
    if payload < BYTES_PER_SAMPLE:
        raise ProtocolError(f"MTU {mtu} demasiado pequeño para transmitir muestras")
    return min(payload // BYTES_PER_SAMPLE, MAX_SAMPLES_PER_PACKET)


def encode_ecg_packet(seq: int, first_sample_index: int, samples: Sequence[int], flags: int = 0) -> bytes:
    """Codifica un paquete ECG_DATA (cabecera de 10 bytes + muestras int16 LE)."""
    n = len(samples)
    if not 1 <= n <= MAX_SAMPLES_PER_PACKET:
        raise ProtocolError(f"sample_count fuera de rango: {n}")
    if not 0 <= flags <= 0xFF:
        raise ProtocolError(f"flags fuera de rango: {flags}")
    header = HEADER_STRUCT.pack(
        PROTOCOL_VERSION, PACKET_TYPE_ECG_SAMPLES, seq % _SEQ_MOD, first_sample_index % _IDX_MOD, n, flags
    )
    try:
        body = struct.pack(f"<{n}h", *samples)
    except struct.error as exc:
        raise ProtocolError(f"muestra fuera de rango int16: {exc}") from exc
    return header + body


def decode_ecg_packet(data: bytes) -> EcgPacket:
    """Decodifica y valida un paquete ECG_DATA."""
    if len(data) < HEADER_SIZE:
        raise ProtocolError(f"paquete demasiado corto ({len(data)} bytes)")
    version, ptype, seq, first_idx, count, flags = HEADER_STRUCT.unpack_from(data)
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"versión de protocolo no soportada: {version}")
    if ptype != PACKET_TYPE_ECG_SAMPLES:
        raise ProtocolError(f"tipo de paquete desconocido: {ptype}")
    expected = HEADER_SIZE + count * BYTES_PER_SAMPLE
    if count == 0 or len(data) != expected:
        raise ProtocolError(f"longitud inválida: {len(data)} bytes, se esperaban {expected}")
    samples = struct.unpack_from(f"<{count}h", data, HEADER_SIZE)
    return EcgPacket(seq=seq, first_sample_index=first_idx, samples=samples, flags=flags, version=version)


def packetize(
    samples: Sequence[int],
    first_sample_index: int = 0,
    first_seq: int = 0,
    mtu: int = DEFAULT_MTU,
    flags: int = 0,
) -> Iterator[bytes]:
    """Divide un bloque de muestras en paquetes ECG_DATA que caben en el MTU."""
    step = max_samples_per_packet(mtu)
    for k, start in enumerate(range(0, len(samples), step)):
        yield encode_ecg_packet(first_seq + k, first_sample_index + start, samples[start:start + step], flags)


def clamp_int16(value: float) -> int:
    return max(INT16_MIN, min(INT16_MAX, int(round(value))))


# ---------------------------------------------------------------------------
# Mensajes JSON
# ---------------------------------------------------------------------------

# Campos obligatorios y tipos por tipo de mensaje. message_schema.json es la
# versión JSON Schema equivalente (las pruebas comprueban que coinciden).
REQUIRED_FIELDS: dict[str, dict[str, tuple[type, ...]]] = {
    "device_info": {
        "v": (int,), "type": (str,), "device_id": (str,), "fw_version": (str,),
        "hw_model": (str,), "sample_rate_hz": (int, float), "sample_format": (str,),
        "adc_bits": (int, type(None)), "mv_per_count": (int, float, type(None)),
        "leads": (int,), "simulated": (bool,),
    },
    "status": {
        "v": (int,), "type": (str,), "state": (str,), "error_code": (int,),
        "error": (str,), "uptime_ms": (int,),
    },
    "alert": {
        "v": (int,), "type": (str,), "event_id": (int,), "event": (str,),
        "beat_class": (str,), "confidence": (int, float), "sample_index": (int,),
        "simulated": (bool,),
    },
    "cmd": {"v": (int,), "type": (str,), "cmd": (str,), "cmd_id": (int,)},
    "cmd_ack": {"v": (int,), "type": (str,), "cmd_id": (int,), "ok": (bool,), "error_code": (int,)},
}

ALERT_EVENTS = ("arrhythmia_suspected", "lead_off", "signal_quality", "device_error")


def validate_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Valida campos obligatorios, tipos y valores enumerados. Devuelve ``msg``."""
    if not isinstance(msg, dict):
        raise ProtocolError("el mensaje debe ser un objeto JSON")
    mtype = msg.get("type")
    spec = REQUIRED_FIELDS.get(mtype)  # type: ignore[arg-type]
    if spec is None:
        raise ProtocolError(f"tipo de mensaje desconocido: {mtype!r}")
    for key, types in spec.items():
        if key not in msg:
            raise ProtocolError(f"{mtype}: falta el campo '{key}'")
        value = msg[key]
        # bool es subclase de int: no aceptarlo donde se espera un número.
        if isinstance(value, bool) and bool not in types:
            raise ProtocolError(f"{mtype}: '{key}' tiene tipo inválido")
        if not isinstance(value, types):
            raise ProtocolError(f"{mtype}: '{key}' tiene tipo inválido")
    if msg["v"] != PROTOCOL_VERSION:
        raise ProtocolError(f"versión de protocolo no soportada: {msg['v']}")
    if mtype == "status" and msg["state"] not in DEVICE_STATES:
        raise ProtocolError(f"estado desconocido: {msg['state']}")
    if mtype in ("status", "cmd_ack") and msg["error_code"] not in ERROR_NAMES:
        raise ProtocolError(f"código de error desconocido: {msg['error_code']}")
    if mtype == "cmd" and msg["cmd"] not in COMMANDS:
        raise ProtocolError(f"comando desconocido: {msg['cmd']}")
    if mtype == "alert":
        if msg["event"] not in ALERT_EVENTS:
            raise ProtocolError(f"evento desconocido: {msg['event']}")
        if msg["beat_class"] not in BEAT_CLASSES + ("-",):
            raise ProtocolError(f"clase de latido desconocida: {msg['beat_class']}")
        if not 0.0 <= float(msg["confidence"]) <= 1.0:
            raise ProtocolError("confidence debe estar en [0, 1]")
    return msg


def encode_json(msg: dict[str, Any]) -> bytes:
    """Valida y serializa un mensaje en JSON compacto UTF-8.

    Límite: ``JSON_MAX_BYTES`` para mensajes que viajan en una notificación o
    escritura (un solo paquete); ``JSON_MAX_READ_BYTES`` para DEVICE_INFO.
    """
    validate_message(msg)
    data = json.dumps(msg, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    limit = JSON_MAX_READ_BYTES if msg["type"] in READ_ONLY_MESSAGES else JSON_MAX_BYTES
    if len(data) > limit:
        raise ProtocolError(f"mensaje {msg['type']} de {len(data)} bytes supera el máximo de {limit}")
    return data


def decode_json(data: bytes) -> dict[str, Any]:
    try:
        msg = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"JSON inválido: {exc}") from exc
    return validate_message(msg)


def make_device_info(
    device_id: str,
    fw_version: str,
    sample_rate_hz: float,
    *,
    hw_model: str = "TBD",
    adc_bits: int | None = None,
    mv_per_count: float | None = None,
    leads: int = 1,
    simulated: bool = False,
) -> dict[str, Any]:
    return {
        "v": PROTOCOL_VERSION, "type": "device_info", "device_id": device_id,
        "fw_version": fw_version, "hw_model": hw_model, "sample_rate_hz": sample_rate_hz,
        "sample_format": _PKT["default_sample_format"], "adc_bits": adc_bits,
        "mv_per_count": mv_per_count, "leads": leads, "simulated": simulated,
    }


def make_status(state: str, uptime_ms: int, error_code: int = 0) -> dict[str, Any]:
    return {
        "v": PROTOCOL_VERSION, "type": "status", "state": state,
        "error_code": error_code, "error": ERROR_NAMES.get(error_code, "UNKNOWN"),
        "uptime_ms": uptime_ms,
    }


def make_alert(
    event_id: int,
    event: str,
    sample_index: int,
    *,
    beat_class: str = "-",
    confidence: float = 0.0,
    simulated: bool = False,
) -> dict[str, Any]:
    return {
        "v": PROTOCOL_VERSION, "type": "alert", "event_id": event_id, "event": event,
        "beat_class": beat_class, "confidence": round(float(confidence), 3),
        "sample_index": sample_index, "simulated": simulated,
    }


def make_command(cmd: str, cmd_id: int, **extra: Any) -> dict[str, Any]:
    return {"v": PROTOCOL_VERSION, "type": "cmd", "cmd": cmd, "cmd_id": cmd_id, **extra}


def make_cmd_ack(cmd_id: int, ok: bool, error_code: int = 0) -> dict[str, Any]:
    return {"v": PROTOCOL_VERSION, "type": "cmd_ack", "cmd_id": cmd_id, "ok": ok, "error_code": error_code}


# ---------------------------------------------------------------------------
# Receptor de referencia (lado central / app)
# ---------------------------------------------------------------------------

@dataclass
class ReceivedChunk:
    first_sample_index: int
    samples: tuple[int, ...]
    lost_samples_before: int
    flags: int


@dataclass
class ReceiverStats:
    packets_ok: int = 0
    packets_lost: int = 0
    packets_duplicate_or_old: int = 0
    packets_invalid: int = 0
    samples_ok: int = 0
    samples_lost: int = 0


@dataclass
class EcgStreamReceiver:
    """Reconstruye el flujo de muestras a partir de paquetes ECG_DATA.

    * Un salto en ``seq`` indica paquetes perdidos; las muestras perdidas se
      calculan con ``first_sample_index`` (exacto aunque cambie el MTU).
    * Paquetes con ``seq`` anterior al esperado se descartan (duplicados o
      desordenados): BLE garantiza orden dentro de una conexión, así que solo
      deberían aparecer tras una reconexión.
    """

    stats: ReceiverStats = field(default_factory=ReceiverStats)
    _expected_seq: int | None = None
    _expected_index: int | None = None

    def reset(self) -> None:
        """Llamar al reconectar: el dispositivo puede reiniciar seq/índices."""
        self._expected_seq = None
        self._expected_index = None

    def feed(self, data: bytes) -> ReceivedChunk | None:
        try:
            pkt = decode_ecg_packet(data)
        except ProtocolError:
            self.stats.packets_invalid += 1
            return None

        lost_samples = 0
        if self._expected_seq is not None:
            seq_gap = (pkt.seq - self._expected_seq) % _SEQ_MOD
            if seq_gap >= _SEQ_MOD // 2:
                self.stats.packets_duplicate_or_old += 1
                return None
            self.stats.packets_lost += seq_gap
            lost_samples = (pkt.first_sample_index - self._expected_index) % _IDX_MOD
            if lost_samples >= _IDX_MOD // 2:  # índice retrocede: flujo reiniciado
                lost_samples = 0
            self.stats.samples_lost += lost_samples

        self._expected_seq = (pkt.seq + 1) % _SEQ_MOD
        self._expected_index = (pkt.first_sample_index + len(pkt.samples)) % _IDX_MOD
        self.stats.packets_ok += 1
        self.stats.samples_ok += len(pkt.samples)
        return ReceivedChunk(pkt.first_sample_index, pkt.samples, lost_samples, pkt.flags)

    def feed_many(self, packets: Iterable[bytes]) -> list[ReceivedChunk]:
        return [c for c in map(self.feed, packets) if c is not None]


# ---------------------------------------------------------------------------
# Reconexión y monitor de enlace
# ---------------------------------------------------------------------------

_RP = CONFIG["reconnect_policy"]


@dataclass
class ReconnectPolicy:
    """Backoff exponencial: 1, 2, 4, 8... s hasta ``max_delay_s``."""

    initial_delay_s: float = _RP["initial_delay_s"]
    max_delay_s: float = _RP["max_delay_s"]
    multiplier: float = _RP["multiplier"]
    max_attempts: int = _RP["max_attempts"]
    attempt: int = 0

    def next_delay(self) -> float | None:
        """Retardo antes del siguiente intento, o ``None`` si se agotaron."""
        if self.attempt >= self.max_attempts:
            return None
        delay = min(self.initial_delay_s * self.multiplier ** self.attempt, self.max_delay_s)
        self.attempt += 1
        return delay

    def reset(self) -> None:
        self.attempt = 0


@dataclass
class LinkMonitor:
    """Declara el enlace perdido si no llegan datos durante ``timeout_s``."""

    timeout_s: float = _RP["link_timeout_s"]
    last_rx_s: float | None = None

    def on_data(self, now_s: float) -> None:
        self.last_rx_s = now_s

    def is_lost(self, now_s: float) -> bool:
        return self.last_rx_s is not None and (now_s - self.last_rx_s) > self.timeout_s
