"""Simulador de dispositivo ECG para probar el software sin hardware.

AVISO: esto es una SIMULACIÓN. No hay radio BLE ni ESP32 involucrados. El
simulador produce exactamente los mismos bytes que definirá el firmware
(paquetes binarios ECG_DATA y mensajes JSON), pero los entrega por un
transporte de prueba:

* ``InMemoryTransport``: para pruebas automáticas.
* ``FileTransport``: graba una captura binaria reproducible.
* ``TcpTransport``: servidor en ``127.0.0.1`` para consumir el flujo desde
  otro proceso (otra herramienta, un adaptador de depuración de la app...).

Formato de trama de los transportes de prueba (NO forma parte del protocolo
BLE, solo encapsula una notificación/escritura GATT):
``u8 char_id | u16 longitud (LE) | payload``.

Todas las alertas generadas aquí provienen de las etiquetas de la señal
sintética (verdad conocida), NO de un modelo de IA, y llevan
``"simulated": true``.

Uso rápido::

    python communication/simulator/mock_ecg_device.py selftest --seed 42
    python communication/simulator/mock_ecg_device.py record --out captures/demo.bin
    python communication/simulator/mock_ecg_device.py serve-tcp --port 8765
    python communication/simulator/mock_ecg_device.py client-tcp --port 8765
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

if __package__ in (None, ""):  # ejecución directa como script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from protocol import ecg_protocol as proto  # noqa: E402

FRAME_HEADER = struct.Struct("<BH")
SIM_FW_VERSION = "sim-0.1.0"
# Escala de cuantización SOLO de la simulación (no es la del ADC real).
SIM_MV_PER_COUNT = 0.005


# ---------------------------------------------------------------------------
# Fuentes de señal
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Wave:
    offset_s: float   # posición respecto al pico R
    amp_mv: float
    width_s: float


NORMAL_BEAT = (
    Wave(-0.20, 0.15, 0.025),   # P
    Wave(-0.035, -0.10, 0.010),  # Q
    Wave(0.0, 1.00, 0.012),     # R
    Wave(0.035, -0.25, 0.012),  # S
    Wave(0.25, 0.30, 0.045),    # T
)
# Latido ventricular prematuro: sin onda P, QRS ancho, T invertida.
PVC_BEAT = (
    Wave(0.0, 1.30, 0.035),
    Wave(0.07, -0.45, 0.030),
    Wave(0.28, -0.40, 0.060),
)


@dataclass(frozen=True)
class Annotation:
    sample_index: int
    symbol: str  # 'N' normal, 'V' ventricular (convención MIT-BIH)


class EcgSource(Protocol):
    sample_rate_hz: float

    def read(self, n: int) -> tuple[np.ndarray, list[Annotation]]:
        """Devuelve ``n`` muestras (mV) y las anotaciones contenidas en ellas."""


@dataclass
class SyntheticEcgSource:
    """ECG sintético basado en suma de gaussianas por onda (P, QRS, T).

    Reproducible: con la misma ``seed`` siempre se generan las mismas muestras
    y anotaciones. No es un modelo fisiológico validado; sirve para probar la
    cadena de datos y la detección de eventos con una verdad conocida.
    """

    sample_rate_hz: float = 360.0
    heart_rate_bpm: float = 72.0
    hrv_std: float = 0.03
    pvc_probability: float = 0.0
    noise_mv: float = 0.02
    baseline_mv: float = 0.10
    baseline_hz: float = 0.25
    powerline_mv: float = 0.0
    powerline_hz: float = 50.0
    seed: int = 0

    def __post_init__(self) -> None:
        # Generadores separados para ritmo y ruido: así la señal no depende del
        # tamaño de los bloques leídos (reproducibilidad exacta).
        rhythm_seed, noise_seed = np.random.SeedSequence(self.seed).spawn(2)
        self._rng = np.random.default_rng(rhythm_seed)
        self._noise_rng = np.random.default_rng(noise_seed)
        self._beats: list[tuple[float, str]] = []  # (instante R en s, símbolo)
        self._next_r_s = 0.5
        self._prev_pvc = False
        self._pos = 0  # índice de la siguiente muestra a entregar

    def _extend_beats(self, until_s: float) -> None:
        rr_mean = 60.0 / self.heart_rate_bpm
        while self._next_r_s < until_s:
            is_pvc = (not self._prev_pvc) and self._rng.random() < self.pvc_probability
            # El PVC llega antes de tiempo; el siguiente latido mantiene el ritmo
            # base, lo que produce la pausa compensatoria.
            r_s = self._next_r_s - (0.3 * rr_mean if is_pvc else 0.0)
            self._beats.append((r_s, "V" if is_pvc else "N"))
            self._next_r_s += rr_mean * (1.0 + self.hrv_std * self._rng.standard_normal())
            self._prev_pvc = is_pvc

    def read(self, n: int) -> tuple[np.ndarray, list[Annotation]]:
        fs = self.sample_rate_hz
        start, end = self._pos, self._pos + n
        t = np.arange(start, end) / fs
        self._extend_beats(end / fs + 1.0)
        x = np.zeros(n)
        anns: list[Annotation] = []
        for r_s, sym in self._beats:
            if r_s < t[0] - 1.0 or r_s > t[-1] + 1.0:
                continue
            for w in (PVC_BEAT if sym == "V" else NORMAL_BEAT):
                c = r_s + w.offset_s
                lo, hi = np.searchsorted(t, [c - 5 * w.width_s, c + 5 * w.width_s])
                if lo < hi:
                    x[lo:hi] += w.amp_mv * np.exp(-0.5 * ((t[lo:hi] - c) / w.width_s) ** 2)
            idx = int(round(r_s * fs))
            if start <= idx < end:
                anns.append(Annotation(idx, sym))
        # Olvidar latidos que ya no influyen en bloques futuros.
        self._beats = [b for b in self._beats if b[0] > t[-1] - 1.0]
        x += self.baseline_mv * np.sin(2 * np.pi * self.baseline_hz * t)
        x += self.powerline_mv * np.sin(2 * np.pi * self.powerline_hz * t)
        x += self.noise_mv * self._noise_rng.standard_normal(n)
        self._pos = end
        return x, anns


class RecordReplaySource:
    """Reproduce un registro WFDB (p. ej. MIT-BIH) con sus anotaciones.

    Requiere ``wfdb`` y el registro descargado (ver ai-model/src/training/dataset_loader.py).
    """

    def __init__(self, record_path: str | Path, channel: int = 0, loop: bool = True) -> None:
        import wfdb  # import diferido: dependencia opcional del simulador

        rec = wfdb.rdrecord(str(record_path), channels=[channel])
        ann = wfdb.rdann(str(record_path), "atr")
        self.sample_rate_hz = float(rec.fs)
        self._signal = rec.p_signal[:, 0].astype(float)
        self._ann = [Annotation(int(s), sym) for s, sym in zip(ann.sample, ann.symbol)]
        self._pos = 0
        self._total = 0
        self._loop = loop

    def read(self, n: int) -> tuple[np.ndarray, list[Annotation]]:
        out = np.empty(n)
        anns: list[Annotation] = []
        filled = 0
        while filled < n:
            if self._pos >= self._signal.size:
                if not self._loop:
                    out[filled:] = 0.0
                    break
                self._pos = 0
            take = min(n - filled, self._signal.size - self._pos)
            out[filled:filled + take] = self._signal[self._pos:self._pos + take]
            offset = self._total + filled - self._pos
            anns += [Annotation(a.sample_index + offset, a.symbol) for a in self._ann
                     if self._pos <= a.sample_index < self._pos + take]
            self._pos += take
            filled += take
        self._total += n
        return out, anns


# ---------------------------------------------------------------------------
# Transportes de prueba (NO son BLE)
# ---------------------------------------------------------------------------

class Transport(Protocol):
    def send(self, char_name: str, payload: bytes) -> None: ...
    def event(self, name: str) -> None: ...


@dataclass
class InMemoryTransport:
    frames: list[tuple[str, bytes]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)

    def send(self, char_name: str, payload: bytes) -> None:
        self.frames.append((char_name, payload))

    def event(self, name: str) -> None:
        self.events.append(name)

    def payloads(self, char_name: str) -> list[bytes]:
        return [p for c, p in self.frames if c == char_name]


def encode_frame(char_name: str, payload: bytes) -> bytes:
    return FRAME_HEADER.pack(proto.CHARACTERISTICS[char_name]["char_id"], len(payload)) + payload


def iter_frames(data: bytes) -> Iterator[tuple[str, bytes]]:
    pos = 0
    while pos + FRAME_HEADER.size <= len(data):
        char_id, length = FRAME_HEADER.unpack_from(data, pos)
        pos += FRAME_HEADER.size
        if pos + length > len(data):
            break  # trama incompleta
        yield proto.CHAR_ID_TO_NAME[char_id], data[pos:pos + length]
        pos += length


class FileTransport:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "wb")

    def send(self, char_name: str, payload: bytes) -> None:
        self._fh.write(encode_frame(char_name, payload))

    def event(self, name: str) -> None:  # los eventos no se graban
        pass

    def close(self) -> None:
        self._fh.close()


class TcpTransport:
    """Servidor TCP local (un cliente). Solo escucha en 127.0.0.1."""

    def __init__(self, port: int, host: str = "127.0.0.1") -> None:
        self._srv = socket.create_server((host, port))
        self._conn: socket.socket | None = None
        self._rx = b""

    def wait_client(self, timeout_s: float | None = None) -> None:
        self._srv.settimeout(timeout_s)
        self._conn, _ = self._srv.accept()
        self._conn.setblocking(False)

    @property
    def connected(self) -> bool:
        return self._conn is not None

    def send(self, char_name: str, payload: bytes) -> None:
        if self._conn is None:
            return
        try:
            self._conn.setblocking(True)
            self._conn.sendall(encode_frame(char_name, payload))
            self._conn.setblocking(False)
        except OSError:
            self._conn = None

    def event(self, name: str) -> None:
        if name == "disconnected" and self._conn is not None:
            self._conn.close()
            self._conn = None

    def poll_writes(self) -> list[tuple[str, bytes]]:
        """Escrituras recibidas del cliente (CONTROL)."""
        if self._conn is None:
            return []
        try:
            chunk = self._conn.recv(4096)
            if chunk == b"":
                self._conn = None
                return []
            self._rx += chunk
        except BlockingIOError:
            pass
        except OSError:
            self._conn = None
            return []
        frames = list(iter_frames(self._rx))
        consumed = sum(FRAME_HEADER.size + len(p) for _, p in frames)
        self._rx = self._rx[consumed:]
        return frames

    def close(self) -> None:
        if self._conn:
            self._conn.close()
        self._srv.close()


# ---------------------------------------------------------------------------
# Dispositivo simulado
# ---------------------------------------------------------------------------

@dataclass
class FaultConfig:
    """Errores de comunicación/señal a inyectar (todos reproducibles por seed)."""

    drop_probability: float = 0.0
    corrupt_probability: float = 0.0
    disconnect_at_s: tuple[float, ...] = ()
    lead_off_intervals_s: tuple[tuple[float, float], ...] = ()
    seed: int = 1


ABNORMAL_SYMBOLS = {"V": "V", "A": "S", "S": "S", "a": "S", "J": "S", "E": "V", "F": "F"}


class MockEcgDevice:
    """Máquina de estados equivalente a la prevista para el firmware.

    advertising --connect()--> connected --start_stream--> streaming
    cualquier estado --disconnect()--> advertising
    """

    def __init__(
        self,
        source: EcgSource,
        transport: Transport,
        *,
        device_id: str = "SIM-0001",
        mv_per_count: float = SIM_MV_PER_COUNT,
        faults: FaultConfig | None = None,
        alert_cooldown_s: float = 2.0,
    ) -> None:
        self.source = source
        self.transport = transport
        self.device_id = device_id
        self.mv_per_count = mv_per_count
        self.faults = faults or FaultConfig()
        self.alert_cooldown_s = alert_cooldown_s
        self._fault_rng = np.random.default_rng(self.faults.seed)
        self.state = "advertising"
        self.error_code = proto.ERROR_CODES["OK"]
        self.mtu = proto.DEFAULT_MTU
        self.sample_index = 0
        self.seq = 0
        self.event_id = 0
        self._last_alert_s = -1e9
        self._pending_disconnects = sorted(self.faults.disconnect_at_s)
        self._lead_off = False
        self.ground_truth: list[Annotation] = []
        self.sent_packets = 0
        self.dropped_packets = 0
        self.dropped_samples = 0

    # -- tiempo de la simulación (basado en muestras, no en reloj real) --
    @property
    def time_s(self) -> float:
        return self.sample_index / self.source.sample_rate_hz

    @property
    def uptime_ms(self) -> int:
        return int(self.time_s * 1000)

    # -- GATT simulado --
    def connect(self, mtu: int = 247) -> None:
        self.mtu = mtu
        self._set_state("connected")
        self.transport.event("connected")

    def disconnect(self) -> None:
        self._set_state("advertising", notify=False)
        self.transport.event("disconnected")

    def read(self, char_name: str) -> bytes:
        if char_name == "DEVICE_INFO":
            return proto.encode_json(proto.make_device_info(
                self.device_id, SIM_FW_VERSION, self.source.sample_rate_hz,
                hw_model="SIMULATOR", mv_per_count=self.mv_per_count, simulated=True))
        if char_name == "STATUS":
            return proto.encode_json(proto.make_status(self.state, self.uptime_ms, self.error_code))
        raise proto.ProtocolError(f"característica no legible: {char_name}")

    def write_control(self, payload: bytes) -> None:
        cmd_id = -1
        try:
            msg = proto.decode_json(payload)
            if msg["type"] != "cmd":
                raise proto.ProtocolError("se esperaba un comando")
            cmd_id = msg["cmd_id"]
        except proto.ProtocolError:
            self._send_json("STATUS", proto.make_cmd_ack(cmd_id, False, proto.ERROR_CODES["INVALID_COMMAND"]))
            return
        if self.state == "advertising":
            return  # sin conexión no hay escrituras
        cmd = msg["cmd"]
        if cmd == "start_stream":
            if self.mtu < proto.MIN_MTU_FOR_JSON:
                self._send_json("STATUS", proto.make_cmd_ack(cmd_id, False, proto.ERROR_CODES["MTU_TOO_SMALL"]))
                return
            self._set_state("streaming")
        elif cmd == "stop_stream":
            self._set_state("connected")
        elif cmd == "get_status":
            self._send_json("STATUS", proto.make_status(self.state, self.uptime_ms, self.error_code))
        # ping y ack_alert solo requieren el acuse.
        self._send_json("STATUS", proto.make_cmd_ack(cmd_id, True))

    # -- bucle principal --
    def tick(self, n_samples: int) -> None:
        """Avanza ``n_samples`` de señal. La señal avanza aunque no haya conexión."""
        signal_mv, anns = self.source.read(n_samples)
        start_index = self.sample_index
        t0 = self.time_s
        self.sample_index += n_samples
        self.ground_truth.extend(anns)

        lead_off = any(a <= t0 < b for a, b in self.faults.lead_off_intervals_s)
        if lead_off:
            signal_mv = np.zeros_like(signal_mv)
        counts = [proto.clamp_int16(v / self.mv_per_count) for v in signal_mv]
        saturated = any(c in (proto.INT16_MIN, proto.INT16_MAX) for c in counts)

        if self.state == "streaming":
            flags = proto.FLAGS["SIMULATED"]
            flags |= proto.FLAGS["LEAD_OFF"] if lead_off else 0
            flags |= proto.FLAGS["SATURATED"] if saturated else 0
            self._handle_lead_off(lead_off, start_index)
            for pkt in proto.packetize(counts, start_index, self.seq, self.mtu, flags):
                self.seq = (self.seq + 1) % (1 << 16)
                self._send_ecg(pkt)
            if not lead_off:
                self._emit_alerts(anns)

        while self._pending_disconnects and self.time_s >= self._pending_disconnects[0]:
            self._pending_disconnects.pop(0)
            if self.state != "advertising":
                self.disconnect()

    def _send_ecg(self, pkt: bytes) -> None:
        if self._fault_rng.random() < self.faults.drop_probability:
            self.dropped_packets += 1
            self.dropped_samples += pkt[8]  # campo sample_count de la cabecera
            return
        if self._fault_rng.random() < self.faults.corrupt_probability:
            pkt = pkt[: len(pkt) - 1]  # truncado: el receptor debe descartarlo
            self.dropped_samples += pkt[8]
        self.sent_packets += 1
        self.transport.send("ECG_DATA", pkt)

    def _handle_lead_off(self, lead_off: bool, index: int) -> None:
        if lead_off == self._lead_off:
            return
        self._lead_off = lead_off
        self.error_code = proto.ERROR_CODES["LEAD_OFF" if lead_off else "OK"]
        self._send_json("STATUS", proto.make_status(self.state, self.uptime_ms, self.error_code))
        if lead_off:
            self.event_id += 1
            self._send_json("ALERT", proto.make_alert(self.event_id, "lead_off", index, simulated=True))

    def _emit_alerts(self, anns: list[Annotation]) -> None:
        fs = self.source.sample_rate_hz
        for a in anns:
            cls = ABNORMAL_SYMBOLS.get(a.symbol)
            if cls is None or a.sample_index / fs - self._last_alert_s < self.alert_cooldown_s:
                continue
            self._last_alert_s = a.sample_index / fs
            self.event_id += 1
            self._send_json("ALERT", proto.make_alert(
                self.event_id, "arrhythmia_suspected", a.sample_index,
                beat_class=cls, confidence=1.0, simulated=True))

    def _set_state(self, state: str, notify: bool = True) -> None:
        self.state = state
        if notify:
            self._send_json("STATUS", proto.make_status(state, self.uptime_ms, self.error_code))

    def _send_json(self, char_name: str, msg: dict) -> None:
        self.transport.send(char_name, proto.encode_json(msg))


# ---------------------------------------------------------------------------
# Lado central de referencia (consume el flujo)
# ---------------------------------------------------------------------------

@dataclass
class CentralSummary:
    stats: proto.ReceiverStats
    alerts: list[dict]
    statuses: list[dict]
    invalid_json: int = 0
    samples: list[int] = field(default_factory=list)


def consume_frames(frames: list[tuple[str, bytes]], keep_samples: bool = False) -> CentralSummary:
    rx = proto.EcgStreamReceiver()
    summary = CentralSummary(rx.stats, [], [])
    for char_name, payload in frames:
        if char_name == "ECG_DATA":
            chunk = rx.feed(payload)
            if chunk and keep_samples:
                summary.samples.extend([0] * chunk.lost_samples_before)
                summary.samples.extend(chunk.samples)
            continue
        try:
            msg = proto.decode_json(payload)
        except proto.ProtocolError:
            summary.invalid_json += 1
            continue
        (summary.alerts if msg["type"] == "alert" else summary.statuses).append(msg)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_source(args: argparse.Namespace) -> EcgSource:
    if args.record:
        return RecordReplaySource(args.record)
    return SyntheticEcgSource(
        sample_rate_hz=args.fs, heart_rate_bpm=args.hr, pvc_probability=args.pvc,
        noise_mv=args.noise, seed=args.seed)


def _faults(args: argparse.Namespace) -> FaultConfig:
    return FaultConfig(drop_probability=args.drop, corrupt_probability=args.corrupt, seed=args.seed + 1)


def cmd_selftest(args: argparse.Namespace) -> int:
    transport = InMemoryTransport()
    dev = MockEcgDevice(_build_source(args), transport, faults=_faults(args))
    dev.connect(mtu=args.mtu)
    dev.write_control(proto.encode_json(proto.make_command("start_stream", 1)))
    print("SIMULACIÓN (sin BLE real)")
    if dev.state != "streaming":
        ack = proto.decode_json(transport.payloads("STATUS")[-1])
        print(f"  start_stream rechazado: {proto.ERROR_NAMES[ack['error_code']]} "
              f"(MTU={args.mtu}, mínimo {proto.MIN_MTU_FOR_JSON})")
        return 2
    block = int(dev.source.sample_rate_hz * 0.1)
    for _ in range(int(args.duration / 0.1)):
        dev.tick(block)
    s = consume_frames(transport.frames)
    print(f"  muestras generadas : {dev.sample_index}")
    print(f"  paquetes enviados  : {dev.sent_packets} (perdidos/corruptos inyectados: "
          f"{dev.dropped_packets}/{s.stats.packets_invalid})")
    print(f"  receptor           : {s.stats}")
    print(f"  alertas            : {len(s.alerts)}  latidos anómalos reales: "
          f"{sum(a.symbol in ABNORMAL_SYMBOLS for a in dev.ground_truth)}")
    print(f"  mensajes STATUS    : {len(s.statuses)}")
    # Todo lo recibido debe ser exactamente lo no perdido, y las pérdidas
    # detectadas no pueden superar las inyectadas (las del final del flujo
    # son indetectables para el receptor).
    ok = (s.stats.samples_ok == dev.sample_index - dev.dropped_samples
          and s.stats.samples_lost <= dev.dropped_samples)
    print(f"  integridad         : {'OK' if ok else 'ERROR'} "
          f"(perdidas inyectadas={dev.dropped_samples}, detectadas={s.stats.samples_lost})")
    return 0 if ok else 1


def cmd_record(args: argparse.Namespace) -> int:
    transport = FileTransport(args.out)
    dev = MockEcgDevice(_build_source(args), transport, faults=_faults(args))
    dev.connect(mtu=args.mtu)
    transport.send("DEVICE_INFO", dev.read("DEVICE_INFO"))
    dev.write_control(proto.encode_json(proto.make_command("start_stream", 1)))
    block = int(dev.source.sample_rate_hz * 0.1)
    for _ in range(int(args.duration / 0.1)):
        dev.tick(block)
    transport.close()
    print(f"Captura simulada guardada en {args.out} ({Path(args.out).stat().st_size} bytes)")
    return 0


def cmd_serve_tcp(args: argparse.Namespace) -> int:
    transport = TcpTransport(args.port)
    print(f"[SIMULADOR] escuchando en 127.0.0.1:{args.port} (TCP de prueba, NO es BLE)")
    try:
        while True:
            dev = MockEcgDevice(_build_source(args), transport, faults=_faults(args))
            transport.wait_client()
            print("[SIMULADOR] cliente conectado")
            dev.connect(mtu=args.mtu)
            transport.send("DEVICE_INFO", dev.read("DEVICE_INFO"))
            period = 0.05
            block = max(1, int(dev.source.sample_rate_hz * period))
            next_t = time.monotonic()
            while transport.connected:
                for char_name, payload in transport.poll_writes():
                    if char_name == "CONTROL":
                        dev.write_control(payload)
                dev.tick(block)
                next_t += period
                time.sleep(max(0.0, next_t - time.monotonic()))
            print("[SIMULADOR] cliente desconectado; esperando otro")
    except KeyboardInterrupt:
        pass
    finally:
        transport.close()
    return 0


def cmd_client_tcp(args: argparse.Namespace) -> int:
    """Central de prueba con reconexión (backoff) sobre el transporte TCP."""
    policy = proto.ReconnectPolicy()
    rx = proto.EcgStreamReceiver()
    alerts = 0
    deadline = time.monotonic() + args.duration
    while time.monotonic() < deadline:
        try:
            sock = socket.create_connection(("127.0.0.1", args.port), timeout=2)
        except OSError:
            delay = policy.next_delay()
            if delay is None:
                print("[CENTRAL] reintentos agotados")
                return 1
            print(f"[CENTRAL] sin conexión; reintento en {delay:.0f} s")
            time.sleep(delay)
            continue
        policy.reset()
        rx.reset()
        sock.sendall(encode_frame("CONTROL", proto.encode_json(proto.make_command("start_stream", 1))))
        buf = b""
        sock.settimeout(proto.LinkMonitor().timeout_s)
        try:
            while time.monotonic() < deadline:
                chunk = sock.recv(4096)
                if not chunk:
                    raise ConnectionError("el dispositivo cerró la conexión")
                buf += chunk
                frames = list(iter_frames(buf))
                buf = buf[sum(FRAME_HEADER.size + len(p) for _, p in frames):]
                for char_name, payload in frames:
                    if char_name == "ECG_DATA":
                        rx.feed(payload)
                    elif char_name == "ALERT":
                        alerts += 1
                        print(f"[CENTRAL] ALERTA {proto.decode_json(payload)}")
        except (OSError, ConnectionError) as exc:
            print(f"[CENTRAL] enlace perdido: {exc}")
        finally:
            sock.close()
    print(f"[CENTRAL] {rx.stats}, alertas={alerts}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("selftest", "record", "serve-tcp", "client-tcp"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--duration", type=float, default=30.0, help="segundos de simulación")
        p.add_argument("--fs", type=float, default=360.0, help="frecuencia de muestreo SIMULADA (Hz)")
        p.add_argument("--hr", type=float, default=72.0, help="frecuencia cardiaca (lpm)")
        p.add_argument("--pvc", type=float, default=0.05, help="probabilidad de latido ventricular prematuro")
        p.add_argument("--noise", type=float, default=0.02, help="ruido blanco (mV)")
        p.add_argument("--mtu", type=int, default=247)
        p.add_argument("--drop", type=float, default=0.0, help="probabilidad de perder un paquete")
        p.add_argument("--corrupt", type=float, default=0.0, help="probabilidad de corromper un paquete")
        p.add_argument("--record", help="ruta WFDB a reproducir (p. ej. ai-model/data/raw/mitdb/100)")
        p.add_argument("--port", type=int, default=8765)
        p.add_argument("--out", default="communication/simulator/captures/sim_capture.bin")
    args = parser.parse_args(argv)
    handlers = {"selftest": cmd_selftest, "record": cmd_record,
                "serve-tcp": cmd_serve_tcp, "client-tcp": cmd_client_tcp}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
