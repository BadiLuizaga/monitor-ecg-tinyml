import json
import re
from pathlib import Path

import pytest

from protocol import ecg_protocol as p

COMM_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = COMM_DIR.parent


# --- paquete binario -------------------------------------------------------

def test_packet_roundtrip():
    samples = [0, 1, -1, 32767, -32768, 1234]
    data = p.encode_ecg_packet(7, 123456, samples, p.FLAGS["SIMULATED"])
    assert len(data) == p.HEADER_SIZE + 2 * len(samples)
    pkt = p.decode_ecg_packet(data)
    assert (pkt.seq, pkt.first_sample_index, pkt.samples) == (7, 123456, tuple(samples))
    assert pkt.has_flag("SIMULATED") and not pkt.has_flag("LEAD_OFF")


def test_packet_golden_bytes():
    """Vector de referencia compartido con el firmware (test_host.cpp)."""
    data = p.encode_ecg_packet(0x0102, 0x0A0B0C0D, [1, -2, 300], 0x04)
    assert data.hex() == "01010201" "0d0c0b0a" "0304" "0100" "feff" "2c01"


@pytest.mark.parametrize("mtu,expected", [(23, 5), (185, 86), (247, 117), (517, 252)])
def test_samples_per_mtu(mtu, expected):
    assert p.max_samples_per_packet(mtu) == expected
    assert len(p.encode_ecg_packet(0, 0, [0] * expected)) <= mtu - p.ATT_OVERHEAD


def test_seq_and_index_wrap():
    pkt = p.decode_ecg_packet(p.encode_ecg_packet(65536 + 5, 2 ** 32 + 9, [1]))
    assert (pkt.seq, pkt.first_sample_index) == (5, 9)


@pytest.mark.parametrize("bad", [
    b"",
    b"\x01\x01\x00",
    p.encode_ecg_packet(0, 0, [1, 2])[:-1],          # truncado
    b"\x02" + p.encode_ecg_packet(0, 0, [1])[1:],    # versión
    b"\x01\x09" + p.encode_ecg_packet(0, 0, [1])[2:],  # tipo
])
def test_invalid_packets_rejected(bad):
    with pytest.raises(p.ProtocolError):
        p.decode_ecg_packet(bad)


@pytest.mark.parametrize("samples", [[], [40000], [0] * 256])
def test_encode_rejects_out_of_range(samples):
    with pytest.raises(p.ProtocolError):
        p.encode_ecg_packet(0, 0, samples)


def test_packetize_covers_all_samples():
    samples = list(range(-500, 500))
    pkts = list(p.packetize(samples, first_sample_index=10, first_seq=3, mtu=23))
    decoded = [p.decode_ecg_packet(b) for b in pkts]
    assert [d.seq for d in decoded] == list(range(3, 3 + len(pkts)))
    assert sum((d.samples for d in decoded), ()) == tuple(samples)
    assert decoded[1].first_sample_index == 10 + p.max_samples_per_packet(23)


# --- receptor --------------------------------------------------------------

def _stream(n_packets, per=10, start_seq=0):
    return [p.encode_ecg_packet(start_seq + k, k * per, list(range(per))) for k in range(n_packets)]


def test_receiver_detects_lost_packets():
    pkts = _stream(10)
    rx = p.EcgStreamReceiver()
    chunks = rx.feed_many(pkts[:3] + pkts[5:])
    assert rx.stats.packets_lost == 2 and rx.stats.samples_lost == 20
    assert chunks[3].lost_samples_before == 20
    assert rx.stats.samples_ok == 80


def test_receiver_discards_duplicates_and_invalid():
    pkts = _stream(4)
    rx = p.EcgStreamReceiver()
    rx.feed_many([pkts[0], pkts[1], pkts[1], pkts[0], b"\x00", pkts[2]])
    assert rx.stats.packets_duplicate_or_old == 2 and rx.stats.packets_invalid == 1
    assert rx.stats.packets_lost == 0 and rx.stats.packets_ok == 3


def test_receiver_handles_seq_wraparound():
    pkts = _stream(4, start_seq=65534)
    rx = p.EcgStreamReceiver()
    rx.feed_many(pkts)
    assert rx.stats.packets_ok == 4 and rx.stats.packets_lost == 0


def test_receiver_reset_after_reconnect():
    rx = p.EcgStreamReceiver()
    rx.feed_many(_stream(5, start_seq=100))
    rx.reset()
    rx.feed_many(_stream(2, start_seq=0))  # el dispositivo reinició contadores
    assert rx.stats.packets_ok == 7 and rx.stats.packets_duplicate_or_old == 0


# --- JSON ------------------------------------------------------------------

def _examples():
    return [
        p.make_device_info("ECG-TINYML-1A2B", "0.1.0", 360, adc_bits=12, mv_per_count=0.0012),
        p.make_status("streaming", 1234),
        p.make_status("connected", 99, p.ERROR_CODES["LEAD_OFF"]),
        p.make_alert(42, "arrhythmia_suspected", 123456789, beat_class="V", confidence=0.91234),
        p.make_alert(43, "lead_off", 5),
        p.make_command("start_stream", 1),
        p.make_cmd_ack(1, True),
    ]


@pytest.mark.parametrize("msg", _examples(), ids=lambda m: m["type"])
def test_json_roundtrip_and_size(msg):
    data = p.encode_json(msg)
    assert p.JSON_MAX_BYTES <= p.MIN_MTU_FOR_JSON - p.ATT_OVERHEAD
    limit = p.JSON_MAX_READ_BYTES if msg["type"] == "device_info" else p.JSON_MAX_BYTES
    assert len(data) <= limit
    assert p.decode_json(data) == json.loads(data)


@pytest.mark.parametrize("mutate", [
    lambda m: m.pop("event_id"),
    lambda m: m.update(confidence=1.5),
    lambda m: m.update(beat_class="X"),
    lambda m: m.update(event="otra_cosa"),
    lambda m: m.update(v=99),
    lambda m: m.update(event_id=True),
    lambda m: m.update(type="desconocido"),
])
def test_invalid_alert_rejected(mutate):
    msg = p.make_alert(1, "arrhythmia_suspected", 0, beat_class="V", confidence=0.5)
    mutate(msg)
    with pytest.raises(p.ProtocolError):
        p.validate_message(msg)


def test_invalid_command_and_json():
    with pytest.raises(p.ProtocolError):
        p.validate_message(p.make_command("format_disk", 1))
    with pytest.raises(p.ProtocolError):
        p.decode_json(b"{no es json")
    with pytest.raises(p.ProtocolError):
        p.encode_json(p.make_device_info("X" * 400, "0", 360))  # > 512 bytes
    with pytest.raises(p.ProtocolError):
        p.encode_json(p.make_alert(1, "lead_off", 10 ** 160))  # notificación > 180 bytes


# --- reconexión ------------------------------------------------------------

def test_reconnect_backoff():
    rp = p.ReconnectPolicy(initial_delay_s=1, max_delay_s=8, multiplier=2, max_attempts=6)
    assert [rp.next_delay() for _ in range(7)] == [1, 2, 4, 8, 8, 8, None]
    rp.reset()
    assert rp.next_delay() == 1


def test_link_monitor():
    lm = p.LinkMonitor(timeout_s=3)
    assert not lm.is_lost(100)  # nunca recibió datos
    lm.on_data(10)
    assert not lm.is_lost(12.9) and lm.is_lost(13.1)


# --- coherencia entre artefactos ------------------------------------------

def test_schema_matches_codec():
    schema = json.loads((COMM_DIR / "protocol" / "message_schema.json").read_text(encoding="utf-8"))
    for mtype, fields in p.REQUIRED_FIELDS.items():
        assert set(schema["$defs"][mtype]["required"]) == set(fields), mtype
    assert schema["$defs"]["status"]["properties"]["state"]["enum"] == list(p.DEVICE_STATES)
    assert schema["$defs"]["cmd"]["properties"]["cmd"]["enum"] == list(p.COMMANDS)
    assert set(schema["$defs"]["alert"]["properties"]["event"]["enum"]) == set(p.ALERT_EVENTS)


def test_example_files_are_valid():
    examples = json.loads((COMM_DIR / "examples" / "sample_messages.json").read_text(encoding="utf-8"))
    for ex in examples["json_messages"]:
        p.validate_message(ex["message"])
    for ex in examples["ecg_packets"]:
        pkt = p.decode_ecg_packet(bytes.fromhex(ex["hex"]))
        assert list(pkt.samples) == ex["decoded"]["samples"]
        assert pkt.seq == ex["decoded"]["seq"]
    p.validate_message(json.loads((COMM_DIR / "protocol" / "alert_message.json").read_text(encoding="utf-8")))


def test_firmware_header_matches_config():
    header = (REPO_DIR / "device-firmware" / "include" / "integration_config.h").read_text(encoding="utf-8")

    def define(name):
        m = re.search(rf"#define\s+{name}\s+(\S+)", header)
        assert m, f"falta {name} en integration_config.h"
        return m.group(1).strip('"')

    assert define("ECG_PROTOCOL_VERSION") == str(p.PROTOCOL_VERSION)
    assert define("ECG_SERVICE_UUID") == p.SERVICE_UUID
    for name, spec in p.CHARACTERISTICS.items():
        assert define(f"ECG_CHAR_{name}_UUID") == spec["uuid"]
    assert define("ECG_PACKET_HEADER_SIZE") == str(p.HEADER_SIZE)
    assert define("ECG_JSON_MAX_BYTES") == str(p.JSON_MAX_BYTES)
    assert define("ECG_MIN_MTU_FOR_JSON") == str(p.MIN_MTU_FOR_JSON)
    for name, value in p.FLAGS.items():
        assert define(f"ECG_FLAG_{name}") == str(value)
    errors = (REPO_DIR / "device-firmware" / "include" / "ble_protocol.h").read_text(encoding="utf-8")
    for name, code in p.ERROR_CODES.items():
        assert re.search(rf"\b{name}\s*=\s*{code}\b", errors), f"código {name} distinto en ble_protocol.h"
