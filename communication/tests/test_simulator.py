import socket
import threading
import time

import numpy as np
import pytest

from protocol import ecg_protocol as p
from simulator import mock_ecg_device as sim


def _device(seed=42, faults=None, pvc=0.1, mtu=247, **kw):
    t = sim.InMemoryTransport()
    dev = sim.MockEcgDevice(sim.SyntheticEcgSource(seed=seed, pvc_probability=pvc), t, faults=faults, **kw)
    dev.connect(mtu=mtu)
    dev.write_control(p.encode_json(p.make_command("start_stream", 1)))
    return dev, t


def _run(dev, seconds, block=36):
    for _ in range(int(seconds * dev.source.sample_rate_hz / block)):
        dev.tick(block)


def test_synthetic_source_is_reproducible():
    a, ann_a = sim.SyntheticEcgSource(seed=5, pvc_probability=0.1).read(5000)
    b, ann_b = sim.SyntheticEcgSource(seed=5, pvc_probability=0.1).read(5000)
    c, _ = sim.SyntheticEcgSource(seed=6, pvc_probability=0.1).read(5000)
    assert np.array_equal(a, b) and ann_a == ann_b
    assert not np.array_equal(a, c)


def test_synthetic_source_block_size_independent():
    """Leer en bloques distintos debe dar la misma señal y anotaciones."""
    s1 = sim.SyntheticEcgSource(seed=9, pvc_probability=0.2, noise_mv=0.0)
    s2 = sim.SyntheticEcgSource(seed=9, pvc_probability=0.2, noise_mv=0.0)
    x1, a1 = s1.read(3600)
    parts = [s2.read(n) for n in (100, 1700, 7, 1793)]
    assert np.allclose(x1, np.concatenate([x for x, _ in parts]))
    assert a1 == [a for _, anns in parts for a in anns]


def test_heart_rate_is_respected():
    _, ann = sim.SyntheticEcgSource(heart_rate_bpm=90, seed=1).read(360 * 60)
    assert 85 <= len(ann) <= 95


def test_stream_integrity_and_alerts():
    dev, t = _device()
    _run(dev, 60)
    s = sim.consume_frames(t.frames, keep_samples=True)
    assert s.stats.samples_ok == dev.sample_index and s.stats.packets_lost == 0
    abnormal = [a for a in dev.ground_truth if a.symbol == "V"]
    assert abnormal and 1 <= len(s.alerts) <= len(abnormal)
    assert all(a["simulated"] and a["beat_class"] == "V" for a in s.alerts)
    # Cooldown: nunca dos alertas a menos de 2 s.
    times = [a["sample_index"] / 360 for a in s.alerts]
    assert all(b - a >= 2.0 for a, b in zip(times, times[1:]))
    assert all(pkt_flags for pkt_flags in [p.decode_ecg_packet(t.payloads("ECG_DATA")[0]).has_flag("SIMULATED")])


def test_samples_survive_quantization():
    dev, t = _device(pvc=0.0)
    _run(dev, 5)
    s = sim.consume_frames(t.frames, keep_samples=True)
    assert len(s.samples) == dev.sample_index
    assert max(map(abs, s.samples)) * dev.mv_per_count < 3.0  # amplitud ECG razonable en mV


def test_injected_packet_loss_is_detected():
    dev, t = _device(faults=sim.FaultConfig(drop_probability=0.1, seed=3))
    _run(dev, 30)
    s = sim.consume_frames(t.frames)
    assert dev.dropped_packets > 0
    assert s.stats.samples_ok == dev.sample_index - dev.dropped_samples
    assert s.stats.samples_lost <= dev.dropped_samples


def test_corrupted_packets_rejected():
    dev, t = _device(faults=sim.FaultConfig(corrupt_probability=0.1, seed=3))
    _run(dev, 10)
    s = sim.consume_frames(t.frames)
    assert s.stats.packets_invalid > 0


def test_lead_off_flag_status_and_alert():
    dev, t = _device(faults=sim.FaultConfig(lead_off_intervals_s=((2.0, 4.0),)))
    _run(dev, 6)
    flagged = [p.decode_ecg_packet(b).has_flag("LEAD_OFF") for b in t.payloads("ECG_DATA")]
    assert any(flagged) and not flagged[0] and not flagged[-1]
    s = sim.consume_frames(t.frames)
    assert [a["event"] for a in s.alerts].count("lead_off") == 1
    codes = [m.get("error_code") for m in s.statuses if m["type"] == "status"]
    assert p.ERROR_CODES["LEAD_OFF"] in codes and codes[-1] == 0


def test_scheduled_disconnect_stops_stream():
    dev, t = _device(faults=sim.FaultConfig(disconnect_at_s=(3.0,)))
    _run(dev, 6)
    assert dev.state == "advertising" and "disconnected" in t.events
    last = p.decode_ecg_packet(t.payloads("ECG_DATA")[-1])
    assert (last.first_sample_index + len(last.samples)) / 360 <= 3.2


def test_commands_and_acks():
    dev, t = _device()
    dev.write_control(p.encode_json(p.make_command("stop_stream", 2)))
    assert dev.state == "connected"
    dev.write_control(b"basura")
    acks = [p.decode_json(b) for b in t.payloads("STATUS")]
    acks = [m for m in acks if m["type"] == "cmd_ack"]
    assert [(a["cmd_id"], a["ok"]) for a in acks] == [(1, True), (2, True), (-1, False)]
    assert acks[-1]["error_code"] == p.ERROR_CODES["INVALID_COMMAND"]


def test_small_mtu_rejects_streaming():
    dev, t = _device(mtu=23)
    assert dev.state == "connected"
    ack = p.decode_json(t.payloads("STATUS")[-1])
    assert ack["ok"] is False and ack["error_code"] == p.ERROR_CODES["MTU_TOO_SMALL"]


def test_device_info_declares_simulation():
    dev, _ = _device()
    info = p.decode_json(dev.read("DEVICE_INFO"))
    assert info["simulated"] is True and info["hw_model"] == "SIMULATOR"
    assert info["sample_rate_hz"] == 360.0


def test_file_capture_roundtrip(tmp_path):
    ft = sim.FileTransport(tmp_path / "cap.bin")
    dev = sim.MockEcgDevice(sim.SyntheticEcgSource(seed=1), ft)
    dev.connect()
    dev.write_control(p.encode_json(p.make_command("start_stream", 1)))
    _run(dev, 2)
    ft.close()
    frames = list(sim.iter_frames((tmp_path / "cap.bin").read_bytes()))
    assert sim.consume_frames(frames).stats.samples_ok == dev.sample_index


def test_selftest_cli():
    assert sim.main(["selftest", "--duration", "5"]) == 0
    assert sim.main(["selftest", "--duration", "5", "--mtu", "23"]) == 2


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_tcp_transport_end_to_end():
    port = _free_port()
    tr = sim.TcpTransport(port)
    dev = sim.MockEcgDevice(sim.SyntheticEcgSource(seed=2), tr)

    def server():
        tr.wait_client(timeout_s=5)
        dev.connect()
        deadline = time.monotonic() + 5
        while dev.state != "streaming" and time.monotonic() < deadline:
            for name, payload in tr.poll_writes():
                dev.write_control(payload)
            time.sleep(0.01)
        _run(dev, 2)
        tr.close()

    th = threading.Thread(target=server)
    th.start()
    with socket.create_connection(("127.0.0.1", port), timeout=5) as c:
        c.sendall(sim.encode_frame("CONTROL", p.encode_json(p.make_command("start_stream", 1))))
        buf = b""
        while True:
            chunk = c.recv(65536)
            if not chunk:
                break
            buf += chunk
    th.join(timeout=10)
    s = sim.consume_frames(list(sim.iter_frames(buf)))
    assert s.stats.samples_ok == dev.sample_index > 0


@pytest.mark.parametrize("bad_frame", [b"\x01\x05\x00ab"])
def test_iter_frames_ignores_incomplete(bad_frame):
    assert list(sim.iter_frames(bad_frame)) == []
