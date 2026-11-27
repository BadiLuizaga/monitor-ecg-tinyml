"""Sonda BLE por consola: comprueba el ESP32 real (solo lectura + start_stream).

Uso:  python tools/ble_probe.py [--seconds 8] [--address XX:XX:...]
No escribe nada en el dispositivo salvo los comandos de CONTROL del protocolo
(start_stream / stop_stream).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecg_desktop.ble_client import CHAR, EcgBleClient, proto  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--address")
    ap.add_argument("--scan-seconds", type=float, default=8.0)
    args = ap.parse_args()

    c = EcgBleClient()
    stats = {"chunks": 0, "samples": 0, "status": [], "alerts": [], "flags": set(), "lost": 0, "min": 32767, "max": -32768}
    c.on_log = lambda s: print("[log]", s)
    c.on_status = lambda m: stats["status"].append(m)
    c.on_alert = lambda m: stats["alerts"].append(m)

    def on_ecg(ch):
        stats["chunks"] += 1
        stats["samples"] += len(ch.samples)
        stats["lost"] += ch.lost_samples_before
        stats["flags"].add(ch.flags)
        stats["min"] = min(stats["min"], min(ch.samples))
        stats["max"] = max(stats["max"], max(ch.samples))

    c.on_ecg = on_ecg
    try:
        print(f"== Escaneando {args.scan_seconds:.0f} s (filtro: Service UUID {proto.SERVICE_UUID}) ==")
        devs = c.run(c.scan(args.scan_seconds), timeout=args.scan_seconds + 15)
        for d in devs:
            print(f"  {d.name!r} {d.address} rssi={d.rssi} uuids={d.service_uuids}")
        if not devs and not args.address:
            print("RESULTADO: no se encontró ningún dispositivo ECG-TINYML")
            return 2
        addr = args.address or devs[0].address
        print(f"== Conectando a {addr} ==")
        info = c.run(c.connect(addr, start_stream=True), timeout=40)
        print("MTU:", c.mtu)
        print("DEVICE_INFO:", info)
        t0 = time.time()
        while time.time() - t0 < args.seconds:
            time.sleep(0.25)
        c.run(c.send_command("ping"), timeout=10)
        c.run(c.send_command("get_status"), timeout=10)
        time.sleep(1.0)
        c.run(c.disconnect(), timeout=15)
        dur = args.seconds
        fs = info["sample_rate_hz"]
        print(f"== ECG: paquetes={stats['chunks']} muestras={stats['samples']} "
              f"(esperadas ~{fs * dur:.0f}) perdidas={stats['lost']} rango=[{stats['min']},{stats['max']}] "
              f"flags={sorted(stats['flags'])}")
        print("Receptor:", c.receiver.stats)
        print("== STATUS recibidos ==")
        for m in stats["status"]:
            print("  ", m)
        print("== ALERT recibidas ==", len(stats["alerts"]))
        for m in stats["alerts"][:5]:
            print("  ", m)
        return 0
    finally:
        c.close()


if __name__ == "__main__":
    raise SystemExit(main())
