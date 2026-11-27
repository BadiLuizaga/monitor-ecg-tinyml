"""Punto de entrada:  python -m ecg_desktop   (desde la carpeta desktop-app)."""
from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from .window import MainWindow


def _selftest_ui(win: MainWindow, out_png: str) -> None:
    """Alimenta la ventana con paquetes SINTÉTICOS locales (sin BLE) y guarda una
    captura. Solo para comprobar el renderizado; marca los datos como simulados."""
    from PySide6.QtCore import QTimer

    from communication.protocol import ecg_protocol as proto
    from tests.test_analysis import pattern

    win._on_info({"v": 1, "type": "device_info", "device_id": "TEST", "fw_version": "selftest", "hw_model": "SIMULATOR",
                  "sample_rate_hz": 360, "sample_format": "int16", "adc_bits": None, "mv_per_count": None,
                  "leads": 1, "simulated": True})
    win._on_status({"v": 1, "type": "status", "state": "streaming", "error_code": 30, "error": "MODEL_NOT_LOADED",
                    "uptime_ms": 65000})
    win._set_conn("streaming")
    sig = pattern(360 * 12, 75)
    state = {"i": 0}

    def feed() -> None:
        i = state["i"]
        if i >= len(sig):
            return
        win._on_ecg(proto.ReceivedChunk(i, tuple(sig[i:i + 60]), 0, proto.FLAGS["SIMULATED"]))
        state["i"] = i + 60

    t = QTimer(win)
    t.timeout.connect(feed)
    t.start(5)
    win._on_alert({"v": 1, "type": "alert", "event_id": 1, "event": "arrhythmia_suspected", "beat_class": "V",
                   "confidence": 0.91, "sample_index": 1234, "simulated": True})

    def shot() -> None:
        win.grab().save(out_png)
        print("captura guardada:", out_png)
        QApplication.quit()

    QTimer.singleShot(3500, shot)


def main() -> int:
    ap = argparse.ArgumentParser(prog="ecg_desktop")
    ap.add_argument("--selftest-ui", metavar="PNG", help="renderiza con datos sintéticos locales y guarda una captura")
    args = ap.parse_args()
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    if args.selftest_ui:
        _selftest_ui(win, args.selftest_ui)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
