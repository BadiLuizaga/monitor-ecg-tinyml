"""Ventana principal (PySide6 + pyqtgraph)."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QMainWindow, QPushButton, QVBoxLayout, QWidget)

from . import theme
from .analysis import HeartRateEstimator, RPeakDetector
from .ble_client import EcgBleClient, FoundDevice, proto

WINDOW_S = 8.0  # segundos visibles en la gráfica
FLAG_LEAD_OFF = proto.FLAGS["LEAD_OFF"]
FLAG_SIMULATED = proto.FLAGS["SIMULATED"]
FLAG_SATURATED = proto.FLAGS["SATURATED"]

# Estados de conexión -> (texto, color de texto, color de fondo)
CONN_STYLES = {
    "idle": ("Desconectado", theme.MUTED, theme.NEUTRAL_BG),
    "scanning": ("Buscando…", theme.ACCENT, "#DBEAFE"),
    "connecting": ("Conectando…", theme.ACCENT, "#DBEAFE"),
    "connected": ("Conectado", theme.OK, theme.OK_BG),
    "streaming": ("Transmitiendo", theme.OK, theme.OK_BG),
    "lost": ("Enlace perdido", theme.WARN, theme.WARN_BG),
    "error": ("Error", theme.DANGER, theme.DANGER_BG),
}
STATE_ES = {"booting": "Iniciando", "advertising": "Anunciando", "connected": "Conectado",
            "streaming": "Transmitiendo", "error": "Error"}
EVENT_ES = {"arrhythmia_suspected": "Sospecha de arritmia", "lead_off": "Electrodo desconectado",
            "signal_quality": "Calidad de señal", "device_error": "Error del dispositivo"}


class Bridge(QObject):
    """Pasa los eventos del hilo BLE al hilo de la interfaz (señales en cola)."""
    ecg = Signal(object)
    status = Signal(dict)
    alert = Signal(dict)
    info = Signal(dict)
    disconnected = Signal()
    log = Signal(str)
    scanned = Signal(list)
    failed = Signal(str)
    connected = Signal(dict)


def card(title: str | None = None) -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName("card")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(16, 14, 16, 14)
    lay.setSpacing(8)
    if title:
        t = QLabel(title.upper())
        t.setObjectName("cardTitle")
        lay.addWidget(t)
    return f, lay


class KeyValue(QWidget):
    def __init__(self, keys: list[str]) -> None:
        super().__init__()
        g = QGridLayout(self)
        g.setContentsMargins(0, 0, 0, 0)
        g.setVerticalSpacing(5)
        self.values: dict[str, QLabel] = {}
        for i, k in enumerate(keys):
            kl = QLabel(k)
            kl.setObjectName("kvKey")
            vl = QLabel("—")
            vl.setObjectName("kv")
            vl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            vl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            g.addWidget(kl, i, 0)
            g.addWidget(vl, i, 1)
            self.values[k] = vl

    def set(self, key: str, value: Any) -> None:
        self.values[key].setText("—" if value is None else str(value))


class MainWindow(QMainWindow):
    def __init__(self, client: EcgBleClient | None = None) -> None:
        super().__init__()
        self.setWindowTitle("CardioAlert · Monitor ECG")
        self.resize(1280, 800)
        self.setStyleSheet(theme.STYLESHEET)

        self.bridge = Bridge()
        self.client = client or EcgBleClient()
        self._wire_client()

        self.fs = 360.0
        self.mv_per_count: float | None = None
        self.simulated_info: bool | None = None
        self.simulated_flag: bool | None = None
        self.detector = RPeakDetector(self.fs)
        self.hr = HeartRateEstimator(self.fs)
        self.devices: list[FoundDevice] = []
        self.conn = "idle"
        self.user_disconnect = False
        self.address: str | None = None
        self.policy = proto.ReconnectPolicy()
        self.link = proto.LinkMonitor()
        self.class_counts = {c: 0 for c in "NSVF"}
        self.model_loaded: bool | None = None
        self.lead_off = False
        self.rx_samples_window: list[tuple[float, int]] = []
        self.peaks: list[int] = []
        self._alert_ids: set[int] = set()
        self._reset_buffer()

        self._build_ui()
        self._set_conn("idle")
        self._refresh_source_badge()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)
        self.slow = QTimer(self)
        self.slow.timeout.connect(self._tick_slow)
        self.slow.start(500)

    # ------------------------------------------------------------------ cliente
    def _wire_client(self) -> None:
        c, b = self.client, self.bridge
        c.on_ecg = b.ecg.emit
        c.on_status = b.status.emit
        c.on_alert = b.alert.emit
        c.on_info = b.info.emit
        c.on_disconnected = b.disconnected.emit
        c.on_log = b.log.emit
        b.ecg.connect(self._on_ecg)
        b.status.connect(self._on_status)
        b.alert.connect(self._on_alert)
        b.info.connect(self._on_info)
        b.disconnected.connect(self._on_disconnected)
        b.log.connect(self._log)
        b.scanned.connect(self._on_scanned)
        b.failed.connect(self._on_failed)
        b.connected.connect(self._on_connected)

    # ----------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(14)

        # --- cabecera
        head = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(0)
        h1 = QLabel("CardioAlert · Monitor ECG")
        h1.setObjectName("h1")
        sub = QLabel("Receptor BLE v1 para laptop · proyecto académico, no es un dispositivo médico")
        sub.setObjectName("sub")
        titles.addWidget(h1)
        titles.addWidget(sub)
        head.addLayout(titles)
        head.addStretch(1)
        self.source_badge = QLabel()
        self.conn_pill = QLabel()
        head.addWidget(self.source_badge)
        head.addSpacing(8)
        head.addWidget(self.conn_pill)
        outer.addLayout(head)

        body = QHBoxLayout()
        body.setSpacing(14)
        outer.addLayout(body, 1)

        # --- columna izquierda
        left = QVBoxLayout()
        left.setSpacing(14)
        body.addLayout(left)

        dev_card, dl = card("Dispositivo")
        self.btn_scan = QPushButton("Buscar dispositivos")
        self.btn_scan.clicked.connect(self._scan)
        self.combo = QComboBox()
        self.combo.setPlaceholderText("Ningún dispositivo")
        self.btn_connect = QPushButton("Conectar")
        self.btn_connect.setObjectName("primary")
        self.btn_connect.clicked.connect(self._toggle_connection)
        self.btn_stream = QPushButton("Pausar transmisión")
        self.btn_stream.clicked.connect(self._toggle_stream)
        self.btn_stream.setEnabled(False)
        for w in (self.btn_scan, self.combo, self.btn_connect, self.btn_stream):
            dl.addWidget(w)
        self.info_kv = KeyValue(["ID", "Firmware", "Modelo HW", "Frecuencia", "Escala", "MTU", "RSSI"])
        dl.addSpacing(6)
        dl.addWidget(self.info_kv)
        left.addWidget(dev_card)

        st_card, sl = card("Estado del dispositivo")
        self.status_kv = KeyValue(["Estado", "Error", "Tiempo activo", "Electrodos"])
        sl.addWidget(self.status_kv)
        left.addWidget(st_card)

        link_card, ll = card("Calidad del enlace")
        self.link_kv = KeyValue(["Paquetes recibidos", "Paquetes perdidos", "Muestras perdidas", "Muestras/s"])
        ll.addWidget(self.link_kv)
        left.addWidget(link_card)
        left.addStretch(1)
        dev_card.setFixedWidth(290)
        st_card.setFixedWidth(290)
        link_card.setFixedWidth(290)

        # --- columna central
        main = QVBoxLayout()
        main.setSpacing(14)
        body.addLayout(main, 1)

        kpis = QHBoxLayout()
        kpis.setSpacing(14)
        main.addLayout(kpis)

        bpm_card, bl = card("Frecuencia cardíaca")
        row = QHBoxLayout()
        self.bpm_label = QLabel("—")
        self.bpm_label.setObjectName("bpm")
        unit = QLabel("lpm")
        unit.setObjectName("unit")
        unit.setAlignment(Qt.AlignBottom)
        row.addWidget(self.bpm_label)
        row.addWidget(unit)
        row.addStretch(1)
        bl.addLayout(row)
        self.bpm_note = QLabel("Calculada en el PC a partir de la señal")
        self.bpm_note.setObjectName("note")
        bl.addWidget(self.bpm_note)
        kpis.addWidget(bpm_card, 1)

        cls_card, cl = card("Clasificación de latidos")
        chips = QHBoxLayout()
        self.chip_counts: dict[str, QLabel] = {}
        self.chip_frames: dict[str, QFrame] = {}
        for c in "NSVF":
            f = QFrame()
            f.setStyleSheet(f"background:{theme.NEUTRAL_BG}; border-radius:10px;")
            fl = QVBoxLayout(f)
            fl.setContentsMargins(10, 6, 10, 6)
            fl.setSpacing(0)
            lab = QLabel(c)
            lab.setAlignment(Qt.AlignCenter)
            lab.setStyleSheet(f"font-size:20px; font-weight:800; color:{theme.CLASS_COLORS[c]}; background:transparent;")
            lab.setToolTip(theme.CLASS_NAMES[c])
            cnt = QLabel("0")
            cnt.setAlignment(Qt.AlignCenter)
            cnt.setStyleSheet(f"font-size:12px; color:{theme.MUTED}; background:transparent;")
            fl.addWidget(lab)
            fl.addWidget(cnt)
            chips.addWidget(f)
            self.chip_counts[c] = cnt
            self.chip_frames[c] = f
        cl.addLayout(chips)
        self.cls_note = QLabel("Sin datos")
        self.cls_note.setObjectName("note")
        self.cls_note.setWordWrap(True)
        cl.addWidget(self.cls_note)
        kpis.addWidget(cls_card, 1)

        sig_card, gl = card("Señal")
        self.sig_label = QLabel("—")
        self.sig_label.setStyleSheet("font-size:22px; font-weight:700;")
        self.sig_note = QLabel("")
        self.sig_note.setObjectName("note")
        self.sig_note.setWordWrap(True)
        gl.addWidget(self.sig_label)
        gl.addWidget(self.sig_note)
        gl.addStretch(1)
        kpis.addWidget(sig_card, 1)

        ecg_card, el = card("Electrocardiograma · derivación única")
        pg.setConfigOptions(antialias=True, background=theme.CARD, foreground=theme.MUTED)
        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(False, False)
        self.plot.hideButtons()
        self.plot.showGrid(x=True, y=True, alpha=0.5)
        self.plot.setXRange(-WINDOW_S, 0, padding=0)
        self.plot.setLabel("bottom", "Tiempo (s)")
        self.plot.getAxis("left").setStyle(tickTextOffset=6)
        self.curve = self.plot.plot(pen=pg.mkPen(theme.ECG, width=2), connect="finite")
        self.peak_item = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(theme.DANGER), pen=pg.mkPen("w", width=1.5))
        self.plot.addItem(self.peak_item)
        self.watermark = pg.TextItem("", anchor=(1, 1), color=theme.WARN)
        self.watermark.setFont(self.font())
        self.plot.addItem(self.watermark)
        el.addWidget(self.plot, 1)
        main.addWidget(ecg_card, 1)

        al_card, al = card("Alertas y eventos  ·  doble clic para confirmar (ack)")
        self.alerts = QListWidget()
        self.alerts.setFixedHeight(130)
        self.alerts.itemDoubleClicked.connect(self._ack_alert)
        al.addWidget(self.alerts)
        main.addWidget(al_card)

        self.statusBar().setStyleSheet(f"color:{theme.MUTED};")
        self.statusBar().showMessage("Listo. Pulsa «Buscar dispositivos».")

    # ---------------------------------------------------------------- utilidades
    def _log(self, text: str) -> None:
        self.statusBar().showMessage(text, 8000)

    def _set_conn(self, key: str) -> None:
        self.conn = key
        text, fg, bg = CONN_STYLES[key]
        self.conn_pill.setText("●  " + text)
        self.conn_pill.setStyleSheet(theme.pill_style(fg, bg))
        busy = key in ("scanning", "connecting")
        connected = key in ("connected", "streaming")
        self.btn_scan.setEnabled(not busy and not connected)
        self.combo.setEnabled(not busy and not connected)
        self.btn_connect.setEnabled(not busy and (connected or self.combo.count() > 0 or self.address is not None))
        self.btn_connect.setText("Desconectar" if connected else "Conectar")
        self.btn_connect.setObjectName("danger" if connected else "primary")
        self.btn_connect.setStyle(self.btn_connect.style())  # reaplica el estilo por objectName
        self.btn_stream.setEnabled(connected)

    def _is_simulated(self) -> bool | None:
        """Conservador: basta que el dispositivo o un paquete digan «simulado»."""
        if self.simulated_info is None and self.simulated_flag is None:
            return None
        return bool(self.simulated_info) or bool(self.simulated_flag)

    def _refresh_source_badge(self) -> None:
        sim = self._is_simulated()
        if sim is None:
            txt, fg, bg = "SIN DATOS", theme.MUTED, theme.NEUTRAL_BG
        elif sim:
            txt, fg, bg = "◆  DATOS SIMULADOS", theme.WARN, theme.WARN_BG
        else:
            txt, fg, bg = "●  DATOS REALES", theme.OK, theme.OK_BG
        self.source_badge.setText(txt)
        self.source_badge.setStyleSheet(theme.pill_style(fg, bg))
        self.watermark.setText("DATOS SIMULADOS · no es una señal fisiológica" if sim else "")

    def _reset_buffer(self) -> None:
        self.n = int(WINDOW_S * self.fs)
        self.buf = np.full(self.n, np.nan)
        self.last_index = -1
        self.peaks.clear()
        self.detector = RPeakDetector(self.fs)
        self.hr = HeartRateEstimator(self.fs)

    # -------------------------------------------------------------- acciones UI
    def _scan(self) -> None:
        self._set_conn("scanning")
        self.statusBar().showMessage("Buscando dispositivos ECG-TINYML (6 s)…")
        fut = self.client.submit(self.client.scan(6.0))
        fut.add_done_callback(lambda f: self._emit_result(f, self.bridge.scanned))

    def _emit_result(self, fut: Any, signal: Signal) -> None:
        try:
            signal.emit(fut.result())
        except Exception as exc:  # errores de bleak: se muestran, no se ocultan
            self.bridge.failed.emit(f"{type(exc).__name__}: {exc}")

    def _on_scanned(self, devs: list[FoundDevice]) -> None:
        self.devices = devs
        self.combo.clear()
        for d in devs:
            self.combo.addItem(f"{d.name or 'sin nombre'}   ({d.address}, {d.rssi} dBm)", d.address)
        if devs:
            self.combo.setCurrentIndex(0)  # con placeholder, addItem no selecciona nada
        self._set_conn("idle")
        self.statusBar().showMessage(f"{len(devs)} dispositivo(s) encontrado(s)." if devs else
                                     "No se encontró ningún ECG-TINYML. ¿Está encendido y anunciando?")

    def _toggle_connection(self) -> None:
        if self.conn in ("connected", "streaming"):
            self.user_disconnect = True
            self.client.submit(self.client.disconnect())
            return
        addr = self.combo.currentData() or self.address
        if addr:
            self.user_disconnect = False
            self.policy.reset()
            self._connect(addr)

    def _connect(self, addr: str) -> None:
        self.address = addr
        self._set_conn("connecting")
        self.statusBar().showMessage(f"Conectando a {addr}…")
        fut = self.client.submit(self.client.connect(addr))
        fut.add_done_callback(lambda f: self._emit_result(f, self.bridge.connected))

    def _on_connected(self, info: dict) -> None:
        self.policy.reset()
        self.link.last_rx_s = None
        self._set_conn("streaming")
        self.btn_stream.setText("Pausar transmisión")
        self.info_kv.set("MTU", self.client.mtu)
        self.statusBar().showMessage("Conectado. start_stream enviado.", 6000)

    def _on_failed(self, msg: str) -> None:
        if "not powered on" in msg or "POWERED_OFF" in msg:
            msg = "el Bluetooth de esta laptop está apagado. Actívalo (Win + A → Bluetooth) y vuelve a intentarlo."
        self._set_conn("error" if self.conn != "scanning" else "idle")
        self.statusBar().showMessage(f"Error: {msg}", 15000)
        if self.conn == "error":
            QTimer.singleShot(4000, lambda: self._set_conn("idle") if self.conn == "error" else None)

    def _toggle_stream(self) -> None:
        pausing = self.btn_stream.text().startswith("Pausar")
        self.client.submit(self.client.send_command("stop_stream" if pausing else "start_stream"))
        self.btn_stream.setText("Reanudar transmisión" if pausing else "Pausar transmisión")

    def _ack_alert(self, item: QListWidgetItem) -> None:
        event_id = item.data(Qt.UserRole)
        if event_id is not None and self.conn in ("connected", "streaming"):
            self.client.submit(self.client.send_command("ack_alert", event_id=int(event_id)))
            item.setText(item.text() + "   ✓ confirmada")

    # ------------------------------------------------------------ eventos BLE
    def _on_info(self, info: dict) -> None:
        self.fs = float(info["sample_rate_hz"])
        self.mv_per_count = info.get("mv_per_count")
        self.simulated_info = bool(info["simulated"])
        self.simulated_flag = None
        self._reset_buffer()
        self.info_kv.set("ID", info["device_id"])
        self.info_kv.set("Firmware", info["fw_version"])
        self.info_kv.set("Modelo HW", info["hw_model"])
        self.info_kv.set("Frecuencia", f"{self.fs:g} Hz")
        self.info_kv.set("Escala", f"{self.mv_per_count} mV/cuenta" if self.mv_per_count else "sin calibrar")
        addr_dev = next((d for d in self.devices if d.address == self.address), None)
        self.info_kv.set("RSSI", f"{addr_dev.rssi} dBm" if addr_dev and addr_dev.rssi is not None else None)
        self.plot.setLabel("left", "Amplitud (mV)" if self.mv_per_count else "Amplitud (cuentas, sin calibrar)")
        self._refresh_source_badge()

    def _on_status(self, st: dict) -> None:
        if st.get("type") == "cmd_ack":
            if not st["ok"]:
                self._log(f"El dispositivo rechazó el comando {st['cmd_id']} (código {st['error_code']})")
            return
        self.status_kv.set("Estado", STATE_ES.get(st["state"], st["state"]))
        self.status_kv.set("Error", f"{st['error']} ({st['error_code']})" if st["error_code"] else "Ninguno")
        up = int(st["uptime_ms"] // 1000)
        self.status_kv.set("Tiempo activo", f"{up // 60} min {up % 60:02d} s")
        self.model_loaded = st["error_code"] != 30
        self.lead_off = st["error_code"] == 10
        self._update_class_note()

    def _update_class_note(self) -> None:
        if self.model_loaded is None:
            self.cls_note.setText("Sin datos")
        elif not self.model_loaded:
            self.cls_note.setText("Sin clasificación: el modelo aún no está cargado en el dispositivo "
                                  "(MODEL_NOT_LOADED). Se mostrará al activar la inferencia.")
        else:
            self.cls_note.setText("Recuento de alertas por clase (el dispositivo solo avisa de las anómalas).")

    def _on_alert(self, a: dict) -> None:
        if a["event_id"] in self._alert_ids:
            return
        self._alert_ids.add(a["event_id"])
        t = time.strftime("%H:%M:%S")
        ev = EVENT_ES.get(a["event"], a["event"])
        cls = a.get("beat_class", "-")
        extra = f" · clase {cls} · {a['confidence'] * 100:.0f} %" if a["event"] == "arrhythmia_suspected" else ""
        sim = "   [simulada]" if a.get("simulated") else ""
        item = QListWidgetItem(f"{t}   {ev}{extra}{sim}")
        item.setData(Qt.UserRole, a["event_id"])
        color = theme.DANGER if a["event"] == "arrhythmia_suspected" else theme.WARN
        item.setForeground(pg.mkColor(color))
        self.alerts.insertItem(0, item)
        if a["event"] == "arrhythmia_suspected" and cls in self.class_counts:
            self.class_counts[cls] += 1
            self.chip_counts[cls].setText(str(self.class_counts[cls]))
            self.chip_frames[cls].setStyleSheet(f"background:{theme.DANGER_BG}; border-radius:10px;")
        if a["event"] == "lead_off":
            self.lead_off = True

    def _on_disconnected(self) -> None:
        self.link.last_rx_s = None
        self.simulated_flag = None
        if self.user_disconnect or self.address is None:
            self._set_conn("idle")
            self.statusBar().showMessage("Desconectado.", 6000)
        else:  # pérdida inesperada: reconexión con backoff (ble_spec.md §7)
            self._set_conn("lost")
            self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        delay = self.policy.next_delay()
        if delay is None:
            self._set_conn("error")
            self.statusBar().showMessage("No se pudo reconectar tras varios intentos.", 0)
            return
        self.statusBar().showMessage(f"Enlace perdido. Reintentando en {delay:g} s "
                                     f"(intento {self.policy.attempt}/{self.policy.max_attempts})…")
        QTimer.singleShot(int(delay * 1000), self._reconnect_now)

    def _reconnect_now(self) -> None:
        if self.user_disconnect or self.conn not in ("lost", "connecting") or not self.address:
            return
        self._set_conn("connecting")
        fut = self.client.submit(self.client.connect(self.address))

        def done(f: Any) -> None:
            try:
                self.bridge.connected.emit(f.result())
            except Exception:
                self.bridge.disconnected.emit()  # vuelve a programar el siguiente intento
        fut.add_done_callback(done)

    def _on_ecg(self, chunk: Any) -> None:
        n = len(chunk.samples)
        if n == 0:
            return
        self.link.on_data(time.monotonic())
        self.rx_samples_window.append((time.monotonic(), n))
        flags = chunk.flags
        self.simulated_flag = bool(flags & FLAG_SIMULATED)
        self.lead_off = bool(flags & FLAG_LEAD_OFF)
        self._refresh_source_badge()
        if self.conn == "lost":
            self._set_conn("streaming")

        lost = min(chunk.lost_samples_before, self.n)
        if lost:
            self.buf = np.roll(self.buf, -lost)
            self.buf[-lost:] = np.nan
            self.hr.reset()
        data = np.asarray(chunk.samples, dtype=float)
        if n >= self.n:
            self.buf[:] = data[-self.n:]
        else:
            self.buf = np.roll(self.buf, -n)
            self.buf[-n:] = data
        self.last_index = chunk.first_sample_index + n - 1

        for k, x in enumerate(chunk.samples):
            if flags & FLAG_LEAD_OFF:
                continue
            r = self.detector.process(float(x))
            if r is not None:
                self.hr.add_peak(r)
                # índice del detector (cuenta de muestras) -> índice de la ventana
                self.peaks.append(r)
        cutoff = self.detector.n - self.n
        self.peaks = [p for p in self.peaks if p >= cutoff]

    # ------------------------------------------------------------------ refresco
    def _scale(self) -> float:
        return self.mv_per_count if self.mv_per_count else 1.0

    def _tick(self) -> None:
        if self.last_index < 0:
            return
        t = np.arange(-self.n + 1, 1) / self.fs
        y = self.buf * self._scale()
        self.curve.setData(t, y)
        finite = y[np.isfinite(y)]
        if finite.size > 10:
            lo, hi = np.percentile(finite, [0.5, 99.5])
            span = max(hi - lo, 1e-9 if self.mv_per_count else 200.0)
            self.plot.setYRange(lo - 0.15 * span, hi + 0.25 * span, padding=0)
            vb = self.plot.getViewBox().viewRange()
            self.watermark.setPos(vb[0][1] - 0.1, vb[1][0])
        det_n = self.detector.n
        if self.peaks:
            xs = [(p - det_n + 1) / self.fs for p in self.peaks]
            ys = []
            for p in self.peaks:
                i = self.n - 1 - (det_n - 1 - p)
                ys.append(y[i] if 0 <= i < self.n and np.isfinite(y[i]) else np.nan)
            self.peak_item.setData(xs, ys)
        else:
            self.peak_item.setData([], [])

    def _tick_slow(self) -> None:
        now = time.monotonic()
        # BPM
        bpm = self.hr.bpm
        self.bpm_label.setText(f"{bpm:.0f}" if bpm and not self.lead_off else "—")
        self.bpm_note.setText("Calculada en el PC a partir de la señal" if bpm else
                              "Esperando suficientes latidos…" if self.conn in ("connected", "streaming") else
                              "Calculada en el PC a partir de la señal")
        # calidad de señal
        if self.conn in ("connected", "streaming") and self.last_index >= 0:
            if self.lead_off:
                self.sig_label.setText("Electrodo suelto")
                self.sig_label.setStyleSheet(f"font-size:22px; font-weight:700; color:{theme.DANGER};")
            else:
                self.sig_label.setText("Señal recibida")
                self.sig_label.setStyleSheet(f"font-size:22px; font-weight:700; color:{theme.OK};")
            self.sig_note.setText("Flujo continuo desde el dispositivo." if self.receiver_ok() else
                                  "Se están perdiendo paquetes.")
            self.status_kv.set("Electrodos", "Suelto (aviso del dispositivo)" if self.lead_off else "Sin aviso de electrodo suelto")
        else:
            self.sig_label.setText("—")
            self.sig_label.setStyleSheet("font-size:22px; font-weight:700;")
            self.sig_note.setText("")
        # enlace
        s = self.client.receiver.stats
        self.link_kv.set("Paquetes recibidos", s.packets_ok)
        self.link_kv.set("Paquetes perdidos", s.packets_lost)
        self.link_kv.set("Muestras perdidas", s.samples_lost)
        self.rx_samples_window = [(t, k) for t, k in self.rx_samples_window if now - t <= 3.0]
        rate = sum(k for _, k in self.rx_samples_window) / 3.0 if self.rx_samples_window else 0
        self.link_kv.set("Muestras/s", f"{rate:.0f}" if self.conn in ("connected", "streaming") else None)
        # pérdida de enlace por falta de datos (3 s) mientras se espera stream
        if (self.conn == "streaming" and self.btn_stream.text().startswith("Pausar")
                and self.link.is_lost(now)):
            self._set_conn("lost")
            self.statusBar().showMessage("Sin datos desde hace 3 s.", 0)

    def receiver_ok(self) -> bool:
        s = self.client.receiver.stats
        return s.packets_lost == 0 or s.packets_ok > 20 * s.packets_lost

    def closeEvent(self, event: Any) -> None:  # noqa: N802 (API de Qt)
        self.user_disconnect = True
        self.client.close()
        super().closeEvent(event)
