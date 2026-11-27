"""Cliente BLE de escritorio (protocolo BLE v1) sobre bleak.

No contiene lógica de interfaz: entrega eventos por callbacks (se invocan en el
hilo del bucle asyncio; la GUI debe pasarlos a su propio hilo). Reutiliza el
códec de referencia ``communication/protocol/ecg_protocol.py`` sin modificarlo.

Flujo (ble_spec.md §7): escanear (filtro por Service UUID) -> conectar ->
comprobar MTU -> leer DEVICE_INFO -> suscribirse a ECG_DATA, ALERT y STATUS ->
escribir ``start_stream``.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine

from bleak import BleakClient, BleakScanner

# El protocolo vive en <repo>/communication; se importa sin copiarlo.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from communication.protocol import ecg_protocol as proto  # noqa: E402

CHAR = {name: info["uuid"] for name, info in proto.CHARACTERISTICS.items()}
MIN_MTU = proto.MIN_MTU_FOR_JSON  # 185: necesario para JSON en una notificación


@dataclass
class FoundDevice:
    name: str
    address: str
    rssi: int | None
    service_uuids: list[str] = field(default_factory=list)


class BleClientError(RuntimeError):
    pass


class EcgBleClient:
    """Conexión con un periférico ECG. Seguro para llamarse desde otros hilos."""

    def __init__(self) -> None:
        self.on_ecg: Callable[[proto.ReceivedChunk], None] = lambda c: None
        self.on_status: Callable[[dict[str, Any]], None] = lambda m: None
        self.on_alert: Callable[[dict[str, Any]], None] = lambda m: None
        self.on_info: Callable[[dict[str, Any]], None] = lambda m: None
        self.on_disconnected: Callable[[], None] = lambda: None
        self.on_log: Callable[[str], None] = lambda s: None

        self.receiver = proto.EcgStreamReceiver()
        self.device_info: dict[str, Any] | None = None
        self.mtu: int | None = None
        self.connected = False
        self._client: BleakClient | None = None
        self._cmd_id = 0
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="ble-loop", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ hilo
    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Coroutine[Any, Any, Any], timeout: float | None = None) -> Any:
        """Ejecuta una corrutina en el bucle BLE y espera el resultado."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def submit(self, coro: Coroutine[Any, Any, Any]) -> "asyncio.futures.Future[Any]":
        """Ejecuta sin esperar (para la GUI)."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def close(self) -> None:
        try:
            self.run(self.disconnect(), timeout=10)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)

    # ---------------------------------------------------------------- escaneo
    async def scan(self, timeout: float = 6.0) -> list[FoundDevice]:
        found: dict[str, FoundDevice] = {}
        seen = await BleakScanner.discover(timeout=timeout, return_adv=True)
        for address, (dev, adv) in seen.items():
            uuids = [u.lower() for u in (adv.service_uuids or [])]
            name = adv.local_name or dev.name or ""
            is_ours = proto.SERVICE_UUID.lower() in uuids or name.startswith(proto.CONFIG["device"]["name_prefix"])
            if is_ours:
                found[address] = FoundDevice(name, address, adv.rssi, uuids)
        return sorted(found.values(), key=lambda d: -(d.rssi or -999))

    # --------------------------------------------------------------- conexión
    async def connect(self, address: str, start_stream: bool = True, timeout: float = 15.0) -> dict[str, Any]:
        if self.connected:
            await self.disconnect()
        self.receiver.reset()  # ble_spec.md §4: al reconectar se reinicia el receptor
        client = BleakClient(address, disconnected_callback=self._handle_disconnect, timeout=timeout)
        await client.connect()
        self._client = client
        self.connected = True
        try:
            self.mtu = getattr(client, "mtu_size", None)
            self.on_log(f"Conectado. MTU={self.mtu}")
            if self.mtu is not None and self.mtu < MIN_MTU:
                raise BleClientError(f"MTU {self.mtu} < {MIN_MTU}: el dispositivo rechazará start_stream (MTU_TOO_SMALL)")
            info = proto.decode_json(bytes(await client.read_gatt_char(CHAR["DEVICE_INFO"])))
            proto.validate_message(info)
            self.device_info = info
            self.on_info(info)
            try:
                st = proto.decode_json(bytes(await client.read_gatt_char(CHAR["STATUS"])))
                self.on_status(st)
            except Exception as exc:  # la lectura inicial de STATUS no es crítica
                self.on_log(f"No se pudo leer STATUS: {exc}")
            await client.start_notify(CHAR["STATUS"], self._on_status_raw)
            await client.start_notify(CHAR["ALERT"], self._on_alert_raw)
            await client.start_notify(CHAR["ECG_DATA"], self._on_ecg_raw)
            if start_stream:
                await self.send_command("start_stream")
        except Exception:
            await self.disconnect()
            raise
        return info

    async def disconnect(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                if client.is_connected:
                    try:
                        await self._write_cmd(client, "stop_stream")
                    except Exception:
                        pass
                    await client.disconnect()
            finally:
                pass
        was = self.connected
        self.connected = False
        if was:
            self.on_disconnected()

    async def send_command(self, cmd: str, **extra: Any) -> int:
        if self._client is None or not self._client.is_connected:
            raise BleClientError("no conectado")
        return await self._write_cmd(self._client, cmd, **extra)

    async def _write_cmd(self, client: BleakClient, cmd: str, **extra: Any) -> int:
        self._cmd_id += 1
        msg = proto.make_command(cmd, self._cmd_id, **extra)
        await client.write_gatt_char(CHAR["CONTROL"], proto.encode_json(msg), response=True)
        return self._cmd_id

    # -------------------------------------------------------------- callbacks
    def _handle_disconnect(self, _client: BleakClient) -> None:
        if self.connected:
            self.connected = False
            self.on_disconnected()

    def _on_ecg_raw(self, _char: Any, data: bytearray) -> None:
        chunk = self.receiver.feed(bytes(data))
        if chunk is not None:
            self.on_ecg(chunk)

    def _on_status_raw(self, _char: Any, data: bytearray) -> None:
        self._dispatch_json(data, self.on_status)

    def _on_alert_raw(self, _char: Any, data: bytearray) -> None:
        self._dispatch_json(data, self.on_alert)

    def _dispatch_json(self, data: bytearray, handler: Callable[[dict[str, Any]], None]) -> None:
        try:
            msg = proto.decode_json(bytes(data))
            proto.validate_message(msg)
        except Exception as exc:
            self.on_log(f"Mensaje JSON inválido ({exc}): {bytes(data)[:60]!r}")
            return
        handler(msg)


def now() -> float:
    return time.monotonic()
