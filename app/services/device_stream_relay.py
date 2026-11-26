import asyncio
import logging
import time
from typing import Dict, Optional, Any

from fastapi import WebSocket

from app.core.supabase import db

logger = logging.getLogger("rumi.device_relay")

MAX_RELAY_FRAME_BYTES = 512 * 1024  # 512 KB safety cap per JPEG frame
RELAY_STALE_SECONDS = 8.0


class DeviceStreamRelay:
    """
    In-memory WebSocket-to-MJPEG relay untuk Kamera Device (HP/Laptop).
    - Publisher (HP pemilik kamera) mengirim frame JPEG ringan (~15 FPS, max 360px) lewat WS /ws/device-stream/{camera_id}.
    - Viewer (Laptop/HP lain di akun yang sama) menonton via GET /api/cameras/{camera_id}/stream (MJPEG).
    - 0% proses background saat tab/kamera ditutup: begitu WS putus, status langsung kembali 'offline'.
    """

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}

    def _schedule_db_status(self, camera_id: str, status_str: str) -> None:
        async def _sync():
            try:
                await db.update_camera_status(camera_id, status_str)
            except Exception as exc:
                logger.warning(f"[DeviceRelay] Gagal sinkron status DB '{camera_id}' -> {status_str}: {exc}")

        try:
            asyncio.create_task(_sync())
        except RuntimeError:
            pass

    def mark_connected(self, camera_id: str, websocket: WebSocket) -> None:
        now = time.time()
        existing = self._sessions.get(camera_id)
        if existing:
            existing["connected"] = True
            existing["ws"] = websocket
            existing["connected_at"] = now
            existing["last_frame_time"] = now
        else:
            self._sessions[camera_id] = {
                "connected": True,
                "ws": websocket,
                "latest_jpeg": None,
                "connected_at": now,
                "last_frame_time": now,
            }
        logger.info(f"[DeviceRelay] Publisher terhubung untuk kamera '{camera_id}'")
        self._schedule_db_status(camera_id, "online")

    def publish_frame(self, camera_id: str, jpeg_bytes: bytes) -> bool:
        if not jpeg_bytes or len(jpeg_bytes) > MAX_RELAY_FRAME_BYTES:
            return False
        session = self._sessions.get(camera_id)
        if not session:
            return False
        session["latest_jpeg"] = jpeg_bytes
        session["last_frame_time"] = time.time()
        session["connected"] = True
        return True

    def mark_disconnected(self, camera_id: str, websocket: Optional[WebSocket] = None) -> None:
        session = self._sessions.get(camera_id)
        if not session:
            return
        # Jangan matikan sesi baru jika yang putus adalah koneksi WS lama
        if websocket is not None and session.get("ws") is not websocket:
            return
        session["connected"] = False
        session["ws"] = None
        session["latest_jpeg"] = None
        logger.info(f"[DeviceRelay] Publisher terputus untuk kamera '{camera_id}'")
        self._schedule_db_status(camera_id, "offline")

    async def stop_camera(self, camera_id: str) -> None:
        session = self._sessions.get(camera_id)
        if not session:
            return
        ws: Optional[WebSocket] = session.get("ws")
        session["connected"] = False
        session["ws"] = None
        session["latest_jpeg"] = None
        if ws is not None:
            try:
                await ws.close(code=1000)
            except Exception:
                pass

    def is_online(self, camera_id: str) -> bool:
        session = self._sessions.get(camera_id)
        if not session or not session.get("connected"):
            return False
        return (time.time() - float(session.get("last_frame_time", 0.0))) < RELAY_STALE_SECONDS

    def get_latest_jpeg(self, camera_id: str) -> Optional[bytes]:
        if not self.is_online(camera_id):
            return None
        session = self._sessions.get(camera_id)
        return session.get("latest_jpeg") if session else None

    def get_live_status(self, camera_id: str, fallback_status: str = "offline") -> str:
        if camera_id in self._sessions:
            return "online" if self.is_online(camera_id) else "offline"
        normalized = str(fallback_status or "offline").lower()
        return "online" if normalized == "online" else "offline"


device_stream_relay = DeviceStreamRelay()
