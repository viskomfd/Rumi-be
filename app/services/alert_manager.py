import asyncio
import json
import logging
from typing import Dict, Any
from fastapi import WebSocket

from app.services.push_service import push_service

logger = logging.getLogger("rumi.alerts")

class AlertManager:
    def __init__(self):
        self.active_connections: Dict[WebSocket, str] = {}

    def connect(self, websocket: WebSocket, user_id: str):
        self.active_connections[websocket] = user_id
        logger.info(f"[WebSocket] Client terhubung. Total aktif: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]
            logger.info(f"[WebSocket] Client terputus. Sisa aktif: {len(self.active_connections)}")

    async def broadcast_fall_alert(self, incident: Dict[str, Any], user_id: str | None = None):
        """Mengirim pesan Fall Alert ke WebSocket aktif dan Web Push (PWA background)."""
        message = {
            "type": "FALL_ALERT",
            "title": "Kejadian Jatuh Terdeteksi!",
            "data": incident
        }
        dead_connections = []
        for connection, connected_user_id in list(self.active_connections.items()):
            if user_id and connected_user_id != user_id:
                continue
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"[WebSocket] Gagal kirim ke client: {e}")
                dead_connections.append(connection)

        for dead in dead_connections:
            self.disconnect(dead)

        # Kirim Web Push secara non-blocking agar HP/laptop tetap menerima notif meski browser ditutup
        if user_id:
            try:
                asyncio.create_task(push_service.send_fall_alert(user_id=user_id, incident=incident))
            except Exception as e:
                logger.warning(f"[WebPush] Gagal menjadwalkan push fall alert: {e}")

alert_manager = AlertManager()
