from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.alert_manager import alert_manager
from app.core.security import decode_access_token
from app.core.supabase import db
import logging

logger = logging.getLogger("rumi.ws")

router = APIRouter(tags=["WebSocket"])

@router.websocket("/ws/alerts")
async def websocket_alerts_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        auth_message = await websocket.receive_json()
        if auth_message.get("type") != "AUTH" or not auth_message.get("token"):
            await websocket.close(code=1008)
            return

        payload = decode_access_token(auth_message["token"])
        user = await db.get_user_by_id(payload.get("id")) if payload else None
        if not user or int(payload.get("ver", 0)) != int(user.get("token_version", 0)):
            await websocket.close(code=1008)
            return
        alert_manager.connect(websocket, user["id"])

        while True:
            # Tetap jaga koneksi tetap hidup dengan mendengarkan ping/pong
            await websocket.receive_text()
            # Echo heartbeat
            await websocket.send_text('{"type": "PONG"}')
    except WebSocketDisconnect:
        alert_manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"[WebSocket] Exception: {e}")
        alert_manager.disconnect(websocket)
