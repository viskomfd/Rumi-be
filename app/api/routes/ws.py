from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.alert_manager import alert_manager
from app.services.device_stream_relay import device_stream_relay
from app.core.security import decode_access_token
from app.core.supabase import db
import json
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


@router.websocket("/ws/device-stream/{camera_id}")
async def websocket_device_stream_publisher(websocket: WebSocket, camera_id: str):
    """
    Publisher WebSocket untuk Kamera Device (HP/Laptop pemilik kamera).
    - Autentikasi via query param `?token=<jwt>` ATAU pesan teks JSON pertama `{"type": "AUTH", "token": "<jwt>"}`.
    - Setelah terautentikasi, kirim frame JPEG binary (disarankan 15 FPS, lebar max 360px, quality 0.6).
    - Viewer di perangkat lain (akun sama) menonton via GET /api/cameras/{camera_id}/stream.
    """
    await websocket.accept()
    token = websocket.query_params.get("token")

    try:
        if not token:
            first_msg = await websocket.receive_text()
            auth_payload = json.loads(first_msg)
            if auth_payload.get("type") != "AUTH" or not auth_payload.get("token"):
                await websocket.close(code=1008)
                return
            token = auth_payload["token"]

        payload = decode_access_token(token)
        user = await db.get_user_by_id(payload.get("id")) if payload else None
        if not user or int(payload.get("ver", 0)) != int(user.get("token_version", 0)):
            await websocket.close(code=1008)
            return

        camera = await db.get_camera_by_id(camera_id)
        if not camera or camera.get("user_id") != user["id"]:
            await websocket.close(code=1008)
            return

        device_stream_relay.mark_connected(camera_id, websocket)
        await websocket.send_text(json.dumps({"type": "READY", "camera_id": camera_id}))

        while True:
            msg = await websocket.receive()
            msg_type = msg.get("type")
            if msg_type == "websocket.disconnect":
                break
            if msg.get("bytes"):
                device_stream_relay.publish_frame(camera_id, msg["bytes"])
            elif msg.get("text"):
                if msg["text"].strip().upper() == "PING":
                    await websocket.send_text('{"type": "PONG"}')
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"[DeviceStreamWS] Exception ({camera_id}): {e}")
    finally:
        device_stream_relay.mark_disconnected(camera_id, websocket)
