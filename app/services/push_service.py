import asyncio
import json
import logging
from typing import Any, Dict, Optional

from pywebpush import WebPushException, webpush

from app.core.config import settings
from app.core.supabase import db

logger = logging.getLogger("rumi.push")


class PushService:
    """Mengirim Web Push (VAPID) ke perangkat pendamping meskipun PWA/browser tertutup."""

    @property
    def is_configured(self) -> bool:
        return bool(settings.VAPID_PUBLIC_KEY.strip() and settings.VAPID_PRIVATE_KEY.strip())

    def _send_sync(self, subscription_info: Dict[str, Any], payload_json: str) -> Optional[int]:
        """Eksekusi synchronous webpush di thread terpisah; kembalikan status code HTTP jika gagal."""
        try:
            webpush(
                subscription_info=subscription_info,
                data=payload_json,
                vapid_private_key=settings.VAPID_PRIVATE_KEY.strip(),
                vapid_claims={"sub": settings.VAPID_CLAIM_EMAIL.strip() or "mailto:admin@rumi.app"},
                ttl=300,
            )
            return None
        except WebPushException as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            logger.warning(f"[WebPush] Gagal kirim push (HTTP {status_code}): {exc}")
            return status_code
        except Exception as exc:
            logger.warning(f"[WebPush] Error kirim push: {exc}")
            return -1

    async def send_to_user(self, user_id: str, payload: Dict[str, Any]) -> Dict[str, int]:
        """Mengirim push notification ke seluruh subscription perangkat milik user."""
        if not self.is_configured:
            logger.debug("[WebPush] VAPID keys belum dikonfigurasi; push dilewati.")
            return {"sent": 0, "failed": 0, "removed": 0}

        subs = await db.get_push_subscriptions(user_id=user_id)
        if not subs:
            return {"sent": 0, "failed": 0, "removed": 0}

        payload_json = json.dumps(payload)
        sent = 0
        failed = 0
        removed = 0

        for sub in subs:
            endpoint = sub.get("endpoint")
            p256dh = sub.get("p256dh")
            auth = sub.get("auth")
            if not endpoint or not p256dh or not auth:
                continue

            sub_info = {
                "endpoint": endpoint,
                "keys": {"p256dh": p256dh, "auth": auth},
            }
            err_status = await asyncio.to_thread(self._send_sync, sub_info, payload_json)
            if err_status is None:
                sent += 1
            else:
                failed += 1
                # 404 Not Found / 410 Gone menandakan token browser/HP sudah dicabut atau kadaluarsa
                if err_status in (404, 410):
                    await db.delete_push_subscription(endpoint=endpoint, user_id=user_id)
                    removed += 1

        return {"sent": sent, "failed": failed, "removed": removed}

    async def send_fall_alert(self, user_id: Optional[str], incident: Dict[str, Any]) -> Dict[str, int]:
        """Membentuk payload ringan khusus notifikasi jatuh untuk Service Worker."""
        if not user_id:
            return {"sent": 0, "failed": 0, "removed": 0}

        person = incident.get("person_name") or "Lansia"
        camera_name = incident.get("camera_name") or "Kamera CCTV"
        location = incident.get("location") or "Ruangan"
        confidence = incident.get("confidence", 0.92)
        try:
            conf_pct = round(float(confidence) * 100)
        except (TypeError, ValueError):
            conf_pct = 92

        payload = {
            "type": "FALL_ALERT",
            "title": "Kejadian Jatuh Terdeteksi!",
            "body": f"{person} terdeteksi jatuh di {location} ({camera_name} · {conf_pct}%).",
            "url": "/dashboard",
            "incident_id": incident.get("id"),
            "camera_name": camera_name,
            "location": location,
            "snapshot_url": incident.get("snapshot_url"),
            "tag": f"rumi-fall-{incident.get('id') or 'alert'}",
        }
        return await self.send_to_user(user_id=user_id, payload=payload)


push_service = PushService()
