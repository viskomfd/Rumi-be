from typing import Any, Dict
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.supabase import db
from app.schemas.push import PushSubscriptionCreate, PushUnsubscribeRequest
from app.services.push_service import push_service

router = APIRouter(prefix="/push", tags=["Web Push Notifications"])


@router.get("/vapid-public-key")
async def get_vapid_public_key() -> Dict[str, Any]:
    """Mengembalikan VAPID public key agar PWA browser dapat memanggil pushManager.subscribe()."""
    return {
        "public_key": settings.VAPID_PUBLIC_KEY.strip(),
        "enabled": push_service.is_configured,
    }


@router.post("/subscribe", status_code=status.HTTP_201_CREATED)
async def subscribe_push(
    payload: PushSubscriptionCreate,
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Menyimpan subscription Web Push milik perangkat pendamping (PWA / Browser)."""
    if not push_service.is_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Web Push belum dikonfigurasi di server (VAPID keys kosong).",
        )

    user_agent = payload.user_agent or request.headers.get("user-agent", "")
    saved = await db.save_push_subscription(
        user_id=current_user["id"],
        endpoint=payload.endpoint,
        p256dh=payload.keys.p256dh,
        auth=payload.keys.auth,
        user_agent=user_agent,
    )
    return {
        "status": "subscribed",
        "subscription_id": saved.get("id"),
        "endpoint": saved.get("endpoint"),
    }


@router.delete("/subscribe", status_code=status.HTTP_200_OK)
async def unsubscribe_push(
    payload: PushUnsubscribeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Menghapus subscription Web Push perangkat saat user mematikan notifikasi atau logout."""
    removed = await db.delete_push_subscription(
        endpoint=payload.endpoint,
        user_id=current_user["id"],
    )
    return {
        "status": "unsubscribed",
        "removed": removed,
    }


@router.post("/test", status_code=status.HTTP_200_OK)
async def send_test_push(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """Mengirim 1 push notification uji coba ke semua perangkat terdaftar milik user."""
    if not push_service.is_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Web Push belum dikonfigurasi di server (VAPID keys kosong).",
        )

    elderly_name = current_user.get("elderly_name") or "Lansia"
    result = await push_service.send_to_user(
        user_id=current_user["id"],
        payload={
            "type": "FALL_ALERT_TEST",
            "title": "Tes Notifikasi Push Rumi",
            "body": f"Notifikasi siaga 24/7 untuk pemantauan {elderly_name} aktif di perangkat ini.",
            "url": "/dashboard",
            "tag": "rumi-push-test",
        },
    )
    return {
        "message": (
            f"Push uji coba terkirim ke {result['sent']} perangkat."
            if result["sent"] > 0
            else "Belum ada perangkat yang berlangganan Web Push untuk akun ini."
        ),
        **result,
    }
