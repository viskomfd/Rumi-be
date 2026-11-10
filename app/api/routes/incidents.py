import os

from fastapi import APIRouter, Query, status, Depends
from typing import List, Optional, Dict, Any
from app.schemas.incident import IncidentCreate, IncidentResponse
from app.core.supabase import db
from app.api.deps import get_current_user
from app.services.alert_manager import alert_manager

router = APIRouter(prefix="/incidents", tags=["Incidents"])

UPLOADS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "static", "uploads"))


def get_clip_url(snapshot_url: Optional[str]) -> Optional[str]:
    if not snapshot_url:
        return None
    stem = os.path.splitext(os.path.basename(snapshot_url))[0]
    for extension in (".webm", ".mp4"):
        if os.path.isfile(os.path.join(UPLOADS_DIR, f"{stem}{extension}")):
            return f"/static/uploads/{stem}{extension}"
    return None

@router.get("", response_model=List[IncidentResponse])
async def list_incidents(
    days: Optional[int] = Query(None, description="Filter hari terakhir, cth: 7, 30, 90"),
    location: Optional[str] = Query(None, description="Filter lokasi kamera, cth: ruang-tamu, all"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    user_cameras = await db.get_cameras(user_id=current_user["id"])
    camera_ids = [c["id"] for c in user_cameras]
    if not camera_ids:
        return []

    incidents = await db.get_incidents(days=days, location=location, camera_ids=camera_ids)
    return [
        IncidentResponse(
            id=inc["id"],
            camera_id=inc.get("camera_id"),
            camera_name=inc.get("camera_name", "Kamera CCTV"),
            location=inc.get("location", "Ruang Tamu"),
            person_name=inc.get("person_name", "Pak Hartono"),
            detected_at=inc.get("detected_at", ""),
            confidence=float(inc.get("confidence", 0.95)),
            snapshot_url=inc.get("snapshot_url"),
            clip_url=get_clip_url(inc.get("snapshot_url")),
            clip_duration=inc.get("clip_duration", "8 dtk"),
            response_time_seconds=float(inc.get("response_time_seconds", 2.4)),
            status=inc.get("status", "unresolved")
        )
        for inc in incidents
    ]

@router.post("", response_model=IncidentResponse, status_code=status.HTTP_201_CREATED)
async def create_incident(incident_in: IncidentCreate):
    incident_data = incident_in.model_dump()

    # Jika camera_id belum diset, coba kaitkan dengan kamera yang cocok berdasarkan nama/lokasi
    if not incident_data.get("camera_id") and incident_data.get("camera_name"):
        if db.is_connected:
            try:
                cam_res = db.client.table("cameras").select("id").ilike("name", f"%{incident_data['camera_name']}%").limit(1).execute()
                if cam_res.data:
                    incident_data["camera_id"] = cam_res.data[0]["id"]
            except Exception:
                pass

    created = await db.create_incident(incident_data)

    # Broadcast ke WebSocket client secara real-time
    await alert_manager.broadcast_fall_alert(created)

    return IncidentResponse(
        id=created["id"],
        camera_id=created.get("camera_id"),
        camera_name=created["camera_name"],
        location=created["location"],
        person_name=created["person_name"],
        detected_at=created["detected_at"],
        confidence=float(created["confidence"]),
        snapshot_url=created.get("snapshot_url"),
        clip_url=get_clip_url(created.get("snapshot_url")),
        clip_duration=created.get("clip_duration", "8 dtk"),
        response_time_seconds=float(created.get("response_time_seconds", 2.4)),
        status=created["status"]
    )
