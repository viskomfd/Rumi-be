import os

from fastapi import APIRouter, Query, status, Depends, Form, File, UploadFile, HTTPException
from typing import List, Optional, Dict, Any
import time
import uuid
from datetime import datetime, timezone
from app.schemas.incident import IncidentCreate, IncidentStatusUpdate, IncidentResponse
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


def _cleanup_incident_media(snapshot_url: Optional[str]) -> None:
    """Hapus file snapshot & klip video terkait di static/uploads (kecuali gambar demo bawaan)."""
    if not snapshot_url:
        return
    basename = os.path.basename(snapshot_url)
    if basename == "fall-snapshot.jpg":
        return
    stem = os.path.splitext(basename)[0]
    for filename in (basename, f"{stem}.webm", f"{stem}.mp4"):
        file_path = os.path.join(UPLOADS_DIR, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

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
async def create_incident(
    incident_in: IncidentCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    incident_data = incident_in.model_dump()

    # Jika camera_id belum diset, coba kaitkan dengan kamera yang cocok berdasarkan nama/lokasi
    if not incident_data.get("camera_id") and incident_data.get("camera_name"):
        if db.is_connected:
            try:
                cam_res = db.client.table("cameras").select("id").eq("user_id", current_user["id"]).ilike("name", f"%{incident_data['camera_name']}%").limit(1).execute()
                if cam_res.data:
                    incident_data["camera_id"] = cam_res.data[0]["id"]
            except Exception:
                pass

    created = await db.create_incident(incident_data)

    # Broadcast ke WebSocket client secara real-time
    await alert_manager.broadcast_fall_alert(created, user_id=current_user["id"])

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


@router.post("/incident-alerts", status_code=status.HTTP_200_OK)
async def handle_incident_alert(
    camera_id: str = Form(...),
    camera_name: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    confidence: float = Form(0.92),
    clip_duration_seconds: Optional[int] = Form(0),
    thumbnail: Optional[UploadFile] = File(None),
    clip: Optional[UploadFile] = File(None),
):
    """
    Menerima alert insiden dari FallDetectionWorker (realtime_fall_detection.py / manager.py).
    Menyimpan snapshot JPEG dan video clip MP4 (jika ada), mencatat ke Supabase,
    dan mengirim broadcast WebSocket real-time ke web dashboard.
    """
    ts = int(time.time() * 1000)
    stem = f"fall_{camera_id[:8]}_{ts}"
    snapshot_filename = f"{stem}.jpg"
    clip_filename = f"{stem}.mp4"

    os.makedirs(UPLOADS_DIR, exist_ok=True)

    snapshot_url = None
    if thumbnail:
        thumb_bytes = await thumbnail.read()
        thumb_path = os.path.join(UPLOADS_DIR, snapshot_filename)
        with open(thumb_path, "wb") as f:
            f.write(thumb_bytes)
        snapshot_url = f"/static/uploads/{snapshot_filename}"

    if clip:
        clip_bytes = await clip.read()
        clip_path = os.path.join(UPLOADS_DIR, clip_filename)
        with open(clip_path, "wb") as f:
            f.write(clip_bytes)

    # Cari data pemilik kamera dari database
    user_id = None
    camera = await db.get_camera_by_id(camera_id)
    if camera:
        user_id = camera.get("user_id")
        camera_name = camera_name or camera.get("name", "Kamera CCTV")
        location = location or camera.get("location", "Ruang Tamu")
    else:
        camera_name = camera_name or "Kamera CCTV"
        location = location or "Ruang Tamu"

    duration_str = f"{clip_duration_seconds} dtk" if (clip_duration_seconds and clip_duration_seconds > 0) else "8 dtk"

    incident_data = {
        "id": str(uuid.uuid4()),
        "camera_id": camera_id,
        "camera_name": camera_name,
        "location": location,
        "person_name": "Pak Hartono",
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "confidence": float(confidence),
        "snapshot_url": snapshot_url or "/static/uploads/fall-snapshot.jpg",
        "clip_duration": duration_str,
        "response_time_seconds": 2.1,
        "status": "unresolved"
    }

    created = await db.create_incident(incident_data)
    await alert_manager.broadcast_fall_alert(created, user_id=user_id)

    return {
        "status": "success",
        "message": "Incident alert berhasil diproses",
        "incident": created
    }


@router.post("/on-device-alert", response_model=IncidentResponse, status_code=status.HTTP_201_CREATED)
async def report_on_device_alert(
    camera_id: Optional[str] = Form(None),
    camera_name: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    person_name: Optional[str] = Form(None),
    confidence: float = Form(0.90),
    clip_duration: str = Form("0 dtk"),
    thumbnail: Optional[UploadFile] = File(None),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Mencatat insiden jatuh dari Kamera Device (On-Device WebAssembly/WebGPU) ke database Riwayat
    dan menyiarkan alert real-time via WebSocket.
    """
    user_id = current_user["id"]
    user_cameras = await db.get_cameras(user_id=user_id)

    resolved_camera = None
    if camera_id:
        resolved_camera = next((c for c in user_cameras if c.get("id") == camera_id), None)

    if not resolved_camera and camera_name:
        target_name = camera_name.strip().lower()
        resolved_camera = next(
            (c for c in user_cameras if (c.get("name") or "").strip().lower() == target_name),
            None,
        )

    if not resolved_camera:
        resolved_camera = next(
            (
                c
                for c in user_cameras
                if (c.get("protocol") or "").upper() == "DEVICE"
                or str(c.get("stream_url") or c.get("url") or "").lower().startswith("device://")
            ),
            None,
        )

    if not resolved_camera:
        resolved_camera = await db.create_camera({
            "user_id": user_id,
            "name": (camera_name or "Kamera Device").strip(),
            "location": (location or "Perangkat Lokal").strip(),
            "stream_url": "device://user",
            "protocol": "DEVICE",
            "status": "online",
            "is_active": False,
        })

    resolved_cam_id = resolved_camera["id"]
    resolved_cam_name = (camera_name or resolved_camera.get("name") or "Kamera Device").strip()
    resolved_location = (location or resolved_camera.get("location") or "Perangkat Lokal").strip()
    resolved_person = (person_name or current_user.get("elderly_name") or "Pak Hartono").strip()

    snapshot_url = None
    if thumbnail is not None:
        thumb_bytes = await thumbnail.read()
        if thumb_bytes:
            os.makedirs(UPLOADS_DIR, exist_ok=True)
            ts = int(time.time() * 1000)
            snapshot_filename = f"fall_{resolved_cam_id[:8]}_{ts}.jpg"
            with open(os.path.join(UPLOADS_DIR, snapshot_filename), "wb") as f:
                f.write(thumb_bytes)
            snapshot_url = f"/static/uploads/{snapshot_filename}"

    incident_data = {
        "id": str(uuid.uuid4()),
        "camera_id": resolved_cam_id,
        "camera_name": resolved_cam_name,
        "location": resolved_location,
        "person_name": resolved_person,
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "confidence": round(float(confidence), 2),
        "snapshot_url": snapshot_url,
        "clip_duration": clip_duration,
        "response_time_seconds": 1.2,
        "status": "unresolved",
    }

    created = await db.create_incident(incident_data)
    await alert_manager.broadcast_fall_alert(created, user_id=user_id)

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
        clip_duration=created.get("clip_duration", clip_duration),
        response_time_seconds=float(created.get("response_time_seconds", 1.2)),
        status=created["status"],
    )


@router.post("/{incident_id}/clip", status_code=status.HTTP_200_OK)
async def upload_incident_clip(
    incident_id: str,
    clip: UploadFile = File(...),
    clip_duration: str = Form("5 dtk"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Menyimpan file klip video (.webm/.mp4) untuk insiden yang sudah tercatat (mis. dari Kamera Device)
    dengan nama file yang selaras dengan snapshot_url agar langsung muncul di halaman Riwayat.
    """
    user_cameras = await db.get_cameras(user_id=current_user["id"])
    camera_ids = [c["id"] for c in user_cameras]
    incidents = await db.get_incidents(camera_ids=camera_ids)
    target = next((inc for inc in incidents if inc.get("id") == incident_id), None)

    snapshot_url = target.get("snapshot_url") if target else None
    if snapshot_url:
        stem = os.path.splitext(os.path.basename(snapshot_url))[0]
    else:
        stem = f"fall_{incident_id[:8]}"

    fname_lower = (clip.filename or "").lower()
    ctype_lower = (clip.content_type or "").lower()
    ext = ".mp4" if (fname_lower.endswith(".mp4") or "mp4" in ctype_lower) else ".webm"
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    clip_bytes = await clip.read()
    clip_path = os.path.join(UPLOADS_DIR, f"{stem}{ext}")
    with open(clip_path, "wb") as f:
        f.write(clip_bytes)

    if db.is_connected:
        try:
            db.client.table("incidents").update({"clip_duration": clip_duration}).eq("id", incident_id).execute()
        except Exception:
            pass
    else:
        for inc in db._mock_incidents:
            if inc.get("id") == incident_id:
                inc["clip_duration"] = clip_duration
                break

    return {
        "status": "success",
        "incident_id": incident_id,
        "clip_url": f"/static/uploads/{stem}{ext}",
        "clip_duration": clip_duration,
    }


@router.patch("/{incident_id}/status", response_model=IncidentResponse)
async def update_incident_status(
    incident_id: str,
    payload: IncidentStatusUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Memperbarui status insiden ('unresolved' | 'acknowledged' | 'resolved').
    Saat frontend mengubah status ke 'acknowledged' atau 'resolved' (mis. setelah user klik Lihat Detail),
    insiden ini tidak lagi muncul sebagai banner alert aktif di Dashboard.
    """
    user_cameras = await db.get_cameras(user_id=current_user["id"])
    camera_ids = {c["id"] for c in user_cameras}

    existing = await db.get_incident_by_id(incident_id)
    if not existing or (existing.get("camera_id") not in camera_ids):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Insiden tidak ditemukan atau bukan milik Anda",
        )

    updated = await db.update_incident_status(incident_id, payload.status)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal memperbarui status insiden",
        )

    return IncidentResponse(
        id=updated["id"],
        camera_id=updated.get("camera_id"),
        camera_name=updated.get("camera_name", "Kamera CCTV"),
        location=updated.get("location", "Ruang Tamu"),
        person_name=updated.get("person_name", "Pak Hartono"),
        detected_at=updated.get("detected_at", ""),
        confidence=float(updated.get("confidence", 0.95)),
        snapshot_url=updated.get("snapshot_url"),
        clip_url=get_clip_url(updated.get("snapshot_url")),
        clip_duration=updated.get("clip_duration", "8 dtk"),
        response_time_seconds=float(updated.get("response_time_seconds", 2.4)),
        status=updated.get("status", payload.status),
    )


@router.delete("/{incident_id}", status_code=status.HTTP_200_OK)
async def delete_incident(
    incident_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Menghapus 1 riwayat insiden milik user beserta file snapshot & klipnya."""
    user_cameras = await db.get_cameras(user_id=current_user["id"])
    camera_ids = {c["id"] for c in user_cameras}

    existing = await db.get_incident_by_id(incident_id)
    if not existing or (existing.get("camera_id") not in camera_ids):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Insiden tidak ditemukan atau bukan milik Anda",
        )

    deleted = await db.delete_incident(incident_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gagal menghapus insiden",
        )

    _cleanup_incident_media(existing.get("snapshot_url"))
    return {"message": "Riwayat insiden berhasil dihapus", "id": incident_id}


@router.delete("", status_code=status.HTTP_200_OK)
async def clear_all_incidents(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Menghapus seluruh riwayat insiden milik user beserta file snapshot & klipnya."""
    user_cameras = await db.get_cameras(user_id=current_user["id"])
    camera_ids = [c["id"] for c in user_cameras]
    if not camera_ids:
        return {"message": "Tidak ada riwayat untuk dihapus", "deleted_count": 0}

    existing_list = await db.get_incidents(camera_ids=camera_ids)
    deleted_count = await db.delete_incidents_by_camera_ids(camera_ids)

    for inc in existing_list:
        _cleanup_incident_media(inc.get("snapshot_url"))

    return {
        "message": "Semua riwayat insiden berhasil dihapus",
        "deleted_count": deleted_count,
    }

