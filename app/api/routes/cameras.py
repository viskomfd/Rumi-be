from fastapi import APIRouter, HTTPException, status, Depends, Request
from typing import List, Optional, Dict, Any
from app.schemas.camera import CameraCreate, CameraUpdate, CameraResponse
from app.core.supabase import db
from app.core.config import settings
from app.api.deps import get_current_user, get_optional_current_user
from app.services.camera_stream_manager import camera_stream_manager
from app.services.device_stream_relay import device_stream_relay

router = APIRouter(prefix="/cameras", tags=["Cameras"])

def detect_protocol(url: str) -> str:
    url_lower = url.lower()
    if url_lower.startswith("device://"):
        return "DEVICE"
    if url_lower.startswith("rtsp://"):
        return "RTSP"
    return "HTTP"

@router.get("", response_model=List[CameraResponse])
async def list_cameras(
    request: Request,
    is_active: Optional[bool] = None,
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user)
):
    # Jika dipanggil oleh background supervisor (manager.py) via X-Internal-Token
    internal_token = request.headers.get("X-Internal-Token")
    if internal_token and internal_token == settings.INTERNAL_SERVICE_TOKEN:
        cameras = await db.get_all_active_cameras()
        return [
            CameraResponse(
                id=c["id"],
                name=c["name"],
                location=c["location"],
                url=c.get("url") or c.get("stream_url", ""),
                protocol=c.get("protocol") or detect_protocol(c.get("url") or c.get("stream_url", "")),
                status=c.get("status", "online"),
                created_at=c.get("created_at")
            )
            for c in cameras
        ]

    # Akses normal oleh pengguna login
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Autentikasi gagal atau token tidak valid"
        )

    cameras = await db.get_cameras(user_id=current_user["id"])
    result: List[CameraResponse] = []
    for c in cameras:
        cam_url = c.get("url") or c.get("stream_url", "")
        cam_proto = c.get("protocol") or detect_protocol(cam_url)
        db_status = str(c.get("status", "online")).lower()
        live_status = (
            device_stream_relay.get_live_status(c["id"], db_status)
            if cam_proto == "DEVICE"
            else camera_stream_manager.get_live_status(c["id"], db_status)
        )
        result.append(
            CameraResponse(
                id=c["id"],
                name=c["name"],
                location=c["location"],
                url=cam_url,
                protocol=cam_proto,
                status="offline" if live_status == "offline" else "online",
                created_at=c.get("created_at"),
            )
        )
    return result


@router.post("", response_model=CameraResponse, status_code=status.HTTP_201_CREATED)
async def add_camera(
    camera_in: CameraCreate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    protocol = detect_protocol(camera_in.url)
    camera_data = {
        "user_id": current_user["id"],
        "name": camera_in.name,
        "location": camera_in.location,
        "stream_url": camera_in.url,
        "protocol": protocol,
        "status": "online",
        "is_active": protocol != "DEVICE"
    }
    created = await db.create_camera(camera_data)
    return CameraResponse(
        id=created["id"],
        name=created["name"],
        location=created["location"],
        url=created.get("stream_url", camera_in.url),
        protocol=created["protocol"],
        status=created["status"],
        created_at=created.get("created_at")
    )

import os
import cv2
import time
import uuid
import asyncio
from fastapi import Request
from fastapi.responses import StreamingResponse
from app.services.camera_stream_manager import camera_stream_manager

@router.patch("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: str,
    camera_in: CameraUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Edit nama, lokasi, dan/atau URL stream kamera milik user."""
    update_fields: Dict[str, Any] = {}
    if camera_in.name is not None:
        update_fields["name"] = camera_in.name.strip()
    if camera_in.location is not None:
        update_fields["location"] = camera_in.location.strip()
    if camera_in.url is not None:
        new_url = camera_in.url.strip()
        new_protocol = detect_protocol(new_url)
        update_fields["stream_url"] = new_url
        update_fields["protocol"] = new_protocol
        update_fields["is_active"] = new_protocol != "DEVICE"

    if not update_fields:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tidak ada field yang diubah (kirim minimal name, location, atau url)",
        )

    updated = await db.update_camera(camera_id, user_id=current_user["id"], update_data=update_fields)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Kamera tidak ditemukan atau bukan milik Anda",
        )

    # Jika URL/nama/lokasi berubah, hentikan worker lama agar koneksi baru dipakai saat stream dibuka lagi
    if "stream_url" in update_fields or "name" in update_fields or "location" in update_fields:
        camera_stream_manager.stop_camera(camera_id)

    cam_url = updated.get("stream_url") or updated.get("url", "")
    cam_proto = updated.get("protocol") or detect_protocol(cam_url)
    db_status = str(updated.get("status", "online")).lower()
    live_status = (
        db_status
        if cam_proto == "DEVICE"
        else camera_stream_manager.get_live_status(updated["id"], db_status)
    )
    return CameraResponse(
        id=updated["id"],
        name=updated["name"],
        location=updated["location"],
        url=cam_url,
        protocol=cam_proto,
        status="offline" if live_status == "offline" else "online",
        created_at=updated.get("created_at"),
    )


@router.patch("/{camera_id}/status")
async def update_camera_status(camera_id: str, payload: Dict[str, Any]):
    """Update status kamera (online / offline) dari UI (Aktifkan/Nonaktifkan) atau worker deteksi jatuh."""
    status_str = str(payload.get("status", "online")).lower()
    if status_str not in ("online", "offline"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Status harus 'online' atau 'offline'")

    if status_str == "offline":
        camera_stream_manager.stop_camera(camera_id)
        await device_stream_relay.stop_camera(camera_id)

    await db.update_camera_status(camera_id, status_str)
    camera = await db.get_camera_by_id(camera_id)
    if camera:
        cam_url = camera.get("stream_url") or camera.get("url", "")
        cam_proto = camera.get("protocol") or detect_protocol(cam_url)
        return {
            "id": camera["id"],
            "camera_id": camera["id"],
            "name": camera.get("name", ""),
            "location": camera.get("location", ""),
            "url": cam_url,
            "protocol": cam_proto,
            "status": status_str,
            "camera_status": status_str,
            "created_at": camera.get("created_at"),
        }

    return {"id": camera_id, "camera_id": camera_id, "status": status_str, "camera_status": status_str}

@router.delete("/{camera_id}", status_code=status.HTTP_200_OK)
async def delete_camera(
    camera_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    camera_stream_manager.stop_camera(camera_id)
    await device_stream_relay.stop_camera(camera_id)
    success = await db.delete_camera(camera_id, user_id=current_user["id"])
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Kamera tidak ditemukan atau bukan milik Anda")
    return {"message": "Kamera berhasil dihapus", "id": camera_id}

@router.get("/{camera_id}/stream")
async def stream_camera(camera_id: str, request: Request):
    """
    Gateway proxy RTSP/HTTP ke MJPEG terpusat:
    Menggunakan CameraStreamManager (1 background worker per kamera)
    sehingga tidak pernah terjadi timeout/freeze saat berganti tab atau halaman.
    """
    camera = await db.get_camera_by_id(camera_id)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Kamera tidak ditemukan")

    if str(camera.get("status", "online")).lower() == "offline":
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Kamera sedang dinonaktifkan")

    stream_url = camera.get("stream_url") or camera.get("url")
    if not stream_url:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="URL stream kamera kosong")

    cam_proto = (camera.get("protocol") or detect_protocol(stream_url)).upper()
    if cam_proto == "DEVICE":
        if not device_stream_relay.is_online(camera_id):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Kamera perangkat sedang offline atau belum menyiarkan.",
            )

        async def device_frame_generator():
            while True:
                if await request.is_disconnected() or not device_stream_relay.is_online(camera_id):
                    break
                jpeg_bytes = device_stream_relay.get_latest_jpeg(camera_id)
                if jpeg_bytes:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                    )
                await asyncio.sleep(0.065)  # ~15 FPS

        return StreamingResponse(
            device_frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    worker = camera_stream_manager.get_or_create_worker(
        camera_id=camera_id,
        user_id=camera.get("user_id", ""),
        stream_url=stream_url,
        name=camera.get("name", "Kamera CCTV"),
        location=camera.get("location", "Ruang Tamu")
    )
    worker.add_client()

    async def frame_generator():
        try:
            while True:
                # Jika browser menutup tab / navigasi ke route lain, atau kamera dinonaktifkan, akhiri response
                if await request.is_disconnected() or not worker.running:
                    break

                jpeg_bytes = camera_stream_manager.get_latest_jpeg(camera_id)
                if jpeg_bytes:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                    )
                await asyncio.sleep(0.04)  # ~25 FPS
        finally:
            worker.remove_client()

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
