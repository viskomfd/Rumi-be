from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Optional, Dict, Any
from app.schemas.camera import CameraCreate, CameraResponse
from app.core.supabase import db
from app.api.deps import get_current_user, get_optional_current_user

router = APIRouter(prefix="/cameras", tags=["Cameras"])

def detect_protocol(url: str) -> str:
    url_lower = url.lower()
    if url_lower.startswith("rtsp://"):
        return "RTSP"
    return "HTTP"

@router.get("", response_model=List[CameraResponse])
async def list_cameras(current_user: Dict[str, Any] = Depends(get_current_user)):
    cameras = await db.get_cameras(user_id=current_user["id"])
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
        "is_active": True
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

@router.delete("/{camera_id}", status_code=status.HTTP_200_OK)
async def delete_camera(
    camera_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    camera_stream_manager.stop_camera(camera_id)
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

    stream_url = camera.get("stream_url") or camera.get("url")
    if not stream_url:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="URL stream kamera kosong")

    worker = camera_stream_manager.get_or_create_worker(
        camera_id=camera_id,
        stream_url=stream_url,
        name=camera.get("name", "Kamera CCTV"),
        location=camera.get("location", "Ruang Tamu")
    )
    worker.add_client()

    async def frame_generator():
        try:
            while True:
                # Jika browser menutup tab / navigasi ke route lain, akhiri response
                if await request.is_disconnected():
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
