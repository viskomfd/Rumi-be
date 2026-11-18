import os
import uuid
from typing import Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Query, HTTPException, Depends

from models_ai.realtime_fall_detection import (
    detector,
    ASPECT_RATIO_THRESHOLD,
    TORSO_ANGLE_THRESHOLD,
    VERTICAL_VELOCITY_THRESHOLD,
    FLOOR_COLLAPSE_THRESHOLD,
    INACTIVITY_SECONDS,
    CONF_THRESHOLD,
)
from app.core.supabase import db
from app.services.alert_manager import alert_manager
from app.api.deps import get_current_user

router = APIRouter(prefix="/model", tags=["AI Model & Vision"])


@router.get("/status")
async def get_model_status():
    """Cek status kesiapan file model yolo11_rumi.pt."""
    return {
        "model_file": detector.model_path,
        "is_model_loaded": detector.is_loaded,
        "mode": "YOLO11-Pose (Active - 4D Grid Search)" if detector.is_loaded else "Mock / Adapter Mode (Model file missing)",
        "thresholds": {
            "aspect_ratio_threshold": ASPECT_RATIO_THRESHOLD,
            "torso_angle_threshold": TORSO_ANGLE_THRESHOLD,
            "vertical_velocity_threshold": VERTICAL_VELOCITY_THRESHOLD,
            "floor_collapse_threshold": FLOOR_COLLAPSE_THRESHOLD,
            "inactivity_seconds": INACTIVITY_SECONDS,
            "confidence_threshold": CONF_THRESHOLD
        },
        "message": (
            "Model YOLO11-pose (yolo11_rumi.pt) kalibrasi 4D Grid Search aktif dan siap mendeteksi jatuh!"
            if detector.is_loaded
            else f"Letakkan file 'yolo11_rumi.pt' di folder 'models_ai/' untuk mengaktifkan model asli."
        )
    }



@router.post("/detect-frame")
async def detect_frame(
    file: UploadFile = File(...),
    return_annotated: bool = Query(False, description="Sertakan gambar beranotasi dalam format base64"),
    camera_name: str = Query("Ruang Tamu · Cam 01", description="Nama kamera sumber"),
    location: str = Query("Ruang Tamu", description="Lokasi kamera"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Menerima 1 frame gambar untuk dianalisis oleh model YOLOv8-Pose.
    Jika terdeteksi jatuh, otomatis menyimpan insiden ke Supabase dan memicu alert real-time ke WebSocket.
    """
    try:
        content = await file.read()
        result = detector.process_frame(content, return_annotated=return_annotated)

        # Jika terdeteksi jatuh, simpan ke database dan broadcast ke WebSocket
        if result.get("fall_detected"):
            incident_id = str(uuid.uuid4())
            snapshot_filename = f"fall_{incident_id[:8]}.jpg"
            static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "static", "uploads"))
            os.makedirs(static_dir, exist_ok=True)
            snapshot_path = os.path.join(static_dir, snapshot_filename)

            with open(snapshot_path, "wb") as f:
                f.write(content)

            incident_data = {
                "id": incident_id,
                "camera_name": camera_name,
                "location": location,
                "person_name": "Pak Hartono",
                "confidence": result.get("confidence", 0.95),
                "snapshot_url": f"/static/uploads/{snapshot_filename}",
                "clip_duration": "8 dtk",
                "response_time_seconds": round(result.get("latency_ms", 24.0) / 1000.0, 2),
                "status": "unresolved"
            }
            saved = await db.create_incident(incident_data)
            await alert_manager.broadcast_fall_alert(saved, user_id=current_user["id"])
            result["incident_recorded"] = saved

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses frame: {str(e)}")


@router.post("/simulate-fall")
async def trigger_simulated_fall(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Endpoint pengujian: Menstimulasi kejadian jatuh buatan.
    Ini akan membuat record baru di database Supabase dan mengirim Fall Alert secara real-time ke WebSocket.
    """
    result = detector.simulate_fall()

    incident_data = {
        "id": str(uuid.uuid4()),
        "camera_name": "Ruang Tamu · Cam 01",
        "location": "Ruang Tamu",
        "person_name": "Pak Hartono",
        "confidence": result["confidence"],
        "snapshot_url": "/static/uploads/fall-snapshot.jpg",
        "clip_duration": "8 dtk",
        "response_time_seconds": 2.1,
        "status": "unresolved"
    }

    saved = await db.create_incident(incident_data)
    await alert_manager.broadcast_fall_alert(saved, user_id=current_user["id"])

    return {
        "message": "Simulasi kejadian jatuh berhasil dipicu dan di-broadcast!",
        "detection": result,
        "incident": saved
    }


@router.post("/reset-trackers")
async def reset_trackers():
    """Reset memori tracker orang yang sedang dipantau."""
    detector.reset_trackers()
    return {"message": "Semua tracker pose telah di-reset"}
