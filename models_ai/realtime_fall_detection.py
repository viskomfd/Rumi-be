"""
Real-time Fall Detection menggunakan YOLOv8-Pose
=================================================
Modul integrasi deteksi jatuh untuk sistem Rumi (FastAPI Backend).
Mendukung inferensi per-frame API, live camera streaming, dan standalone execution.

Requirement:
    pip install ultralytics opencv-python requests
"""

import os
import sys
import time
import math
import base64
import logging
from collections import deque
from typing import Dict, Any, Optional, Tuple, List

import cv2
import numpy as np

logger = logging.getLogger("rumi.model")

# =========================
# KONFIGURASI THRESHOLD
# =========================
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov8n-pose.pt")
CONF_THRESHOLD = 0.35                # confidence deteksi orang (diturunkan agar posisi rebah di lantai tetap terdeteksi)

# Threshold hasil kalibrasi dari eksperimen deteksi jatuh
ASPECT_RATIO_THRESHOLD = 0.55        # width/height bbox > ini -> indikasi rebah
TORSO_ANGLE_THRESHOLD = 40.0         # derajat dari vertikal -> indikasi rebah
VERTICAL_VELOCITY_THRESHOLD = 20.0   # px/frame turun cepat -> indikasi jatuh
INACTIVITY_SECONDS = 1.5             # diam di posisi rebah >= 1.5 detik -> fall confirmed (responsif untuk tes)

POSTURE_WINDOW = 5                   # jumlah frame terakhir yang dicek untuk voting
POSTURE_VOTE_RATIO = 0.5             # minimal proporsi frame TRUE dalam window agar dianggap stabil

ALERT_API_URL = "http://127.0.0.1:8000/api/incidents"

# Index keypoint COCO (dipakai YOLOv8-Pose): 0=hidung, 5=bahu kiri, 6=bahu kanan,
# 11=pinggul kiri, 12=pinggul kanan, 15=pergelangan kaki kiri, 16=pergelangan kaki kanan
NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 0, 5, 6, 11, 12


def calc_torso_angle(shoulder_mid, hip_mid) -> float:
    """Sudut kemiringan torso terhadap sumbu vertikal (0 derajat = berdiri tegak)."""
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    angle_from_vertical = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))
    return float(angle_from_vertical)


def calc_aspect_ratio(box) -> float:
    """Menghitung perbandingan lebar terhadap tinggi bounding box."""
    x1, y1, x2, y2 = box
    w, h = (x2 - x1), (y2 - y1)
    return float(w / (h + 1e-6))


def midpoint(p1, p2) -> Tuple[float, float]:
    """Menghitung titik tengah antara dua koordinat."""
    return ((float(p1[0]) + float(p2[0])) / 2.0, (float(p1[1]) + float(p2[1])) / 2.0)


class PersonTracker:
    """Menyimpan histori posisi tiap orang (per track id) untuk hitung
    vertical velocity dan durasi tidak bergerak (inactivity)."""

    def __init__(self, history_len: int = 10):
        self.hip_y_history = deque(maxlen=history_len)
        self.posture_flags = deque(maxlen=POSTURE_WINDOW)
        self.fall_suspected_since: Optional[float] = None
        self.fall_start_pos: Optional[Tuple[float, float]] = None
        self.last_hip_pos: Optional[Tuple[float, float]] = None
        self.alert_sent: bool = False

    def update_posture_flag(self, is_fall_posture: bool) -> bool:
        """Catat hasil deteksi per-frame, lalu putuskan status STABIL berdasarkan voting."""
        self.posture_flags.append(is_fall_posture)
        if len(self.posture_flags) < min(3, self.posture_flags.maxlen):
            return is_fall_posture
        true_ratio = sum(self.posture_flags) / len(self.posture_flags)
        return true_ratio >= POSTURE_VOTE_RATIO

    def update_hip(self, hip_mid: Tuple[float, float]):
        self.hip_y_history.append(hip_mid[1])
        self.last_hip_pos = hip_mid

    def vertical_velocity(self) -> float:
        if len(self.hip_y_history) < 2:
            return 0.0
        return float(self.hip_y_history[-1] - self.hip_y_history[0])

    def is_inactive(self, current_hip: Tuple[float, float], movement_threshold: float = 40.0) -> bool:
        if self.fall_start_pos is None:
            self.fall_start_pos = current_hip
            return True
        dist = math.dist(current_hip, self.fall_start_pos)
        return dist < movement_threshold


def send_alert(
    frame: np.ndarray,
    track_id: int,
    confidence: float,
    camera_name: str = "Ruang Tamu · Cam 01",
    location: str = "Ruang Tamu",
    camera_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Simpan snapshot dan kirim data insiden ke backend FastAPI."""
    import requests

    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
    snapshot_filename = f"fall_{track_id}_{int(time.time())}.jpg"

    # Simpan snapshot ke folder static backend
    backend_static = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static", "uploads"))
    os.makedirs(backend_static, exist_ok=True)
    snapshot_path = os.path.join(backend_static, snapshot_filename)
    cv2.imwrite(snapshot_path, frame)

    payload = {
        "camera_id": camera_id,
        "camera_name": camera_name,
        "location": location,
        "person_name": "Pak Hartono",
        "confidence": round(float(confidence), 2),
        "snapshot_url": f"/static/uploads/{snapshot_filename}",
        "clip_duration": "8 dtk",
        "status": "unresolved"
    }

    try:
        res = requests.post(ALERT_API_URL, json=payload, timeout=3)
        if res.status_code in [200, 201]:
            logger.info(f"[ALERT SENT] Kejadian jatuh berhasil dikirim ke backend: {payload}")
            return res.json()
        else:
            logger.warning(f"[ALERT WARNING] Backend merespons status {res.status_code}: {res.text}")
    except Exception as e:
        logger.warning(f"[ALERT WARNING] Gagal mengirim alert ke {ALERT_API_URL}: {e}")

    return payload


class FallDetector:
    """Class wrapper untuk inferensi YOLOv8-pose dan algoritma deteksi jatuh."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.model = None
        self.is_loaded = False
        self.trackers: Dict[int, PersonTracker] = {}
        self._load_model()

    def _load_model(self):
        """Memuat model YOLOv8 Pose."""
        if not os.path.exists(self.model_path):
            logger.warning(
                f"[FallDetector] File model '{self.model_path}' belum ditemukan. "
                "Berjalan dalam mode Simulasi/Adapter."
            )
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            logger.info(f"[FallDetector] Model YOLOv8-pose berhasil dimuat dari: {self.model_path}")
        except Exception as e:
            logger.error(f"[FallDetector] Gagal memuat YOLOv8 model: {e}")

    def reset_trackers(self):
        self.trackers.clear()

    def process_cv2_frame(
        self,
        frame: np.ndarray,
        on_fall_callback=None,
        camera_name: str = "Ruang Tamu · Cam 01",
        location: str = "Ruang Tamu",
        camera_id: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Memproses satu frame OpenCV numpy array untuk deteksi jatuh.
        Mengembalikan (annotated_frame, telemetry_dict).
        """
        start_time = time.time()
        telemetry: Dict[str, Any] = {
            "person_detected": False,
            "fall_detected": False,
            "status": "NORMAL",
            "confidence": 0.0,
            "aspect_ratio": 0.0,
            "torso_angle": 0.0,
            "v_velocity": 0.0,
            "latency_ms": 0.0,
            "track_id": None,
            "bounding_box": None,
            "source": "yolov8n-pose"
        }

        if not self.is_loaded or self.model is None:
            telemetry["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            telemetry["source"] = "mock_adapter"
            return frame, telemetry

        # Jalankan tracking / inferensi pose
        try:
            results = self.model.track(frame, persist=True, conf=CONF_THRESHOLD, verbose=False)
        except Exception:
            results = self.model(frame, conf=CONF_THRESHOLD, verbose=False)
        annotated_frame = results[0].plot()

        if results[0].boxes is not None and len(results[0].boxes) > 0 and results[0].keypoints is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = (
                results[0].boxes.id.int().cpu().tolist()
                if results[0].boxes.id is not None
                else [1] * len(boxes)
            )
            keypoints = results[0].keypoints.xy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else [0.9] * len(boxes)

            # Single-person tracker persistence:
            # Jika hanya ada 1 orang di frame dan track_id berganti saat jatuh, wariskan status tracker
            if len(boxes) == 1 and track_ids[0] not in self.trackers and len(self.trackers) > 0:
                old_tracker = list(self.trackers.values())[-1]
                if old_tracker.fall_suspected_since is not None and not old_tracker.alert_sent:
                    self.trackers[track_ids[0]] = old_tracker

            for box, track_id, kpts, conf in zip(boxes, track_ids, keypoints, confidences):
                if track_id not in self.trackers:
                    self.trackers[track_id] = PersonTracker()
                tracker = self.trackers[track_id]

                shoulder_mid = midpoint(kpts[L_SHOULDER], kpts[R_SHOULDER])
                hip_mid = midpoint(kpts[L_HIP], kpts[R_HIP])

                aspect_ratio = calc_aspect_ratio(box)
                torso_angle = calc_torso_angle(shoulder_mid, hip_mid)

                tracker.update_hip(hip_mid)
                v_velocity = tracker.vertical_velocity()

                # Heuristic & Motion Analysis:
                # Seseorang HANYA terindikasi jatuh jika torso miring mendekati mendatar (> 45 derajat)
                # DAN (bounding box melebar mendatar > 0.85 ATAU bergerak turun dengan cepat).
                # Ini mencegah false-positive saat berdiri tegak di depan kamera.
                is_torso_inclined = torso_angle > 45.0
                is_box_horizontal = aspect_ratio > 0.85
                is_dropping_fast = v_velocity > 20.0

                raw_posture_flag = is_torso_inclined and (is_box_horizontal or is_dropping_fast)

                posture_indicates_fall = tracker.update_posture_flag(raw_posture_flag)

                status_text = "Normal"
                fall_confirmed = False

                if posture_indicates_fall:
                    status_text = "Suspected Fall"
                    if tracker.fall_suspected_since is None:
                        tracker.fall_suspected_since = time.time()
                        tracker.fall_start_pos = hip_mid

                    duration = time.time() - tracker.fall_suspected_since
                    inactive = tracker.is_inactive(hip_mid)

                    if duration >= INACTIVITY_SECONDS and inactive and not tracker.alert_sent:
                        status_text = "FALL DETECTED"
                        fall_confirmed = True
                        tracker.alert_sent = True

                        if on_fall_callback:
                            try:
                                on_fall_callback(frame, track_id, float(conf))
                            except Exception as e:
                                logger.error(f"Error in on_fall_callback: {e}")
                        else:
                            send_alert(frame, track_id, float(conf), camera_name, location, camera_id)
                else:
                    tracker.fall_suspected_since = None
                    tracker.fall_start_pos = None
                    tracker.alert_sent = False

                # Gambar status teks ke frame
                x1, y1 = int(box[0]), int(box[1])
                color = (0, 0, 255) if "FALL" in status_text else ((0, 165, 255) if "Suspected" in status_text else (0, 255, 0))
                cv2.putText(
                    annotated_frame, f"ID {track_id}: {status_text}",
                    (x1, max(y1 - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                )
                if "FALL DETECTED" in status_text:
                    # Gambar border merah peringatan di sekeliling frame video
                    h_frame, w_frame = annotated_frame.shape[:2]
                    cv2.rectangle(annotated_frame, (0, 0), (w_frame - 1, h_frame - 1), (0, 0, 255), 6)
                    cv2.putText(
                        annotated_frame, "PERINGATAN: JATUH TERDETEKSI!",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
                    )

                # Set telemetry
                telemetry.update({
                    "person_detected": True,
                    "fall_detected": bool(fall_confirmed),
                    "status": status_text,
                    "confidence": round(float(conf), 2),
                    "aspect_ratio": round(aspect_ratio, 2),
                    "torso_angle": round(torso_angle, 2),
                    "v_velocity": round(v_velocity, 2),
                    "track_id": track_id,
                    "bounding_box": [float(b) for b in box],
                })

        telemetry["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        return annotated_frame, telemetry

    def process_frame(self, frame_bytes: Optional[bytes] = None, return_annotated: bool = False) -> Dict[str, Any]:
        """
        Memproses 1 frame gambar dalam bentuk bytes (untuk endpoint HTTP multipart / upload).
        """
        start_time = time.time()
        if frame_bytes is None:
            return self.simulate_fall()

        # Decode bytes ke OpenCV numpy array
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Gagal mendecode frame gambar.")

        annotated_frame, telemetry = self.process_cv2_frame(frame)

        if return_annotated:
            _, buffer = cv2.imencode(".jpg", annotated_frame)
            telemetry["annotated_image_base64"] = base64.b64encode(buffer).decode("utf-8")

        telemetry["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        return telemetry

    def simulate_fall(self) -> Dict[str, Any]:
        """Method helper untuk menguji trigger insiden jatuh langsung dari backend."""
        return {
            "fall_detected": True,
            "confidence": 0.97,
            "person_detected": True,
            "status": "FALL DETECTED",
            "aspect_ratio": 1.45,
            "torso_angle": 68.5,
            "v_velocity": 34.0,
            "bounding_box": [80.0, 250.0, 460.0, 390.0],
            "latency_ms": 28.4,
            "source": "simulation_trigger"
        }


# Instance global singleton
detector = FallDetector()


def main():
    """Jalankan deteksi langsung secara mandiri di webcam/CCTV."""
    video_source = 0
    if len(sys.argv) > 1:
        video_source = sys.argv[1]

    print("=" * 60)
    print(f" Memulai Deteksi Jatuh Real-time pada sumber: {video_source}")
    print(" Tekan 'q' di jendela video untuk keluar.")
    print("=" * 60)

    cap = cv2.VideoCapture(video_source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame, mencoba menyambung ulang...")
            time.sleep(1)
            continue

        annotated_frame, telemetry = detector.process_cv2_frame(frame)

        cv2.imshow("Rumi Fall Detection (YOLOv8-Pose) - Press 'q' to quit", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
