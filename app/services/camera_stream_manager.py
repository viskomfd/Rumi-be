import os
import cv2
import time
import uuid
import logging
import threading
import asyncio
from typing import Dict, Optional, Tuple, Any

import numpy as np
from app.core.supabase import db
from app.services.alert_manager import alert_manager
from models_ai.realtime_fall_detection import detector

logger = logging.getLogger("rumi.camera_stream")

# Konfigurasi OpenCV FFMPEG:
# - rtsp_transport;tcp  : Menggunakan TCP agar tidak ada packet loss
# - timeout;5000000     : Socket timeout 5 detik (5.000.000 us), bukan 30 detik default
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp;timeout;5000000"


class CameraWorker:
    """
    Worker kamera CCTV dengan arsitektur Producer-Consumer terpisah:
    - Thread 1 (Capture): Khusus membaca cap.read() secepat mungkin tanpa sleep/delay.
      Ini mencegah internal buffer OpenCV penuh dan mencegah stream timeout / freeze.
    - Thread 2 (Process): Khusus encoding JPEG dan inferensi AI pada frame terbaru.
    """

    def __init__(self, camera_id: str, stream_url: str, name: str, location: str):
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.name = name
        self.location = location

        self.running = False
        self.capture_thread: Optional[threading.Thread] = None
        self.process_thread: Optional[threading.Thread] = None

        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_jpeg: Optional[bytes] = None

        self.frame_lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.last_frame_time: float = 0
        self.is_online = False
        self.client_count = 0

    def start(self):
        if self.running:
            return
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, name=f"Cap-{self.name}", daemon=True)
        self.process_thread = threading.Thread(target=self._process_loop, name=f"Proc-{self.name}", daemon=True)
        self.capture_thread.start()
        self.process_thread.start()
        logger.info(f"[CameraWorker] Worker dimulai untuk kamera '{self.name}' ({self.camera_id})")

    def stop(self):
        self.running = False
        self.new_frame_event.set()
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.5)
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=1.5)
        logger.info(f"[CameraWorker] Worker dihentikan untuk kamera '{self.name}'")

    def add_client(self):
        self.client_count += 1

    def remove_client(self):
        self.client_count = max(0, self.client_count - 1)

    def _capture_loop(self):
        """
        Thread Producer: Hanya bertugas membaca cap.read() secepat mungkin.
        TIDAK ADA pemrosesan berat, TIDAK ADA sleep di sini!
        """
        reconnect_delay = 1.0

        while self.running:
            cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                self.is_online = False
                cap.release()
                time.sleep(reconnect_delay)
                reconnect_delay = min(5.0, reconnect_delay * 1.5)
                continue

            self.is_online = True
            reconnect_delay = 1.0
            consecutive_empty = 0

            while self.running and cap.isOpened():
                success, frame = cap.read()
                if not success or frame is None or frame.size == 0:
                    consecutive_empty += 1
                    time.sleep(0.01)
                    if consecutive_empty > 30:
                        logger.warning(f"[CameraWorker] Tidak ada frame dari '{self.name}', reconnect...")
                        break
                    continue

                consecutive_empty = 0
                with self.frame_lock:
                    self.latest_raw_frame = frame
                    self.last_frame_time = time.time()
                self.new_frame_event.set()

            cap.release()
            self.is_online = False
            time.sleep(1.0)

    def _dispatch_fall_alert_async(self, fall_frame: np.ndarray, track_id: int, conf: float):
        """
        Menyimpan snapshot dan mengirim notifikasi Supabase/WebSocket
        di thread latar belakang terpisah agar TIDAK PERNAH menghentikan stream kamera.
        """
        def _bg_task():
            try:
                snapshot_fn = f"fall_{self.camera_id[:8]}_{int(time.time())}.jpg"
                static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "static", "uploads"))
                os.makedirs(static_dir, exist_ok=True)
                cv2.imwrite(os.path.join(static_dir, snapshot_fn), fall_frame)

                incident_payload = {
                    "id": str(uuid.uuid4()),
                    "camera_id": self.camera_id,
                    "camera_name": self.name,
                    "location": self.location,
                    "person_name": "Pak Hartono",
                    "confidence": round(float(conf), 2),
                    "snapshot_url": f"/static/uploads/{snapshot_fn}",
                    "clip_duration": "8 dtk",
                    "response_time_seconds": 2.1,
                    "status": "unresolved"
                }

                # Simpan ke Supabase & siarkan via WebSocket
                asyncio.run(db.create_incident(incident_payload))
                asyncio.run(alert_manager.broadcast_fall_alert(incident_payload))
                logger.info(f"[CameraWorker] Alert jatuh berhasil dikirim: {self.name}")
            except Exception as e:
                logger.error(f"[CameraWorker] Gagal dispatch alert async: {e}")

        threading.Thread(target=_bg_task, daemon=True).start()

    def _process_loop(self):
        """
        Thread Consumer: Mengambil frame terbaru, menjalankan inferensi YOLO, dan meng-encode JPEG.
        """
        frame_idx = 0
        last_process_time = 0

        while self.running:
            # Tunggu frame baru dari capture thread (timeout 100ms)
            if not self.new_frame_event.wait(timeout=0.1):
                continue
            self.new_frame_event.clear()

            # Pacing ~25-30 FPS
            now = time.time()
            if now - last_process_time < 0.033:
                time.sleep(0.005)
                continue
            last_process_time = now

            with self.frame_lock:
                if self.latest_raw_frame is None:
                    continue
                current_frame = self.latest_raw_frame.copy()

            frame_idx += 1

            # Jalankan inferensi pose setiap 2 frame agar CPU tetap ringan
            if frame_idx % 2 == 0 or self.latest_annotated_frame is None:
                try:
                    annotated, _ = detector.process_cv2_frame(
                        current_frame,
                        on_fall_callback=self._dispatch_fall_alert_async,
                        camera_name=self.name,
                        location=self.location,
                        camera_id=self.camera_id
                    )
                except Exception as e:
                    logger.warning(f"[CameraWorker] Error inferensi pose: {e}")
                    annotated = current_frame
            else:
                annotated = self.latest_annotated_frame if self.latest_annotated_frame is not None else current_frame

            # Encode frame ke format JPEG
            ret, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 68])
            if ret:
                jpeg_bytes = buf.tobytes()
                with self.frame_lock:
                    self.latest_annotated_frame = annotated
                    self.latest_jpeg = jpeg_bytes


class CameraStreamManager:
    """Manager koleksi worker kamera terpusat (singleton)."""

    def __init__(self):
        self.workers: Dict[str, CameraWorker] = {}
        self.lock = threading.Lock()

    def get_or_create_worker(self, camera_id: str, stream_url: str, name: str, location: str) -> CameraWorker:
        with self.lock:
            if camera_id in self.workers:
                worker = self.workers[camera_id]
                # Jika stream_url berubah, perbarui worker
                if worker.stream_url != stream_url:
                    worker.stop()
                    worker = CameraWorker(camera_id, stream_url, name, location)
                    self.workers[camera_id] = worker
                    worker.start()
                elif not worker.running:
                    worker.start()
                return worker

            worker = CameraWorker(camera_id, stream_url, name, location)
            self.workers[camera_id] = worker
            worker.start()
            return worker

    def get_latest_jpeg(self, camera_id: str) -> Optional[bytes]:
        worker = self.workers.get(camera_id)
        if not worker:
            return None
        with worker.frame_lock:
            return worker.latest_jpeg

    def is_camera_online(self, camera_id: str) -> bool:
        worker = self.workers.get(camera_id)
        if not worker:
            return False
        return worker.is_online and (time.time() - worker.last_frame_time < 5.0)

    def stop_camera(self, camera_id: str):
        with self.lock:
            worker = self.workers.pop(camera_id, None)
            if worker:
                worker.stop()

    def stop_all(self):
        with self.lock:
            for worker in self.workers.values():
                worker.stop()
            self.workers.clear()


camera_stream_manager = CameraStreamManager()
