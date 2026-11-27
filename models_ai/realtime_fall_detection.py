import time
import math
from collections import deque
import os
import re
import threading
import subprocess
import base64
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np
import requests
from ultralytics import YOLO
import torch
import imageio_ffmpeg

# =============================================================================
# KONFIGURASI GLOBAL & THRESHOLD HEURISTIK JATUH
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO_SOURCE = os.path.join(BASE_DIR, "test_fall.mp4")

# =============================================================================
# KONFIGURASI PARAMETER DETEKSI JATUH (RUMI v2 — KALIBRASI 4D GRID SEARCH + ADAPTIVE TEMPORAL)
# =============================================================================
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.40"))
ASPECT_RATIO_THRESHOLD      = 0.85   # Lebar vs Tinggi kotak pembatas
TORSO_ANGLE_THRESHOLD       = 45.0   # Kemiringan sudut torso (derajat)
VERTICAL_VELOCITY_THRESHOLD = 13.0   # Kecepatan turun instan (px/step @480p)
FLOOR_COLLAPSE_THRESHOLD    = 0.42   # Parameter ke-4: Rasio batas atas kotak (y1 / frame_h)

SLOW_FALL_DROP_THRESHOLD    = 45.0   # Penurunan vertikal kumulatif (px @480p)
SHORT_DROP_GUARD_THRESHOLD  = 25.0   # Guard penurunan cepat (8-step) untuk membedakan duduk lambat
INACTIVITY_SECONDS          = float(os.getenv("INACTIVITY_SECONDS", "1.2"))
INACTIVITY_FAST_SECONDS     = float(os.getenv("INACTIVITY_FAST_SECONDS", "0.45"))
UPRIGHT_RECOVERY_FRAMES     = int(os.getenv("UPRIGHT_RECOVERY_FRAMES", "2"))
POSTURE_WINDOW              = 5      # Ukuran sliding window voting postur
POSTURE_VOTE_RATIO          = 0.4    # Minimal 2 dari 5 frame terindikasi jatuh
SAME_SPOT_COOLDOWN          = 4.0    # Mencegah spam alert pada titik koordinat yang sama

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
ALERT_API_URL = f"{API_BASE_URL}/api/incident-alerts"
CAMERA_STATUS_URL = f"{API_BASE_URL}/api/cameras"

DEBUG_GUI = os.getenv("DEBUG_GUI", "true").lower() in ("true", "1")
INFERENCE_FPS = int(os.getenv("INFERENCE_TARGET_FPS", "10"))
CLIP_BUFFER_SECONDS = 5  # 5 detik sebelum dan 5 detik sesudah jatuh

# Index keypoint COCO Pose
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

torch.set_num_threads(4)


# =============================================================================
# HELPER MATEMATIKA & POSE
# =============================================================================

def mask_url(url: str) -> str:
    """Masks password in rtsp://user:password@host -> rtsp://user:***@host"""
    return re.sub(r'(://[^:]+:)([^@]+)(@)', r'\1***\3', str(url))


def calc_torso_angle(shoulder_mid, hip_mid):
    """Menghitung sudut kemiringan torso terhadap sumbu vertikal (0 = tegak, 90 = mendatar)."""
    dx = hip_mid[0] - shoulder_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def calc_shoulder_tilt(l_sh, r_sh):
    """Menghitung kemiringan garis bahu terhadap sumbu horizontal."""
    dx = abs(r_sh[0] - l_sh[0])
    dy = abs(r_sh[1] - l_sh[1])
    return math.degrees(math.atan2(dy, dx + 1e-6))


def calc_aspect_ratio(box):
    """Menghitung rasio lebar dibanding tinggi bounding box."""
    x1, y1, x2, y2 = box
    w, h = (x2 - x1), (y2 - y1)
    return w / (h + 1e-6)


def calc_floor_collapse_ratio(box, frame_h):
    """
    Menghitung rasio posisi vertikal batas atas bounding box (y1) terhadap tinggi frame.
    Berfungsi mendeteksi longitudinal fall (jatuh membujur ke arah lensa kamera)
    di mana aspect_ratio tetap sempit namun seluruh tubuh berada di area lantai bawah.
    """
    return box[1] / (frame_h + 1e-6)


def get_box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def get_valid_point(kpts, person_kpts, idx1, idx2, conf_thresh=0.25):
    """Mengambil titik tengah jika kedua keypoint valid, atau salah satu yang valid."""
    c1 = person_kpts[idx1][2] if len(person_kpts) > idx1 else 0
    c2 = person_kpts[idx2][2] if len(person_kpts) > idx2 else 0
    if c1 > conf_thresh and c2 > conf_thresh:
        return ((kpts[idx1][0] + kpts[idx2][0]) / 2, (kpts[idx1][1] + kpts[idx2][1]) / 2)
    elif c1 > conf_thresh:
        return (kpts[idx1][0], kpts[idx1][1])
    elif c2 > conf_thresh:
        return (kpts[idx2][0], kpts[idx2][1])
    return None


# =============================================================================
# EXPORT VIDEO BROWSER-COMPATIBLE (H.264 / AVC1)
# =============================================================================

def export_browser_compatible_clip(frames: list, output_path: str, fps: int = 20) -> bool:
    """
    Meng-encode kumpulan frame menjadi video MP4 dengan codec H.264 / yuv420p
    menggunakan static binary FFmpeg dari imageio-ffmpeg.
    Dipastikan langsung dapat diputar pada elemen <video> browser tanpa kendala codec.
    """
    if not frames:
        return False
    try:
        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        h, w = frames[0].shape[:2]
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "bgr24",
            "-r", str(max(1, fps)),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-movflags", "+faststart",
            output_path
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        for f in frames:
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        proc.wait()
        return proc.returncode == 0
    except Exception as e:
        print(f"[ERROR] Transcoding video clip gagal: {e}")
        return False


def dispatch_alert_async(
    api_url: str,
    camera_id: str,
    camera_name: str,
    location: str,
    confidence: float,
    snapshot_frame: np.ndarray,
    clip_frames: list,
    fps: int = 20
):
    """Mengirim alert beserta snapshot dan video clip ke FastAPI secara non-blocking."""
    def _worker():
        ts = int(time.time() * 1000)
        thumb_file = f"temp_thumb_{camera_id}_{ts}.jpg"
        clip_file = f"temp_clip_{camera_id}_{ts}.mp4"
        files = {}
        opened_handles = []

        try:
            cv2.imwrite(thumb_file, snapshot_frame)
            if os.path.exists(thumb_file):
                f_thumb = open(thumb_file, "rb")
                opened_handles.append(f_thumb)
                files["thumbnail"] = ("thumb.jpg", f_thumb, "image/jpeg")

            has_clip = export_browser_compatible_clip(clip_frames, clip_file, fps=fps)
            if has_clip and os.path.exists(clip_file):
                f_clip = open(clip_file, "rb")
                opened_handles.append(f_clip)
                files["clip"] = ("clip.mp4", f_clip, "video/mp4")

            payload = {
                "camera_id": camera_id,
                "camera_name": camera_name,
                "location": location,
                "confidence": round(confidence, 2),
                "clip_duration_seconds": int(len(clip_frames) / max(1, fps)) if has_clip else 0
            }

            resp = requests.post(api_url, data=payload, files=files, timeout=12)
            if resp.status_code == 200:
                print(f"[ALERT DISPATCHED] Kamera '{camera_name}' (ID: {camera_id}) -> Insiden tercatat (HTTP 200)")
            else:
                print(f"[ALERT ERROR] Server merespon {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[ALERT EXCEPTION] Gagal mengirim alert ke backend: {e}")
        finally:
            for handle in opened_handles:
                try:
                    handle.close()
                except Exception:
                    pass
            if os.path.exists(thumb_file):
                try:
                    os.remove(thumb_file)
                except OSError:
                    pass
            if os.path.exists(clip_file):
                try:
                    os.remove(clip_file)
                except OSError:
                    pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


# =============================================================================
# PERSON TRACKER
# =============================================================================

class PersonTracker:
    def __init__(self, display_id, history_len=20):
        self.display_id = display_id
        self.hip_y_history = deque(maxlen=history_len)
        self.box_y_history = deque(maxlen=history_len)
        self.box_h_history = deque(maxlen=history_len)
        self.posture_flags = deque(maxlen=POSTURE_WINDOW)
        self.fall_suspected_since = None
        self.last_center = None
        self.last_box = None
        self.max_upright_height = 0.0
        self.baseline_fcr = None
        self.last_seen_time = time.time()
        self.last_eval_time = None
        self.alert_sent = False

        # State Machine Flags untuk validasi riwayat jatuh (RUMI v2)
        self.has_been_upright = False
        self.last_fall_motion_time = -999.0
        self.fall_confirmed_latch = False
        self.suspect_grace_until = -999.0
        self.is_longitudinal_fall = False
        self.upright_streak = 0
        self.peak_v = 0.0
        self.peak_short_drop = 0.0

    def update_posture_flag(self, is_fall_posture):
        self.posture_flags.append(bool(is_fall_posture))
        if len(self.posture_flags) < 2:
            return bool(is_fall_posture)
        true_ratio = sum(self.posture_flags) / len(self.posture_flags)
        return true_ratio >= POSTURE_VOTE_RATIO

    def update_state(self, box, hip_pos=None, is_upright_normal=False, current_time=None):
        center = get_box_center(box)
        box_h = box[3] - box[1]
        self.box_y_history.append(center[1])
        self.box_h_history.append(box_h)

        if is_upright_normal and box_h > self.max_upright_height:
            self.max_upright_height = box_h

        if hip_pos is not None:
            self.hip_y_history.append(hip_pos[1])

        self.last_center = center
        self.last_box = box
        now = current_time if current_time is not None else time.time()
        self.last_seen_time = now
        self.last_eval_time = now

    def vertical_velocity(self):
        """Kecepatan turun instan per-step (px/step)."""
        if len(self.hip_y_history) >= 2:
            instant_v = self.hip_y_history[-1] - self.hip_y_history[-2]
            window_v = (self.hip_y_history[-1] - self.hip_y_history[0]) / max(1, len(self.hip_y_history) - 1)
            return max(instant_v, window_v)
        if len(self.box_y_history) >= 2:
            instant_v = self.box_y_history[-1] - self.box_y_history[-2]
            window_v = (self.box_y_history[-1] - self.box_y_history[0]) / max(1, len(self.box_y_history) - 1)
            return max(instant_v, window_v)
        return 0.0

    def total_vertical_drop(self):
        """Total penurunan posisi Y dalam window 20 step."""
        drop_hip = (self.hip_y_history[-1] - min(self.hip_y_history)) if len(self.hip_y_history) >= 3 else 0.0
        drop_box = (self.box_y_history[-1] - min(self.box_y_history)) if len(self.box_y_history) >= 3 else 0.0
        return max(drop_hip, drop_box)

    def short_vertical_drop(self, n=8):
        """Penurunan vertikal cepat dalam window pendek (8 step) untuk menyaring duduk/membungkuk lambat."""
        drop_hip = (self.hip_y_history[-1] - min(list(self.hip_y_history)[-n:])) if len(self.hip_y_history) >= 3 else 0.0
        drop_box = (self.box_y_history[-1] - min(list(self.box_y_history)[-n:])) if len(self.box_y_history) >= 3 else 0.0
        return max(drop_hip, drop_box)

    def is_inactive(self, current_box, movement_threshold=25):
        if self.last_center is None:
            return True
        current_center = get_box_center(current_box)
        dist = math.dist(current_center, self.last_center)
        return dist < movement_threshold


# =============================================================================
# UNIFIED RUMI v2 SPATIO-TEMPORAL EVALUATOR (1:1 PARITY WITH COLAB BENCHMARK)
# =============================================================================

def evaluate_rumi_v2_step(
    tracker: PersonTracker,
    box,
    kpts,
    person_kpts,
    frame_h: int,
    current_time: float,
    room_state: Optional[Dict[str, Any]] = None,
    rf_model=None,
    rf_thresh: float = 0.60,
    severe_rf_thresh: float = 0.85,
) -> Dict[str, Any]:
    """
    Evaluasi kinematika + temporal RUMI v2 (sinkron 1:1 dengan eksperimen Colab Cell 10/11).
    Mendukung mode Rule-Based (default, ~70 FPS) maupun Hybrid RF8 opsional.
    """
    dt = (
        (current_time - tracker.last_eval_time)
        if (tracker.last_eval_time is not None and current_time > tracker.last_eval_time)
        else (1.0 / 25.0)
    )
    dt = min(max(dt, 1e-3), 0.5)

    frame_scale = frame_h / 480.0
    vel_thresh = VERTICAL_VELOCITY_THRESHOLD * frame_scale
    drop_thresh = SLOW_FALL_DROP_THRESHOLD * frame_scale
    inact_thresh = 25.0 * frame_scale

    box_h = float(box[3] - box[1])
    aspect_ratio = calc_aspect_ratio(box)
    floor_collapse_ratio = calc_floor_collapse_ratio(box, frame_h)
    bot_r = float(box[3]) / (frame_h + 1e-6)

    hip_pos = get_valid_point(kpts, person_kpts, L_HIP, R_HIP, 0.35)
    if hip_pos is not None and hip_pos[1] > (frame_h * 0.96):
        hip_pos = None

    shoulder_pos = get_valid_point(kpts, person_kpts, L_SHOULDER, R_SHOULDER, 0.30)
    knee_pos = get_valid_point(kpts, person_kpts, L_KNEE, R_KNEE, 0.25)
    ankle_pos = get_valid_point(kpts, person_kpts, L_ANKLE, R_ANKLE, 0.25)

    has_valid_torso = (shoulder_pos is not None and hip_pos is not None)
    torso_angle = calc_torso_angle(shoulder_pos, hip_pos) if has_valid_torso else None
    torso_len = math.dist(shoulder_pos, hip_pos) if has_valid_torso else 0.0
    sh_hip_dy_norm = ((hip_pos[1] - shoulder_pos[1]) / max(1.0, box_h)) if has_valid_torso else 0.0

    # Fix ordering: cek inaktivitas terhadap frame sebelumnya, lalu update history sebelum hitung kinematika
    inactive = tracker.is_inactive(box, movement_threshold=inact_thresh)
    tracker.update_state(box, hip_pos, current_time=current_time)
    v_velocity = tracker.vertical_velocity()
    total_drop = tracker.total_vertical_drop()
    short_drop = tracker.short_vertical_drop(8)
    v_vel_sec = (v_velocity / max(frame_scale, 1e-6)) / dt
    h_ratio = (box_h / tracker.max_upright_height) if tracker.max_upright_height > 0 else 1.0
    fcr_delta = (floor_collapse_ratio - tracker.baseline_fcr) if tracker.baseline_fcr is not None else 0.0

    # Guard Close-Up / Half-Body Webcam (memastikan lower-body tidak terlihat agar tidak salah blokir jatuh di lantai)
    lower_body_absent = (knee_pos is None and ankle_pos is None)
    is_closeup_upper_body = lower_body_absent and floor_collapse_ratio < 0.45 and (
        (shoulder_pos is not None and hip_pos is None)
        or (has_valid_torso and torso_len < (box_h * 0.14) and aspect_ratio < 0.95)
    )

    # 1. Deteksi Prone Forward-Fall (jatuh membujur ke depan di bagian bawah frame)
    is_prone_forward = (
        (floor_collapse_ratio > 0.46)
        and (bot_r >= 0.95)
        and (fcr_delta >= 0.30)
        and not is_closeup_upper_body
    )

    # 2. Guard Duduk Tegak Normal (dilindungi agar tidak membajak jatuh telungkup/ke depan)
    is_seated_upright = (
        has_valid_torso
        and torso_angle is not None
        and torso_angle < 24.0
        and sh_hip_dy_norm > 0.14
        and (tracker.max_upright_height == 0 or h_ratio >= 0.62)
        and aspect_ratio < 0.80
        and not is_prone_forward
    )

    # 3. Analisis Perspektif Ruangan (Parameter ke-4: floor_collapse_ratio)
    is_longitudinal_floor_pose = (
        (
            (box[3] > (frame_h * 0.68))
            and (floor_collapse_ratio > FLOOR_COLLAPSE_THRESHOLD)
            and (box_h < (frame_h * 0.52))
            and not is_seated_upright
            and not is_closeup_upper_body
        )
        or is_prone_forward
    )
    tracker.is_longitudinal_fall = is_longitudinal_floor_pose

    # 4. Hukum Anatomi Vertikal (Validasi Berdiri Tegak & Duduk Normal)
    is_anatomically_upright = is_closeup_upper_body or is_seated_upright
    if not is_longitudinal_floor_pose and has_valid_torso and not is_anatomically_upright:
        shoulder_above_hip = sh_hip_dy_norm > 0.12
        if ankle_pos is not None:
            hip_above_lower = (ankle_pos[1] - hip_pos[1]) > (box_h * 0.12)
        elif knee_pos is not None:
            hip_above_lower = (knee_pos[1] - hip_pos[1]) > (box_h * 0.08)
        else:
            hip_above_lower = True

        if shoulder_above_hip and hip_above_lower:
            if torso_angle is not None and torso_angle < 32.0 and aspect_ratio < 0.60:
                is_anatomically_upright = True
            elif aspect_ratio < 0.75 and (torso_angle is not None and torso_angle < 38.0):
                is_anatomically_upright = True

    # 5. Deteksi Lonjakan Kinetika Jatuh dengan Rapid-Descent Guard
    drop_ok = (total_drop > drop_thresh) and (short_drop > SHORT_DROP_GUARD_THRESHOLD * frame_scale)
    h_collapse_ok = (
        tracker.max_upright_height > 0
        and h_ratio < 0.52
        and (short_drop > 18.0 * frame_scale or v_velocity > vel_thresh * 0.6)
    )
    has_kinetic_drop = (
        (v_velocity > vel_thresh)
        or drop_ok
        or h_collapse_ok
        or is_prone_forward
    )

    if has_kinetic_drop and tracker.has_been_upright:
        tracker.last_fall_motion_time = current_time
        tracker.suspect_grace_until = max(tracker.suspect_grace_until, current_time + 3.5)
        tracker.peak_v = max(tracker.peak_v, v_velocity / max(frame_scale, 1e-6))
        tracker.peak_short_drop = max(tracker.peak_short_drop, short_drop / max(frame_scale, 1e-6))
        if room_state is not None:
            room_state["room_last_motion_time"] = current_time

    # 6. Identifikasi Postur Tergeletak di Lantai & Tingkat Keparahan (Rule-Based atau Hybrid RF8)
    geom_severe = (
        (aspect_ratio > 1.15)
        or (has_valid_torso and torso_angle is not None and torso_angle > 62.0 and floor_collapse_ratio > FLOOR_COLLAPSE_THRESHOLD)
        or (has_valid_torso and shoulder_pos[1] >= (hip_pos[1] - 8) and floor_collapse_ratio > FLOOR_COLLAPSE_THRESHOLD)
    )
    rf_prob = 0.0
    if rf_model is not None:
        x_vec = np.array(
            [[aspect_ratio, torso_angle or 0.0, v_vel_sec, floor_collapse_ratio, bot_r, h_ratio, fcr_delta, sh_hip_dy_norm]],
            dtype=np.float32,
        )
        rf_prob = float(rf_model.predict_proba(x_vec)[0, 1])
        is_floor_posture = (not is_closeup_upper_body) and (not is_anatomically_upright) and ((rf_prob >= rf_thresh) or is_prone_forward)
        is_severe = rf_prob >= severe_rf_thresh
        inact_fast = 0.35
    else:
        is_floor_posture = False
        if is_longitudinal_floor_pose:
            is_floor_posture = True
        elif not is_anatomically_upright and not is_closeup_upper_body:
            if has_valid_torso and shoulder_pos[1] >= (hip_pos[1] - 8):
                is_floor_posture = True
            elif torso_angle is not None and torso_angle > TORSO_ANGLE_THRESHOLD:
                is_floor_posture = True
            elif aspect_ratio > ASPECT_RATIO_THRESHOLD:
                is_floor_posture = True
        is_severe = geom_severe
        inact_fast = INACTIVITY_FAST_SECONDS

    recent_drop = (current_time - tracker.last_fall_motion_time) <= 4.0
    in_grace_period = current_time < tracker.suspect_grace_until
    room_recent_fall = False
    if room_state is not None:
        room_recent_fall = (
            (current_time - room_state.get("room_last_motion_time", 0.0) <= 4.0)
            and (current_time - room_state.get("room_last_upright_time", 0.0) <= 8.0)
        )
    has_fall_dynamics = (
        recent_drop
        or in_grace_period
        or room_recent_fall
        or (tracker.fall_suspected_since is not None and is_floor_posture)
    )

    # 7. Keputusan State Machine dengan Hysteresis Pemulihan Tegak
    if is_anatomically_upright:
        tracker.upright_streak += 1
        tracker.has_been_upright = True
        if not is_closeup_upper_body:
            if box_h > tracker.max_upright_height:
                tracker.max_upright_height = box_h
            if tracker.baseline_fcr is None or floor_collapse_ratio < tracker.baseline_fcr:
                tracker.baseline_fcr = floor_collapse_ratio
        if room_state is not None:
            room_state["room_last_upright_time"] = current_time
    else:
        tracker.upright_streak = 0

    should_reset = is_anatomically_upright and (
        (not tracker.fall_confirmed_latch and tracker.fall_suspected_since is None)
        or (tracker.upright_streak >= UPRIGHT_RECOVERY_FRAMES)
    )
    if should_reset:
        tracker.fall_confirmed_latch = False
        tracker.fall_suspected_since = None
        tracker.suspect_grace_until = -999.0
        tracker.alert_sent = False
        tracker.posture_flags.clear()
        tracker.peak_v = 0.0
        tracker.peak_short_drop = 0.0
        raw_posture_flag = False
    else:
        raw_posture_flag = bool(tracker.fall_confirmed_latch or (is_floor_posture and has_fall_dynamics))

    # 8. Temporal Buffer Voting & Two-Tier Adaptive Inactivity Confirmation
    posture_indicates_fall = tracker.update_posture_flag(raw_posture_flag)
    confirmed_fall = False
    if posture_indicates_fall:
        if tracker.fall_suspected_since is None:
            tracker.fall_suspected_since = current_time
        duration = current_time - tracker.fall_suspected_since
        strong_kin = (tracker.peak_v >= VERTICAL_VELOCITY_THRESHOLD) or (tracker.peak_short_drop >= 35.0)
        req_inactivity = inact_fast if (is_severe and strong_kin and floor_collapse_ratio > FLOOR_COLLAPSE_THRESHOLD) else INACTIVITY_SECONDS
        if (duration >= req_inactivity and inactive) or tracker.fall_confirmed_latch:
            confirmed_fall = True
            tracker.fall_confirmed_latch = True
    else:
        if not tracker.fall_confirmed_latch:
            tracker.fall_suspected_since = None
            tracker.alert_sent = False

    return {
        "aspect_ratio": aspect_ratio,
        "torso_angle": torso_angle or 0.0,
        "floor_collapse_ratio": floor_collapse_ratio,
        "v_velocity": v_velocity,
        "rf_prob": rf_prob,
        "posture_indicates_fall": posture_indicates_fall,
        "confirmed_fall": confirmed_fall,
    }


# =============================================================================
# THREADED CAMERA STREAM READER
# =============================================================================

class ThreadedCamera:
    """
    Membaca frame secara terus menerus dalam thread terpisah agar buffer tidak lag
    dan menangani auto-reconnect dengan exponential backoff jika stream terputus.
    """
    def __init__(self, source, on_status_change=None):
        self.source = int(source) if str(source).isdigit() else source
        self.on_status_change = on_status_change
        self.cap = None
        self.frame = None
        self.fps = 25.0
        self.is_running = False
        self.is_connected = False
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _notify_status(self, connected: bool):
        if self.is_connected != connected:
            self.is_connected = connected
            if self.on_status_change:
                self.on_status_change("ONLINE" if connected else "OFFLINE")

    def _capture_loop(self):
        backoff = 2
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                masked = mask_url(str(self.source))
                print(f"[STREAM] Mencoba koneksi ke {masked}...")
                self.cap = cv2.VideoCapture(self.source)

                if self.cap.isOpened():
                    fps_prop = self.cap.get(cv2.CAP_PROP_FPS)
                    if fps_prop and 5.0 <= fps_prop <= 60.0:
                        self.fps = fps_prop
                    self._notify_status(True)
                    backoff = 2
                    print(f"[STREAM] Berhasil tersambung ke {masked} ({self.fps:.1f} FPS)")
                else:
                    self._notify_status(False)
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 30)
                    continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Jika sumber adalah file video lokal (test_fall.mp4), loop otomatis dari awal
                if isinstance(self.source, str) and os.path.isfile(self.source):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.04)
                    continue

                self._notify_status(False)
                if self.cap:
                    self.cap.release()
                self.cap = None
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue

            with self.lock:
                self.frame = frame

            time.sleep(1.0 / (self.fps * 1.5))

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        self._notify_status(False)


# =============================================================================
# WORKER DETEKSI JATUH PER KAMERA
# =============================================================================

class FallDetectionWorker:
    """
    Worker deteksi jatuh independen per kamera.
    Mendukung frame sampling, circular buffer forensic MP4, dan auto-reconnect.
    """
    def __init__(
        self,
        camera_id: str = "default_cam",
        camera_name: str = "Kamera Utama",
        location: str = "Ruangan",
        stream_url: str = DEFAULT_VIDEO_SOURCE,
        model_instance: Optional[YOLO] = None,
        debug_gui: bool = False
    ):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.location = location
        self.stream_url = stream_url
        self.debug_gui = debug_gui
        self.is_running = False
        self.worker_thread = None

        # Memori global ruangan untuk mengatasi ByteTrack ID Switching saat jatuh
        self.room_last_upright_time = 0.0
        self.room_last_motion_time = 0.0

        # Shared YOLO model atau inisialisasi lokal
        if model_instance is not None:
            self.model = model_instance
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            custom_model_env = os.getenv("YOLO_MODEL_PATH")
            candidate_paths = [
                custom_model_env,
                os.path.join(BASE_DIR, "..", "yolo11_rumi.pt"),
                os.path.join(BASE_DIR, "yolo11_rumi.pt"),
                "yolo11_rumi.pt",
                os.path.join(BASE_DIR, "..", "yolo11n-pose.pt"),
                os.path.join(BASE_DIR, "yolo11n-pose.pt"),
                "yolo11n-pose.pt"
            ]
            model_path = next((p for p in candidate_paths if p and os.path.exists(p)), "yolo11_rumi.pt")
            print(f"[WORKER] Memuat model YOLO lokal ({device}): {model_path}")
            self.model = YOLO(model_path).to(device)

    def _report_status(self, status: str):
        masked = mask_url(self.stream_url)
        print(f"[STATUS] Kamera '{self.camera_name}' ({self.camera_id}): {status}")
        try:
            requests.patch(
                f"{CAMERA_STATUS_URL}/{self.camera_id}/status",
                json={"status": status},
                timeout=3
            )
        except Exception:
            pass

    def start(self):
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._run, daemon=True)
        self.worker_thread.start()
        return self

    def stop(self):
        self.is_running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)

    def _run(self):
        camera = ThreadedCamera(self.stream_url, on_status_change=self._report_status).start()
        trackers = {}
        last_alert_time = 0.0
        last_alert_pos = None
        prev_small_gray = None

        # Circular buffer untuk klip forensik (e.g. 5 detik sebelum jatuh)
        buffer_len = int(camera.fps * CLIP_BUFFER_SECONDS)
        pre_fall_buffer = deque(maxlen=max(30, buffer_len))
        post_fall_recording = False
        post_fall_frames = []
        pending_incident = None

        frame_count = 0
        inference_interval = max(1, int(camera.fps / INFERENCE_FPS))

        print(f"[WORKER] FallDetectionWorker dimulai untuk '{self.camera_name}' (ID: {self.camera_id})")

        while self.is_running:
            frame = camera.read()
            if frame is None:
                time.sleep(0.04)
                continue

            frame_count += 1
            pre_fall_buffer.append(frame)

            # Jika sedang merekam frame setelah jatuh
            if post_fall_recording:
                post_fall_frames.append(frame)
                if len(post_fall_frames) >= int(camera.fps * CLIP_BUFFER_SECONDS):
                    post_fall_recording = False
                    if pending_incident is not None:
                        combined_frames = list(pending_incident["pre_frames"]) + post_fall_frames
                        dispatch_alert_async(
                            api_url=ALERT_API_URL,
                            camera_id=self.camera_id,
                            camera_name=self.camera_name,
                            location=self.location,
                            confidence=pending_incident["confidence"],
                            snapshot_frame=pending_incident["snapshot_frame"],
                            clip_frames=combined_frames,
                            fps=int(camera.fps)
                        )
                    pending_incident = None
                    post_fall_frames = []

            # Frame Sampling: Jalankan inferensi YOLO hanya setiap N frame untuk hemat compute
            if frame_count % inference_interval != 0:
                time.sleep(0.005)
                continue

            # 1. SCENE CUT DETECTION (Sensitivitas 12.0 untuk transisi klip dalam ruangan yang sama)
            small_gray = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY)
            if prev_small_gray is not None:
                scene_diff = np.mean(cv2.absdiff(small_gray, prev_small_gray))
                if scene_diff > 12.0:
                    trackers.clear()
                    last_alert_pos = None
                    last_alert_time = 0.0
                    self.room_last_upright_time = 0.0
                    self.room_last_motion_time = 0.0
            prev_small_gray = small_gray

            # 2. INFERENSI POSE & TRACKING
            results = self.model.track(
                frame,
                imgsz=640,
                persist=True,
                conf=CONF_THRESHOLD,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            annotated_frame = results[0].plot() if self.debug_gui else None
            current_time = time.time()
            active_ids = set()
            frame_h, frame_w = frame.shape[:2]

            if results[0].boxes.id is not None and results[0].keypoints is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                keypoints_xy = results[0].keypoints.xy.cpu().numpy()
                keypoints_data = results[0].keypoints.data.cpu().numpy()

                frame_scale = frame_h / 480.0
                relink_radius = 160.0 * frame_scale
                room_state = {
                    "room_last_upright_time": self.room_last_upright_time,
                    "room_last_motion_time": self.room_last_motion_time,
                }

                for i, (box, raw_track_id, kpts) in enumerate(zip(boxes, track_ids, keypoints_xy)):
                    box_center = get_box_center(box)

                    # 3. SPATIAL ID RE-LINKING (Radius diskalakan ke resolusi agar tidak terputus saat jatuh membujur)
                    if raw_track_id not in trackers:
                        inherited_tracker = None
                        min_dist = relink_radius
                        for old_id, old_tracker in list(trackers.items()):
                            if old_id not in track_ids and (current_time - old_tracker.last_seen_time) < 3.0:
                                if old_tracker.last_center is not None:
                                    dist = math.dist(box_center, old_tracker.last_center)
                                    if dist < min_dist:
                                        min_dist = dist
                                        inherited_tracker = (old_id, old_tracker)

                        if inherited_tracker is not None:
                            old_id, tracker_obj = inherited_tracker
                            tracker_obj.display_id = raw_track_id
                            tracker_obj.last_seen_time = current_time
                            del trackers[old_id]
                            trackers[raw_track_id] = tracker_obj
                        else:
                            trackers[raw_track_id] = PersonTracker(display_id=raw_track_id)

                    tracker = trackers[raw_track_id]
                    active_ids.add(raw_track_id)

                    person_kpts = keypoints_data[i]
                    step_out = evaluate_rumi_v2_step(
                        tracker=tracker,
                        box=box,
                        kpts=kpts,
                        person_kpts=person_kpts,
                        frame_h=frame_h,
                        current_time=current_time,
                        room_state=room_state,
                    )
                    aspect_ratio = step_out["aspect_ratio"]
                    floor_collapse_ratio = step_out["floor_collapse_ratio"]
                    v_velocity = step_out["v_velocity"]
                    posture_indicates_fall = step_out["posture_indicates_fall"]
                    confirmed_fall = step_out["confirmed_fall"]

                    status_text = "Normal"
                    if confirmed_fall:
                        status_text = "FALL DETECTED"
                        if not tracker.alert_sent:
                            is_same_spot_spam = (
                                last_alert_pos is not None
                                and math.dist(box_center, last_alert_pos) < 100
                                and (current_time - last_alert_time) < SAME_SPOT_COOLDOWN
                            )
                            if not is_same_spot_spam:
                                print(f"[FALL DETECTED] Person #{tracker.display_id} di '{self.camera_name}'!")
                                snapshot_img = frame.copy()
                                x1, y1, x2, y2 = [int(v) for v in box]
                                cv2.rectangle(snapshot_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(
                                    snapshot_img,
                                    f"FALL DETECTED: ID {tracker.display_id}",
                                    (x1, max(25, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 0, 255),
                                    2
                                )

                                post_fall_recording = True
                                post_fall_frames = []
                                pending_incident = {
                                    "confidence": 0.92,
                                    "snapshot_frame": snapshot_img,
                                    "pre_frames": list(pre_fall_buffer)
                                }

                                last_alert_time = current_time
                                last_alert_pos = box_center
                            tracker.alert_sent = True
                    elif posture_indicates_fall:
                        status_text = "FALL DETECTED" if tracker.alert_sent else "Suspected Fall"

                    if self.debug_gui and annotated_frame is not None:
                        x1, y1 = int(box[0]), int(box[1])
                        text_y = y1 - 10 if y1 > 30 else y1 + 25
                        color = (0, 0, 255) if "FALL" in status_text else ((0, 165, 255) if "Suspected" in status_text else (0, 255, 0))
                        label = f"ID {tracker.display_id}: {status_text} | AR:{aspect_ratio:.2f} | FCR:{floor_collapse_ratio:.2f} | Vel:{v_velocity:.1f}"
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52,
                            color,
                            2,
                        )

                self.room_last_upright_time = room_state["room_last_upright_time"]
                self.room_last_motion_time = room_state["room_last_motion_time"]

            # Bersihkan tracker kadaluarsa
            for raw_id, tracker in list(trackers.items()):
                if raw_id not in active_ids:
                    if (current_time - tracker.last_seen_time) > 3.0:
                        del trackers[raw_id]

            if self.debug_gui and annotated_frame is not None:
                cv2.imshow(f"Rumi Vision - {self.camera_name}", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        camera.stop()
        if self.debug_gui:
            cv2.destroyAllWindows()
        print(f"[WORKER] FallDetectionWorker untuk '{self.camera_name}' dihentikan.")


# =============================================================================
# WRAPPER ADAPTER UNTUK FASTAPI & CAMERA STREAM MANAGER
# =============================================================================

class FallDetector:
    """
    Wrapper adapter untuk inferensi YOLO11-pose dan pipeline deteksi jatuh RUMI (Kalibrasi 4D Grid Search).
    Kompatibel dengan CameraStreamManager dan API endpoint /api/model.
    """
    def __init__(self, model_path: Optional[str] = None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.infer_lock = threading.Lock()
        self.camera_states: Dict[str, Dict[str, Any]] = {}

        candidate_paths = [
            model_path,
            os.getenv("YOLO_MODEL_PATH"),
            os.path.join(BASE_DIR, "yolo11_rumi.pt"),
            os.path.join(BASE_DIR, "..", "yolo11_rumi.pt"),
            os.path.join(BASE_DIR, "..", "..", "yolo11_rumi.pt"),
            "yolo11_rumi.pt",
            os.path.join(BASE_DIR, "yolo11n-pose.pt"),
            os.path.join(BASE_DIR, "..", "yolo11n-pose.pt"),
            "yolo11n-pose.pt",
            os.path.join(BASE_DIR, "yolov8n-pose.pt"),
            os.path.join(BASE_DIR, "..", "yolov8n-pose.pt"),
            "yolov8n-pose.pt"
        ]
        self.model_path = next((p for p in candidate_paths if p and os.path.exists(p)), os.path.join(BASE_DIR, "yolo11_rumi.pt"))
        self.model = None
        self.is_loaded = False
        self._load_model()

    def _get_cam_state(self, camera_id: Optional[str]) -> Dict[str, Any]:
        cam_key = camera_id or "default"
        if cam_key not in self.camera_states:
            self.camera_states[cam_key] = {
                "trackers": {},
                "room_last_upright_time": 0.0,
                "room_last_motion_time": 0.0,
                "last_alert_time": 0.0,
                "last_alert_pos": None,
                "prev_small_gray": None,
            }
        return self.camera_states[cam_key]

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[FallDetector WARNING] File model '{self.model_path}' belum ditemukan.")
            return
        try:
            self.model = YOLO(self.model_path).to(self.device)
            self.is_loaded = True
            print(f"[FallDetector] Berhasil memuat model YOLO ({self.device}): {self.model_path}")
        except Exception as e:
            print(f"[FallDetector ERROR] Gagal memuat model: {e}")

    def reset_trackers(self, camera_id: Optional[str] = None):
        if camera_id:
            self.camera_states.pop(camera_id, None)
        else:
            self.camera_states.clear()

    def process_cv2_frame(
        self,
        frame: np.ndarray,
        on_fall_callback=None,
        camera_name: str = "Ruang Tamu · Cam 01",
        location: str = "Ruang Tamu",
        camera_id: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Memproses 1 frame OpenCV numpy array menggunakan YOLO11-pose + Kalibrasi 4D Grid Search.
        Mengembalikan (annotated_frame, telemetry).
        """
        start_time = time.time()
        telemetry: Dict[str, Any] = {
            "person_detected": False,
            "fall_detected": False,
            "status": "NORMAL",
            "confidence": 0.0,
            "aspect_ratio": 0.0,
            "torso_angle": 0.0,
            "floor_collapse_ratio": 0.0,
            "v_velocity": 0.0,
            "latency_ms": 0.0,
            "track_id": None,
            "bounding_box": []
        }

        if not self.is_loaded or self.model is None or frame is None:
            return frame, telemetry

        state = self._get_cam_state(camera_id)
        trackers: Dict[int, PersonTracker] = state["trackers"]

        annotated_frame = frame.copy()
        current_time = time.time()
        frame_h, frame_w = frame.shape[:2]

        # 1. Scene cut detection
        small_gray = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY)
        if state["prev_small_gray"] is not None:
            scene_diff = np.mean(cv2.absdiff(small_gray, state["prev_small_gray"]))
            if scene_diff > 12.0:
                trackers.clear()
                state["last_alert_pos"] = None
                state["last_alert_time"] = 0.0
                state["room_last_upright_time"] = 0.0
                state["room_last_motion_time"] = 0.0
        state["prev_small_gray"] = small_gray

        with self.infer_lock:
            try:
                results = self.model.track(
                    frame,
                    imgsz=640,
                    persist=True,
                    conf=CONF_THRESHOLD,
                    tracker="bytetrack.yaml",
                    verbose=False
                )
            except Exception:
                try:
                    results = self.model.predict(
                        frame,
                        imgsz=640,
                        conf=CONF_THRESHOLD,
                        verbose=False
                    )
                except Exception:
                    return frame, telemetry

        if not results or len(results) == 0:
            return annotated_frame, telemetry

        res = results[0]
        if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
            return annotated_frame, telemetry

        boxes = res.boxes.xyxy.cpu().numpy()
        keypoints_xy = res.keypoints.xy.cpu().numpy()
        keypoints_data = res.keypoints.data.cpu().numpy()

        if res.boxes.id is not None:
            track_ids = res.boxes.id.int().cpu().tolist()
        else:
            track_ids = list(range(len(boxes)))

        confs = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else [0.9] * len(boxes)

        active_ids = set()
        fall_confirmed_any = False

        frame_scale = frame_h / 480.0
        relink_radius = 160.0 * frame_scale

        for i, (box, raw_track_id, kpts, conf) in enumerate(zip(boxes, track_ids, keypoints_xy, confs)):
            box_center = get_box_center(box)

            # Spatial ID Re-linking (Radius diskalakan ke resolusi agar tidak terputus saat jatuh membujur)
            if raw_track_id not in trackers:
                inherited_tracker = None
                min_dist = relink_radius
                for old_id, old_tracker in list(trackers.items()):
                    if old_id not in track_ids and (current_time - old_tracker.last_seen_time) < 3.0:
                        if old_tracker.last_center is not None:
                            dist = math.dist(box_center, old_tracker.last_center)
                            if dist < min_dist:
                                min_dist = dist
                                inherited_tracker = (old_id, old_tracker)

                if inherited_tracker is not None:
                    old_id, tracker_obj = inherited_tracker
                    tracker_obj.display_id = raw_track_id
                    tracker_obj.last_seen_time = current_time
                    del trackers[old_id]
                    trackers[raw_track_id] = tracker_obj
                else:
                    trackers[raw_track_id] = PersonTracker(display_id=raw_track_id)

            tracker = trackers[raw_track_id]
            active_ids.add(raw_track_id)

            person_kpts = keypoints_data[i]
            step_out = evaluate_rumi_v2_step(
                tracker=tracker,
                box=box,
                kpts=kpts,
                person_kpts=person_kpts,
                frame_h=frame_h,
                current_time=current_time,
                room_state=state,
                rf_model=getattr(self, "rf_model", None),
            )
            aspect_ratio = step_out["aspect_ratio"]
            torso_angle = step_out["torso_angle"]
            floor_collapse_ratio = step_out["floor_collapse_ratio"]
            v_velocity = step_out["v_velocity"]
            posture_indicates_fall = step_out["posture_indicates_fall"]
            confirmed_fall = step_out["confirmed_fall"]

            status_text = "NORMAL"
            if confirmed_fall:
                status_text = "FALL DETECTED"
                fall_confirmed_any = True

                if not tracker.alert_sent:
                    is_same_spot_spam = (
                        state["last_alert_pos"] is not None
                        and math.dist(box_center, state["last_alert_pos"]) < 100
                        and (current_time - state["last_alert_time"]) < SAME_SPOT_COOLDOWN
                    )
                    if not is_same_spot_spam and on_fall_callback:
                        try:
                            snapshot_img = annotated_frame.copy()
                            x1, y1, x2, y2 = [int(v) for v in box]
                            cv2.rectangle(snapshot_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(
                                snapshot_img,
                                f"FALL DETECTED: ID {tracker.display_id}",
                                (x1, max(25, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2
                            )
                            on_fall_callback(snapshot_img, tracker.display_id, float(conf))
                            state["last_alert_time"] = current_time
                            state["last_alert_pos"] = box_center
                        except Exception as e:
                            print(f"[FallDetector] Callback error: {e}")
                    tracker.alert_sent = True
            elif posture_indicates_fall:
                status_text = "FALL DETECTED" if tracker.alert_sent else "SUSPECTED FALL"

            # Draw visual bounding box and telemetry label
            x1, y1, x2, y2 = [int(v) for v in box]
            color = (0, 0, 255) if "FALL DETECTED" in status_text else ((0, 165, 255) if "SUSPECTED" in status_text else (0, 255, 0))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {tracker.display_id}: {status_text} | AR:{aspect_ratio:.2f} | FCR:{floor_collapse_ratio:.2f} | Vel:{v_velocity:.1f}"
            cv2.putText(
                annotated_frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2
            )

            telemetry.update({
                "person_detected": True,
                "fall_detected": bool(fall_confirmed_any),
                "status": status_text,
                "confidence": round(float(conf), 2),
                "aspect_ratio": round(float(aspect_ratio), 2),
                "torso_angle": round(float(torso_angle or 0.0), 2),
                "floor_collapse_ratio": round(float(floor_collapse_ratio), 2),
                "v_velocity": round(float(v_velocity), 2),
                "track_id": tracker.display_id,
                "bounding_box": [float(b) for b in box]
            })

        for raw_id, tr in list(trackers.items()):
            if raw_id not in active_ids and (current_time - tr.last_seen_time) > 3.0:
                del trackers[raw_id]

        telemetry["latency_ms"] = round((time.time() - start_time) * 1000, 2)
        return annotated_frame, telemetry

    def process_frame(self, frame_bytes: Optional[bytes] = None, return_annotated: bool = False) -> Dict[str, Any]:
        start_time = time.time()
        if frame_bytes is None:
            return self.simulate_fall()

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
        return {
            "fall_detected": True,
            "confidence": 0.98,
            "person_detected": True,
            "status": "FALL DETECTED",
            "aspect_ratio": 1.45,
            "torso_angle": 68.5,
            "floor_collapse_ratio": 0.52,
            "v_velocity": 34.0,
            "bounding_box": [80.0, 250.0, 460.0, 390.0],
            "latency_ms": 24.5,
            "source": "simulation_trigger"
        }

# Global singleton detector
detector = FallDetector()


# =============================================================================
# STANDALONE EXECUTION (Untuk pengujian lokal)
# =============================================================================


def main():
    print("=== Rumi Fall Detection Engine (Standalone Mode) ===")
    worker = FallDetectionWorker(
        camera_id="cam_demo_01",
        camera_name="Ruang Tamu",
        location="Lantai 1",
        stream_url=DEFAULT_VIDEO_SOURCE,
        debug_gui=DEBUG_GUI
    )
    try:
        worker.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nMenghentikan worker...")
        worker.stop()
        print("Selesai.")


if __name__ == "__main__":
    main()
