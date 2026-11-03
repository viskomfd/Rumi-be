import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from app.core.config import settings

logger = logging.getLogger("rumi.db")

class SupabaseService:
    def __init__(self):
        self.client = None
        self.is_connected = False
        self._init_client()
        # In-memory store fallback jika Supabase credentials belum diisi
        self._init_mock_store()

    def _init_client(self):
        url = settings.SUPABASE_URL.strip()
        if url.endswith("/rest/v1/"):
            url = url[:-9]
        elif url.endswith("/rest/v1"):
            url = url[:-8]
        url = url.rstrip("/")

        key = (settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_KEY).strip()

        if url and key and not url.startswith("https://your-project"):
            try:
                from supabase import create_client, Client
                self.client: Client = create_client(url, key)
                self.is_connected = True
                logger.info("[Database] Berhasil terhubung ke Supabase!")
            except Exception as e:
                logger.warning(f"[Database] Gagal terhubung ke Supabase: {e}. Menggunakan in-memory mock store.")
        else:
            logger.info("[Database] Supabase URL/Key belum dikonfigurasi di .env. Menggunakan in-memory mock store.")

    def _init_mock_store(self):
        """Mock in-memory database jika Supabase belum dikonfigurasi."""
        self._mock_users: List[Dict[str, Any]] = [
            {
                "id": "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
                "name": "Andi Pratama",
                "email": "andi.pratama@keluarga.id",
                # bcrypt hash untuk 'password123'
                "hashed_password": "$2b$12$EXI5IEvOzDdsutOy7bSjeOKvTq2ulS9CAZkxQbIr814KP9u3besge",
                "role": "Pendamping",
                "avatar_initials": "AP",
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        ]
        self._mock_cameras: List[Dict[str, Any]] = [
            {
                "id": "c1111111-1111-1111-1111-111111111111",
                "user_id": "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
                "name": "Ruang Tamu",
                "location": "Ruang Tamu",
                "stream_url": "http://192.168.1.50:8080/video",
                "protocol": "HTTP",
                "status": "online",
                "is_active": True,
                "created_at": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
            },
            {
                "id": "c2222222-2222-2222-2222-222222222222",
                "user_id": "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
                "name": "Garasi",
                "location": "Garasi",
                "stream_url": "rtsp://admin:123456@192.168.1.53:554/stream1",
                "protocol": "RTSP",
                "status": "offline",
                "is_active": True,
                "created_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
            }
        ]
        self._mock_incidents: List[Dict[str, Any]] = [
            {
                "id": str(uuid.uuid4()),
                "camera_id": "c1111111-1111-1111-1111-111111111111",
                "camera_name": "Ruang Tamu · Cam 01",
                "location": "Ruang Tamu",
                "person_name": "Pak Hartono",
                "detected_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
                "confidence": 0.98,
                "snapshot_url": "/static/uploads/fall-snapshot.jpg",
                "clip_duration": "8 dtk",
                "response_time_seconds": 2.1,
                "status": "unresolved"
            },
            {
                "id": str(uuid.uuid4()),
                "camera_id": None,
                "camera_name": "Kamar Utama · Cam 02",
                "location": "Kamar Utama",
                "person_name": "Pak Hartono",
                "detected_at": (datetime.now(timezone.utc) - timedelta(days=1, hours=3)).isoformat(),
                "confidence": 0.95,
                "snapshot_url": "/static/uploads/fall-snapshot.jpg",
                "clip_duration": "12 dtk",
                "response_time_seconds": 2.4,
                "status": "resolved"
            },
            {
                "id": str(uuid.uuid4()),
                "camera_id": None,
                "camera_name": "Dapur · Cam 03",
                "location": "Dapur",
                "person_name": "Pak Hartono",
                "detected_at": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(),
                "confidence": 0.92,
                "snapshot_url": "/static/uploads/fall-snapshot.jpg",
                "clip_duration": "6 dtk",
                "response_time_seconds": 2.6,
                "status": "resolved"
            }
        ]
        self._mock_settings: Dict[str, Any] = {
            "popup_alert": True,
            "sound_alarm": True,
            "push_notification": True,
            "daily_digest": False
        }

    # ==========================
    # USER OPERATIONS
    # ==========================
    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        email = email.lower().strip()
        if self.is_connected:
            try:
                res = self.client.table("users").select("*").eq("email", email).execute()
                if res.data and len(res.data) > 0:
                    return res.data[0]
            except Exception as e:
                logger.error(f"[Supabase] Error get_user_by_email: {e}")
        
        for u in self._mock_users:
            if u["email"].lower() == email:
                return u
        return None

    async def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        if self.is_connected:
            try:
                res = self.client.table("users").select("*").eq("id", user_id).execute()
                if res.data and len(res.data) > 0:
                    return res.data[0]
            except Exception as e:
                logger.error(f"[Supabase] Error get_user_by_id: {e}")
        
        for u in self._mock_users:
            if u["id"] == user_id:
                return u
        return None

    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        user_data["id"] = user_data.get("id") or str(uuid.uuid4())
        user_data["created_at"] = datetime.now(timezone.utc).isoformat()
        user_data["email"] = user_data["email"].lower().strip()

        if self.is_connected:
            try:
                res = self.client.table("users").insert(user_data).execute()
                if res.data:
                    return res.data[0]
            except Exception as e:
                logger.error(f"[Supabase] Error create_user: {e}")

        self._mock_users.append(user_data)
        return user_data

    # ==========================
    # CAMERA OPERATIONS
    # ==========================
    async def get_cameras(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.is_connected:
            try:
                query = self.client.table("cameras").select("*")
                if user_id:
                    query = query.eq("user_id", user_id)
                res = query.order("created_at", desc=False).execute()
                return res.data or []
            except Exception as e:
                logger.error(f"[Supabase] Error get_cameras: {e}")

        if user_id:
            return [c for c in self._mock_cameras if c.get("user_id") == user_id]
        return self._mock_cameras

    async def create_camera(self, camera_data: Dict[str, Any]) -> Dict[str, Any]:
        camera_data["id"] = camera_data.get("id") or str(uuid.uuid4())
        camera_data["created_at"] = datetime.now(timezone.utc).isoformat()

        if self.is_connected:
            try:
                res = self.client.table("cameras").insert(camera_data).execute()
                if res.data:
                    return res.data[0]
            except Exception as e:
                logger.error(f"[Supabase] Error create_camera: {e}")

        self._mock_cameras.append(camera_data)
        return camera_data

    async def delete_camera(self, camera_id: str, user_id: Optional[str] = None) -> bool:
        if self.is_connected:
            try:
                query = self.client.table("cameras").delete().eq("id", camera_id)
                if user_id:
                    query = query.eq("user_id", user_id)
                query.execute()
                return True
            except Exception as e:
                logger.error(f"[Supabase] Error delete_camera: {e}")

        initial_len = len(self._mock_cameras)
        self._mock_cameras = [
            c for c in self._mock_cameras
            if not (c["id"] == camera_id and (not user_id or c.get("user_id") == user_id))
        ]
        return len(self._mock_cameras) < initial_len

    # ==========================
    # INCIDENT OPERATIONS
    # ==========================
    async def get_incidents(self, days: Optional[int] = None, location: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.is_connected:
            try:
                query = self.client.table("incidents").select("*").order("detected_at", desc=True)
                if location and location.lower() != "all":
                    query = query.ilike("location", f"%{location}%")
                if days:
                    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
                    query = query.gte("detected_at", start_date)
                res = query.execute()
                return res.data
            except Exception as e:
                logger.error(f"[Supabase] Error get_incidents: {e}")

        results = self._mock_incidents
        if location and location.lower() != "all":
            results = [inc for inc in results if location.lower() in inc.get("location", "").lower()]
        return results

    async def create_incident(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        incident_data["id"] = incident_data.get("id") or str(uuid.uuid4())
        incident_data["detected_at"] = incident_data.get("detected_at") or datetime.now(timezone.utc).isoformat()

        if self.is_connected:
            try:
                res = self.client.table("incidents").insert(incident_data).execute()
                if res.data:
                    return res.data[0]
            except Exception as e:
                logger.error(f"[Supabase] Error create_incident: {e}")

        self._mock_incidents.insert(0, incident_data)
        return incident_data

    # ==========================
    # SETTINGS OPERATIONS
    # ==========================
    async def get_settings(self) -> Dict[str, Any]:
        return self._mock_settings

    async def update_settings(self, settings_data: Dict[str, Any]) -> Dict[str, Any]:
        self._mock_settings.update(settings_data)
        return self._mock_settings

db = SupabaseService()
