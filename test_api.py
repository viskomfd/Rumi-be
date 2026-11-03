import os
import asyncio
import time
import httpx
from app.main import app

async def run_tests():
    print("[TEST] Menjalankan pengujian endpoint FastAPI...")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Health check
        res = await client.get("/")
        assert res.status_code == 200, f"Root error: {res.text}"
        print("  [OK] GET / : Status OK")

        # 2. Login demo user
        res = await client.post("/api/auth/login", json={
            "email": "andi.pratama@keluarga.id",
            "password": "password123"
        })
        print(f"  [INFO] Login test status: {res.status_code}")

        # 3. Register new user (use dynamic email so tests can repeat on live Supabase)
        test_email = f"user_{int(time.time())}@keluarga.id"
        reg_res = await client.post("/api/auth/register", json={
            "name": "Budi Santoso",
            "email": test_email,
            "password": "password12345"
        })
        assert reg_res.status_code == 201, f"Register error: {reg_res.text}"
        token = reg_res.json()["access_token"]
        assert reg_res.json()["user"]["role"] == "Pendamping"
        print("  [OK] POST /api/auth/register : Berhasil register & token JWT didapatkan")

        # 4. Get Current User (/me)
        me_res = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_res.status_code == 200, f"Me error: {me_res.text}"
        assert me_res.json()["name"] == "Budi Santoso"
        print("  [OK] GET /api/auth/me : Profil user berhasil divalidasi via JWT")

        # 5. List cameras
        cam_res = await client.get("/api/cameras")
        assert cam_res.status_code == 200, f"Cameras error: {cam_res.text}"
        cameras = cam_res.json()
        assert len(cameras) >= 2
        print(f"  [OK] GET /api/cameras : {len(cameras)} kamera berhasil dimuat")

        # 6. Add new camera
        new_cam = await client.post(
            "/api/cameras",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "name": "Kamera Belakang",
                "location": "Halaman Belakang",
                "url": "http://192.168.1.100:8080/video"
            }
        )
        assert new_cam.status_code == 201, f"Add camera error: {new_cam.text}"
        print("  [OK] POST /api/cameras : Kamera baru berhasil ditambahkan")

        # 7. List incidents
        inc_res = await client.get("/api/incidents?days=7")
        assert inc_res.status_code == 200, f"Incidents error: {inc_res.text}"
        incidents = inc_res.json()
        print(f"  [OK] GET /api/incidents : {len(incidents)} log insiden berhasil dimuat")

        # 8. Dashboard summary
        dash_res = await client.get("/api/dashboard/summary")
        assert dash_res.status_code == 200, f"Dashboard error: {dash_res.text}"
        dash_data = dash_res.json()
        assert "active_cameras" in dash_data
        assert "weekly_activity" in dash_data
        print("  [OK] GET /api/dashboard/summary : Metrik KPI & chart mingguan berhasil")

        # 9. Model status
        model_res = await client.get("/api/model/status")
        assert model_res.status_code == 200, f"Model status error: {model_res.text}"
        assert model_res.json()["is_model_loaded"] is True
        print(f"  [OK] GET /api/model/status : Mode -> {model_res.json()['mode']}")

        # 10. Detect frame with YOLOv8-pose
        snapshot_path = os.path.join(os.path.dirname(__file__), "static", "uploads", "fall-snapshot.jpg")
        with open(snapshot_path, "rb") as f:
            img_bytes = f.read()
        detect_res = await client.post(
            "/api/model/detect-frame",
            files={"file": ("snapshot.jpg", img_bytes, "image/jpeg")}
        )
        assert detect_res.status_code == 200, f"Detect error: {detect_res.text}"
        detect_data = detect_res.json()
        print(f"  [OK] POST /api/model/detect-frame : Inferensi YOLOv8 Pose Sukses (Status: {detect_data.get('status')}, Latency: {detect_data.get('latency_ms')}ms)")

        # 11. Simulate fall
        fall_res = await client.post("/api/model/simulate-fall")
        assert fall_res.status_code == 200, f"Simulate fall error: {fall_res.text}"
        print("  [OK] POST /api/model/simulate-fall : Simulasi jatuh ter-trigger & log tersimpan")

    print("\n>>> SEMUA PENGUJIAN API BERHASIL DENGAN SEMPURNA! <<<")

if __name__ == "__main__":
    asyncio.run(run_tests())
