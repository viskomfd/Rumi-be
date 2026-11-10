import unittest
import uuid

import httpx

from app.core.supabase import db
from app.main import app


class AuthAccountTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        db.is_connected = False
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_profile_password_and_logout_all(self):
        email = f"account-{uuid.uuid4()}@example.com"
        register = await self.client.post("/api/auth/register", json={
            "name": "Budi Santoso",
            "email": email,
            "password": "password123",
        })
        self.assertEqual(register.status_code, 201)
        token = register.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        profile = await self.client.patch("/api/auth/me", headers=headers, json={
            "name": "Budi Baru",
            "elderly_name": "Ibu Siti",
        })
        self.assertEqual(profile.status_code, 200)
        self.assertEqual(profile.json()["elderly_name"], "Ibu Siti")

        wrong_password = await self.client.post("/api/auth/change-password", headers=headers, json={
            "current_password": "salah",
            "new_password": "password456",
        })
        self.assertEqual(wrong_password.status_code, 400)

        changed = await self.client.post("/api/auth/change-password", headers=headers, json={
            "current_password": "password123",
            "new_password": "password456",
        })
        self.assertEqual(changed.status_code, 204)
        self.assertEqual((await self.client.get("/api/auth/me", headers=headers)).status_code, 401)

        login = await self.client.post("/api/auth/login", json={
            "email": email,
            "password": "password456",
        })
        new_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        self.assertEqual((await self.client.post("/api/auth/logout-all", headers=new_headers)).status_code, 204)
        self.assertEqual((await self.client.get("/api/auth/me", headers=new_headers)).status_code, 401)


if __name__ == "__main__":
    unittest.main()
