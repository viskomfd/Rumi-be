-- ====================================================================
-- RUMI FALL DETECTION - SUPABASE DATABASE SCHEMA
-- Jalankan skrip ini di: Supabase Dashboard -> SQL Editor -> New Query
-- ====================================================================

-- 1. Enable UUID extension jika belum aktif
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 2. TABEL PENGGUNA (USERS)
CREATE TABLE IF NOT EXISTS public.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    role VARCHAR(100) DEFAULT 'Pendamping',
    avatar_initials VARCHAR(10) DEFAULT 'RA',
    elderly_name VARCHAR(100),
    token_version INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- Aman dijalankan pada database lama yang tabel users-nya sudah ada.
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS elderly_name VARCHAR(100);
ALTER TABLE public.users ADD COLUMN IF NOT EXISTS token_version INTEGER NOT NULL DEFAULT 0;

-- 3. TABEL KAMERA CCTV (CAMERAS)
CREATE TABLE IF NOT EXISTS public.cameras (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    location VARCHAR(255) NOT NULL,
    stream_url TEXT NOT NULL,
    protocol VARCHAR(20) NOT NULL DEFAULT 'HTTP', -- HTTP atau RTSP
    status VARCHAR(20) NOT NULL DEFAULT 'online', -- online atau offline
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- 4. TABEL RIWAYAT INSIDEN JATUH (INCIDENTS)
CREATE TABLE IF NOT EXISTS public.incidents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id UUID REFERENCES public.cameras(id) ON DELETE SET NULL,
    camera_name VARCHAR(255),
    location VARCHAR(255),
    person_name VARCHAR(255) DEFAULT 'Pak Hartono',
    detected_at TIMESTAMPTZ DEFAULT now(),
    confidence FLOAT DEFAULT 0.95,
    snapshot_url TEXT,
    clip_duration VARCHAR(50) DEFAULT '8 dtk',
    response_time_seconds FLOAT DEFAULT 2.4,
    status VARCHAR(50) DEFAULT 'unresolved', -- unresolved, acknowledged, resolved
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 5. TABEL PENGATURAN NOTIFIKASI (USER_SETTINGS)
CREATE TABLE IF NOT EXISTS public.user_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID UNIQUE REFERENCES public.users(id) ON DELETE CASCADE,
    popup_alert BOOLEAN DEFAULT true,
    sound_alarm BOOLEAN DEFAULT true,
    push_notification BOOLEAN DEFAULT true,
    daily_digest BOOLEAN DEFAULT false,
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- 6. ROW LEVEL SECURITY (RLS) & POLICIES
-- Mengaktifkan RLS agar aman dan menghilangkan peringatan di Supabase
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.cameras ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.incidents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_settings ENABLE ROW LEVEL SECURITY;

-- Policy: Mengizinkan akses untuk backend (Service Role & API Client)
CREATE POLICY "Allow full access for backend" ON public.users FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow full access for backend" ON public.cameras FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow full access for backend" ON public.incidents FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Allow full access for backend" ON public.user_settings FOR ALL USING (true) WITH CHECK (true);

-- 7. INDEXES UNTUK PERFORMA QUERY
CREATE INDEX IF NOT EXISTS idx_cameras_user_id ON public.cameras(user_id);
CREATE INDEX IF NOT EXISTS idx_incidents_detected_at ON public.incidents(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_incidents_camera_id ON public.incidents(camera_id);

-- ====================================================================
-- SEED DATA AWAL (SESUAI DENGAN DOKUMENTASI & DESAIN FRONTEND RUMI)
-- ====================================================================

-- Akun demo: andi.pratama@keluarga.id / password123
INSERT INTO public.users (id, name, email, hashed_password, role, avatar_initials)
VALUES (
    'a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d',
    'Andi Pratama',
    'andi.pratama@keluarga.id',
    '$2b$12$EXI5IEvOzDdsutOy7bSjeOKvTq2ulS9CAZkxQbIr814KP9u3besge',
    'Pendamping',
    'AP'
)
ON CONFLICT (email) DO NOTHING;

-- Settings demo
INSERT INTO public.user_settings (user_id, popup_alert, sound_alarm, push_notification, daily_digest)
VALUES (
    'a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d',
    true,
    true,
    true,
    false
)
ON CONFLICT (user_id) DO NOTHING;

-- Kamera demo
INSERT INTO public.cameras (id, user_id, name, location, stream_url, protocol, status)
VALUES 
(
    'c1111111-1111-1111-1111-111111111111',
    'a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d',
    'Ruang Tamu',
    'Ruang Tamu',
    'http://192.168.1.50:8080/video',
    'HTTP',
    'online'
),
(
    'c2222222-2222-2222-2222-222222222222',
    'a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d',
    'Garasi',
    'Garasi',
    'rtsp://admin:123456@192.168.1.53:554/stream1',
    'RTSP',
    'offline'
)
ON CONFLICT (id) DO NOTHING;

-- Riwayat insiden demo
INSERT INTO public.incidents (id, camera_id, camera_name, location, person_name, detected_at, confidence, snapshot_url, clip_duration, response_time_seconds, status)
VALUES 
(
    gen_random_uuid(),
    'c1111111-1111-1111-1111-111111111111',
    'Ruang Tamu · Cam 01',
    'Ruang Tamu',
    'Pak Hartono',
    now() - interval '2 hours',
    0.98,
    '/static/uploads/fall-snapshot.jpg',
    '8 dtk',
    2.1,
    'unresolved'
),
(
    gen_random_uuid(),
    NULL,
    'Kamar Utama · Cam 02',
    'Kamar Utama',
    'Pak Hartono',
    now() - interval '1 day',
    0.95,
    '/static/uploads/fall-snapshot.jpg',
    '12 dtk',
    2.4,
    'resolved'
),
(
    gen_random_uuid(),
    NULL,
    'Dapur · Cam 03',
    'Dapur',
    'Pak Hartono',
    now() - interval '3 days',
    0.92,
    '/static/uploads/fall-snapshot.jpg',
    '6 dtk',
    2.6,
    'resolved'
);
