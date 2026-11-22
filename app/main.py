import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.routes import auth, cameras, incidents, dashboard, model, ws

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend API untuk Sistem Deteksi Jatuh Real-time (Rumi) dengan Supabase & YOLOv8 Pose.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_origin_regex=r"^https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Mount folder static untuk menyajikan file snapshot bukti jatuh
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
os.makedirs(os.path.join(static_dir, "uploads"), exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Registrasi Router API
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(cameras.router, prefix=settings.API_V1_STR)
app.include_router(incidents.router, prefix=settings.API_V1_STR)
app.include_router(dashboard.router, prefix=settings.API_V1_STR)
app.include_router(model.router, prefix=settings.API_V1_STR)
app.include_router(ws.router)

# Endpoint langsung untuk menerima alert dari FallDetectionWorker (realtime_fall_detection.py)
app.post("/api/incident-alerts", tags=["Incidents"])(incidents.handle_incident_alert)


@app.get("/", tags=["Health Check"])
async def root():
    return {
        "status": "online",
        "app_name": settings.PROJECT_NAME,
        "docs_url": "/docs",
        "api_version": "v1"
    }
