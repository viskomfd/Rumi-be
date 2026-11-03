from fastapi import APIRouter, Depends
from typing import Optional, Dict, Any
from app.schemas.dashboard import DashboardStats, WeeklyActivityItem
from app.core.supabase import db
from app.api.deps import get_optional_current_user

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get("/summary", response_model=DashboardStats)
async def get_dashboard_summary(current_user: Optional[Dict[str, Any]] = Depends(get_optional_current_user)):
    user_id = current_user["id"] if current_user else None
    cameras = await db.get_cameras(user_id=user_id)
    incidents = await db.get_incidents()

    active_count = len([c for c in cameras if c.get("status") == "online"])
    total_count = len(cameras)
    latest = incidents[0] if incidents else None

    # Sample weekly distribution matching design
    weekly = [
        WeeklyActivityItem(day="Sen", value=0),
        WeeklyActivityItem(day="Sel", value=1),
        WeeklyActivityItem(day="Rab", value=0),
        WeeklyActivityItem(day="Kam", value=2),
        WeeklyActivityItem(day="Jum", value=0),
        WeeklyActivityItem(day="Sab", value=1),
        WeeklyActivityItem(day="Min", value=0),
    ]

    return DashboardStats(
        active_cameras=active_count,
        total_cameras=total_count,
        uptime_percentage=99.6,
        latency_seconds=2.4,
        today_incidents=1,
        latest_incident=latest,
        weekly_activity=weekly
    )
