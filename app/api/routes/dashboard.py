from fastapi import APIRouter, Depends
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone, timedelta
from app.schemas.dashboard import DashboardStats, WeeklyActivityItem
from app.core.supabase import db
from app.api.deps import get_current_user

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

def calculate_weekly_distribution(incidents: List[Dict[str, Any]]) -> Tuple[List[WeeklyActivityItem], int]:
    """Menghitung jumlah insiden per hari dalam 7 hari terakhir (WIB) dan jumlah insiden hari ini."""
    wib_tz = timezone(timedelta(hours=7))
    now_wib = datetime.now(wib_tz)
    today_date = now_wib.date()

    # Urutan hari dari Senin sampai Minggu
    days_order = [
        ("Sen", 0),
        ("Sel", 1),
        ("Rab", 2),
        ("Kam", 3),
        ("Jum", 4),
        ("Sab", 5),
        ("Min", 6),
    ]
    day_counts = {code: 0 for code, _ in days_order}
    today_count = 0

    for inc in incidents:
        dt_raw = inc.get("detected_at")
        if not dt_raw:
            continue
        try:
            dt = datetime.fromisoformat(dt_raw.replace("Z", "+00:00")).astimezone(wib_tz)
            inc_date = dt.date()

            # Hitung insiden hari ini
            if inc_date == today_date:
                today_count += 1

            # Hitung jika dalam rentang 7 hari terakhir (0 s/d 6 hari yang lalu)
            diff_days = (today_date - inc_date).days
            if 0 <= diff_days <= 6:
                day_idx = dt.weekday()
                for code, idx in days_order:
                    if idx == day_idx:
                        day_counts[code] += 1
                        break
        except Exception:
            continue

    weekly_items = [
        WeeklyActivityItem(day=code, value=day_counts[code])
        for code, _ in days_order
    ]

    return weekly_items, today_count

@router.get("/summary", response_model=DashboardStats)
async def get_dashboard_summary(current_user: Dict[str, Any] = Depends(get_current_user)):
    cameras = await db.get_cameras(user_id=current_user["id"])
    camera_ids = [c["id"] for c in cameras]
    incidents = await db.get_incidents(camera_ids=camera_ids) if camera_ids else []

    active_count = len([c for c in cameras if c.get("status") == "online"])
    total_count = len(cameras)
    latest = incidents[0] if incidents else None

    # Hitung data riil mingguan dan insiden hari ini
    weekly, today_incidents = calculate_weekly_distribution(incidents)

    return DashboardStats(
        active_cameras=active_count,
        total_cameras=total_count,
        uptime_percentage=99.6,
        latency_seconds=2.4,
        today_incidents=today_incidents,
        latest_incident=latest,
        weekly_activity=weekly
    )
