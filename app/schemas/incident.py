from pydantic import BaseModel
from typing import Optional

class IncidentCreate(BaseModel):
    camera_id: Optional[str] = None
    camera_name: str
    location: str
    person_name: str = "Pak Hartono"
    confidence: float = 0.95
    snapshot_url: Optional[str] = "/static/uploads/fall-snapshot.jpg"
    clip_duration: str = "8 dtk"
    status: str = "unresolved"

class IncidentResponse(BaseModel):
    id: str
    camera_id: Optional[str] = None
    camera_name: str
    location: str
    person_name: str
    detected_at: str
    confidence: float
    snapshot_url: Optional[str]
    clip_url: Optional[str] = None
    clip_duration: str
    response_time_seconds: float = 2.4
    status: str
