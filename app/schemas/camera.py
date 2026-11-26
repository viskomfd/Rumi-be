from pydantic import BaseModel, Field
from typing import Optional, Literal

class CameraCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    location: str = Field(..., min_length=1, max_length=100)
    url: str = Field(..., description="Stream URL (HTTP, HTTPS, RTSP, or device://)")

class CameraUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    location: Optional[str] = Field(None, min_length=1, max_length=100)
    url: Optional[str] = Field(None, min_length=1, description="Stream URL (HTTP, HTTPS, RTSP, or device://)")

class CameraResponse(BaseModel):
    id: str
    name: str
    location: str
    url: str
    protocol: Literal["HTTP", "RTSP", "DEVICE"]
    status: Literal["online", "offline"] = "online"
    created_at: Optional[str] = None
