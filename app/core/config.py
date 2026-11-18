import os
from typing import List, Union
# pyrefly: ignore[missing-import]
from pydantic_settings import BaseSettings
# pyrefly: ignore[missing-import]
from pydantic import field_validator

class Settings(BaseSettings):
    PROJECT_NAME: str = "Rumi Fall Detection API"
    API_V1_STR: str = "/api"
    SECRET_KEY: str = "rumi_secret_key_change_me_in_production_12345"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # Supabase Configuration
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""

    # Model Configuration
    MODEL_PATH: str = "models_ai/yolo11_rumi.pt"

    # Internal Worker Token
    INTERNAL_SERVICE_TOKEN: str = "rumi_internal_worker_secret_2026"

    # CORS Configuration
    CORS_ORIGINS: Union[str, List[str]] = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

settings = Settings()
