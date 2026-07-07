from pydantic_settings import BaseSettings
from typing import List
import json

class Settings(BaseSettings):
    cors_origins: List[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    max_upload_size_mb: int = 15
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        # Allow parsing list from comma-separated string if provided via env
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str):
            if field_name == "cors_origins":
                try:
                    return json.loads(raw_val)
                except Exception:
                    return [x.strip() for x in raw_val.split(",") if x.strip()]
            return cls.json_loads(raw_val)

settings = Settings()
