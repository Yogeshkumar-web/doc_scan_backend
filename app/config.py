from pydantic_settings import BaseSettings
import json

class Settings(BaseSettings):
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_upload_size_mb: int = 15
    scan_max_dimension: int = 2200
    log_level: str = "INFO"

    @property
    def cors_origin_list(self) -> list[str]:
        raw_value = self.cors_origins.strip()
        if not raw_value:
            return []

        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return [origin.strip() for origin in raw_value.split(",") if origin.strip()]

        if isinstance(parsed, list):
            return [str(origin).strip() for origin in parsed if str(origin).strip()]

        if isinstance(parsed, str):
            return [origin.strip() for origin in parsed.split(",") if origin.strip()]

        return []

    class Config:
        env_file = ".env"

settings = Settings()
