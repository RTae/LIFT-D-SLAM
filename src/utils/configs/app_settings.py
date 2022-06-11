from pydantic import BaseSettings
from functools import lru_cache

class AppSettings(BaseSettings):
    CAMERA_FOCUS_LENGTH: int
    CAMERA_WIDTH_SIZE: int
    CAMERA_HEIGHT_SIZE: int

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return AppSettings()
