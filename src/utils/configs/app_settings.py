import numpy as np
from pydantic import BaseSettings
from functools import lru_cache

class AppSettings(BaseSettings):
    CAMERA_FOCUS_LENGTH: int
    CAMERA_WIDTH_SIZE: int
    CAMERA_HEIGHT_SIZE: int
    CAMERA_SCALE: np.arary = np.array([ 
                                        [CAMERA_FOCUS_LENGTH, 0, CAMERA_WIDTH_SIZE],
                                        [0, CAMERA_FOCUS_LENGTH, CAMERA_HEIGHT_SIZE],
                                        [0, 0, 1]
                                     ])

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return AppSettings()
