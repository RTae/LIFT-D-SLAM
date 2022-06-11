import numpy as np
from pydantic import BaseSettings
from functools import lru_cache

class AppSettings(BaseSettings):
    CAMERA_FOCUS_LENGTH: int
    CAMERA_WIDTH_SIZE: int
    CAMERA_HEIGHT_SIZE: int

    class Config:
        env_file = ".env"
    
    def get_camera_scale(self):
        return np.array([ 
                         [self.CAMERA_FOCUS_LENGTH, 0, self.CAMERA_WIDTH_SIZE],
                         [0, self.CAMERA_FOCUS_LENGTH, self.CAMERA_HEIGHT_SIZE],
                         [0, 0, 1]
                        ])

@lru_cache()
def get_settings():
    return AppSettings()
