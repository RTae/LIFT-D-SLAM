from src.utils.configs.app_settings import get_settings
from src.provider.feature.ORBFeature import ORBFeature
import src.provider.visualizes as visualizes
import src.models as models
from typing import Any
import numpy as np

class Tracking:
    
    def run(self, desc_dict: visualizes.ThreeDViewer, image: np.array, fe: ORBFeature, count: Any= get_settings().get_camera_scale()) -> models.Frame:
        '''
        Main pipline
        '''

        if image is None:
            return 
            
        frame = models.Frame(desc_dict, image, count, fe)
        desc_dict.frames.append(frame)

        return frame