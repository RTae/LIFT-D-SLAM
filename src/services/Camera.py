from src.utils.configs.app_settings import get_settings
from typing import Tuple
import numpy as np
import argparse
import cv2

class Camera:

    def __init__(self, argparse: argparse.ArgumentParser.parse_args) -> None:
        self.__video_path = argparse.video_path
        self.__cap = cv2.VideoCapture(self.__video_path)

    def __resize(self, frame: np.array(), size: Tuple[int]=(get_settings().CAMERA_WIDTH_SIZE, get_settings().CAMERA_HEIGHT_SIZE)) -> np.array:
        return cv2.resize(frame, size)

    def is_open(self) -> bool:
        return self.__cap.isOpen()

    def get_frame(self) -> np.array():
        ret, frame = self.__cap.read()
        
        if(not ret):
            return None
        
        return self.__resize(frame)
