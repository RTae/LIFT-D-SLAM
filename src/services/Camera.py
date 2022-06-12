from src.utils.configs.app_settings import get_settings
from typing import Tuple
import numpy as np
import argparse
import cv2

class Camera:

    def __init__(self, argparse: argparse.ArgumentParser.parse_args) -> None:
        self.__video_path = argparse.video_path
        self.__cap = cv2.VideoCapture(self.__video_path)

    def __resize(self, frame: np.array, size: Tuple[int]=(get_settings().CAMERA_WIDTH_SIZE, get_settings().CAMERA_HEIGHT_SIZE)) -> np.array:
        """
        It resizes the frame to the size specified in the settings.
        
        :param frame: The frame to be resized
        :type frame: np.array
        :param size: The size of the image to be returned
        :type size: Tuple[int]
        :return: A numpy array of the frame.
        """
        return cv2.resize(frame, size)

    def is_open(self) -> bool:
        """
        > This function returns a boolean value that indicates whether the video capture object is open or
        not
        :return: A boolean value.
        """
        return self.__cap.isOpened()

    def __get_frame(self) -> np.array:
        """
        > It reads a frame from the video stream, resizes it, and returns it
        :return: A numpy array of the frame
        """
        ret, frame = self.__cap.read()
        
        if(not ret):
            return None
        
        return frame

    def get_total_frame(self) -> int:
        return self.__cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    def run(self) -> np.array:
        '''
        Main pipline
        '''

        frame = self.__get_frame()

        if frame is None:
            return None

        return self.__resize(frame)
    
