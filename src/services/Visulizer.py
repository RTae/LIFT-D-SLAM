import numpy as np
import cv2

class Visulizer:

    def show(self, frame: np.array()) -> None:
        if (frame is None):
            return 
        
        cv2.imshow("Camera View", frame)    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return 
    