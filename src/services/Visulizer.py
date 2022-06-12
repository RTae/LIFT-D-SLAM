import src.provider.visualizes as visualizes
import numpy as np
import cv2

class Visulizer:

    def __init__(self, three_display: visualizes.ThreeDViewer) -> None:
        self.three_display = three_display

    def show(self, frame_o: np.array, frame_f: np.array) -> None:
        if (frame_o is None or frame_f is None):
            return 
        
        # Camera view
        cv2.imshow("Camera View", frame_f)    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return 

        # 3d view cloud point
        self.three_display.display()
