import src.provider.visualizes as visualizes
import numpy as np
import cv2

class Visulizer:

    def __init__(self, display: visualizes.Display, three_display: visualizes.ThreeDViewer) -> None:
        self.display = display
        self.three_display = three_display

    def show(self, frame_o: np.array, frame_f: np.array) -> None:
        if (frame_o is None):
            return 
        
        # Camera view
        cv2.imshow("Camera View", frame_o)    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return 

        # Feature 2d view
        self.display.display_2d(frame_f)

        # 3d view cloud point
        self.threeDisplay.display()
