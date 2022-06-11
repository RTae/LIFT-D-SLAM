import numpy as np
import cv2

class ORBFeature:
    def extract_feature(self, image: np.array):
        
        orb = cv2.ORB_create()
        pts = cv2.goodFeaturesToTrack(
                np.mean(image, axis=2).astype(np.uint8), 
                1000,
                qualityLevel=0.01,
                minDistance=7
            )
        key_pts = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
        key_pts, descriptors = orb.compute(image, key_pts)

        return np.array([(kp.pt[0], kp.pt[1]) for kp in key_pts]), descriptors

    def normalize(self, count_inv, pts):

        return np.dot(count_inv, np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1).T).T[:, 0:2]

    def denormalize(self, count, pt):

        ret = np.dot(count, np.array([pt[0], pt[1], 1.0]))
        ret /= ret[2]
        return int(round(ret[0])), int(round(ret[1]))