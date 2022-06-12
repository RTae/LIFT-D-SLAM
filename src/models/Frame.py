import numpy as np

class Frame:
  def __init__(self, desc_dict, image, count, fe: any):
    self.count = count
    self.count_inv = np.linalg.inv(self.count)
    self.pose = np.eye(4)
    self.h, self.w = image.shape[0:2]
    
    # Extract feature
    key_pts, self.descriptors = fe.extract_feature(image, convert_to_uint8=True)
    self.key_pts = fe.normalize(self.count_inv, key_pts)
    self.pts = [None]*len(self.key_pts)
    self.id = len(desc_dict.frames)