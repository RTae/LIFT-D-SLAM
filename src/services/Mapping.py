from skimage.transform import FundamentalMatrixTransform
from src.utils.configs.app_settings import get_settings
from src.provider import visualizes, feature
from skimage.measure import ransac
import src.models as models
from typing import Any
import numpy as np
import logging
import cv2

np.set_printoptions(suppress=True)

class Point():
  def __init__(self, mapp, loc):
    self.pt = loc
    self.frames = []
    self.idxs = []
    
    self.id = len(mapp.points)
    mapp.points.append(self)

  def add_observation(self, frame, idx):
    frame.pts[idx] = self
    self.frames.append(frame)
    self.idxs.append(idx)

class Mapping:

  def __init__(self, fe: feature.ORBFeature, camera_scale: Any=get_settings().get_camera_scale()) -> None:
    self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    self.camera_scale = camera_scale
    self.fe = fe

  def __extract_rt(self, feature):
    """
    It takes a 3x3 matrix and returns a 4x4 matrix
    
    :param feature: the feature matrix, which is a 3x3 matrix
    :return: The rotation matrix and translation vector.
    """
    W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
    U, _, vt = np.linalg.svd(feature)
    if np.linalg.det(vt) < 0:
        vt *= -1.0
    R = np.dot(np.dot(U, W), vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), vt)
    t = U[:, 2]
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret
  
  def __matching(self, frame_1: models.Frame, frame_2: models.Frame):
    """
    It takes two frames, finds the matches between them, and then uses RANSAC to find the fundamental
    matrix between them
    
    :param frame_1: the first frame
    :type frame_1: models.Frame
    :param frame_2: the current frame
    :type frame_2: models.Frame
    :return: The indices of the keypoints that are inliers, the rotation and translation matrix
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(frame_1.descriptors, frame_2.descriptors, k=2)

    # Lowe's ratio test
    ret = []
    x1,x2 = [], []

    for m,n in matches:
        if m.distance < 0.75*n.distance:
            pts1 = frame_1.key_pts[m.queryIdx]
            pts2 = frame_2.key_pts[m.trainIdx]

            # travel less than 10% of diagonal and be within orb distance 32
            if np.linalg.norm((pts1-pts2)) < 0.1*np.linalg.norm([frame_1.w, frame_1.h]) and m.distance < 32:
                # keep around indices
                # TODO: refactor this to not be O(N^2)
                if m.queryIdx not in x1 and m.trainIdx not in x2:
                    x1.append(m.queryIdx)
                    x2.append(m.trainIdx)

                    ret.append((pts1, pts2))

    # no duplicates
    assert(len(set(x1)) == len(x1))
    assert(len(set(x2)) == len(x2))

    assert len(ret) >= 8
    ret = np.array(ret)
    x1 = np.array(x1)
    x2 = np.array(x2)

    # fit matrix
    model, f_pts = ransac((ret[:, 0], ret[:, 1]),
                            FundamentalMatrixTransform,
                            min_samples=8,
                            residual_threshold=0.001,
                            max_trials=100)
    
    logging.info("Matches: %d -> %d -> %d -> %d" % (len(frame_1.descriptors), len(matches), len(f_pts), sum(f_pts)))

    # ignore outliers
    rt = self.__extract_rt(model.params)

    # return
    return x1[f_pts], x2[f_pts], rt

  def __triangulate(self, pose1, pose2, pts1, pts2):
  
    ret = np.zeros((pts1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)

    for i, p in enumerate(zip(pts1, pts2)):
      A = np.zeros((4,4))
      A[0] = p[0][0] * pose1[2] - pose1[0]
      A[1] = p[0][1] * pose1[2] - pose1[1]
      A[2] = p[1][0] * pose2[2] - pose2[0]
      A[3] = p[1][1] * pose2[2] - pose2[1]
      _, _, vt = np.linalg.svd(A)
      ret[i] = vt[3]

    return ret
  
  def __tranform_feature(self, frame_o: np.array, three_display: visualizes.ThreeDViewer, frame_1, frame_2, x1, x2, rt):

    frame_1.pose = np.dot(rt, frame_2.pose)
    frame_f = frame_o.copy()

    for i,idx in enumerate(x2):
      if frame_2.pts[idx] is not None:
        frame_2.pts[idx].add_observation(frame_1,x1[i])

    # homogeneous 3-D coords
    pts4d = self.__triangulate(frame_1.pose, frame_2.pose, frame_1.key_pts[x1], frame_2.key_pts[x2])
    pts4d /= pts4d[:, 3:]
    unmatched_points = np.array([frame_1.pts[i] is None for i in x1])

    logging.info("Adding:  %d points" % np.sum(unmatched_points))
    
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0) & unmatched_points

    for i,p in enumerate(pts4d):
      if not good_pts4d[i]:
          continue

      pt = Point(three_display, p)
      pt.add_observation(frame_1, x1[i])
      pt.add_observation(frame_2, x2[i])

    for pt1, pt2 in zip(frame_1.key_pts[x1], frame_2.key_pts[x2]):
      u1, v1 = self.fe.denormalize(self.camera_scale, pt1)
      u2, v2 = self.fe.denormalize(self.camera_scale, pt2)
      cv2.circle(frame_f, (u1, v1), color=(0,255,0), radius=1)
      cv2.line(frame_f, (u1, v1), (u2, v2), color=(255, 255,0))
    
    return frame_f

  def run(self, frame: np.array, frame_data: models.Frame, three_display: visualizes.ThreeDViewer):
      '''
      Main pipline
      '''
      if frame is None:
          return

      if frame_data.id == 0:
          return
      
      frame_1 = three_display.frames[-1]
      frame_2 = three_display.frames[-2]

      x1, x2, rt = self.__matching(frame_1, frame_2)
      return self.__tranform_feature(frame, three_display, frame_1, frame_2, x1, x2, rt)