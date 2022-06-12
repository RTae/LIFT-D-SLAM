from multiprocessing import Process, Queue
import OpenGL.GL as gl
import numpy as np
import pypangolin as pangolin

class ThreeDViewer():
  def __init__(self):
    self.frames = []
    self.points = []
    self.state = None
    self.q = None

    self.__on_init()

  def __on_init(self):
    """
    It creates a new process that runs the function `viewer_thread` in parallel with the main process
    """
    self.q = Queue()
    self.vp = Process(
              target=self.viewer_thread, 
              args=(self.q,)
            )
    self.vp.daemon = True
    self.vp.start()

  def viewer_thread(self, q):
    """
    It creates a new thread that runs the viewer_init and viewer_refresh functions
    
    :param q: the queue that the viewer thread will read from
    """
    self.viewer_init(1024, 768)
    while 1:
      self.viewer_refresh(q)

  def viewer_init(self, w, h):
    """
    We create a window, enable depth testing, create a camera, and create a display
    
    :param w: width of the window
    :param h: height of the window
    """
    pangolin.CreateWindowAndBind('Main', w, h)
    gl.glEnable(gl.GL_DEPTH_TEST)

    self.scam = pangolin.OpenGlRenderState(
      pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
      pangolin.ModelViewLookAt(0, -10, -8,
                               0, 0, 0,
                               0, -1, 0))
    self.handler = pangolin.Handler3D(self.scam)

    # Create Interactive View in window
    self.dcam = pangolin.CreateDisplay()
    self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -w/h)
    self.dcam.SetHandler(self.handler)

  def viewer_refresh(self, q):
    """
    It takes a queue of data, and if the queue is not empty, it gets the data from the queue and stores
    it in the state variable. Then, it clears the screen, activates the camera, draws the keypoints and
    poses, and finishes the frame
    
    :param q: a queue that contains the current state of the SLAM system
    """
    if self.state is None or not q.empty():
      self.state = q.get()

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(0, 0, 0, 0)
    self.dcam.Activate(self.scam)

    # draw keypoints
    gl.glPointSize(2)
    gl.glColor3f(0.184314, 0.309804, 0.184314)
    pangolin.DrawPoints(self.state[1]+1)
    gl.glPointSize(1)
    gl.glColor3f(0.3099, 0.3099,0.184314)
    pangolin.DrawPoints(self.state[1])

    # draw poses
    gl.glColor3f(0.0, 1.0, 1.0)
    pangolin.DrawCameras(self.state[0])

    pangolin.FinishFrame()

  def display(self):
    """
    It takes a list of frames and a list of points, and puts them into a queue
    :return: The poses and points of the frames and points in the map.
    """
    if self.q is None:
      return

    poses, pts = [], []
    for f in self.frames:
      poses.append(f.pose)

    for p in self.points:
      pts.append(p.pt)

    self.q.put((np.array(poses), np.array(pts)))