import pygame
from pygame.locals import DOUBLEBUF

class Display(object):

  def __init__(self, w, h):
    pygame.init()
    self.screen = pygame.display.set_mode((w, h), DOUBLEBUF)
    self.surface = pygame.Surface(self.screen.get_size()).convert()

  def display_2d(self, img):
    pygame.surfarray.blit_array(self.surface, img.swapaxes(0,1)[:, :, [0,1,2]])
    self.screen.blit(self.surface, (0,0))
    pygame.display.flip()
