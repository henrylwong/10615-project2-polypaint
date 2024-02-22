import copy
import math
import torch
from torch import nn
import torchgeometry
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode 
bicubic = InterpolationMode.BICUBIC
import warnings
import numpy as np

from param2stroke import special_sigmoid

class Polygon(nn.Module):
  def __init__(self, opt, 
               num_sides=None, scale=None, 
               color=None, alpha=None, 
               angle=None, xt=None, yt=None):
    super().__init__()

    self.MIN_SIDES = opt.MIN_SIDES
    self.MAX_SIDES = opt.MAX_SIDES
    self.MAX_ALPHA = opt.MAX_ALPHA

    if num_sides == None:
      pass
    if scale == None:
      pass
    if color == None:
      color = (torch.rand(3).to(device) * 0.4) + 0.3
    if angle == None:
      angle = (torch.rand(1) * 2 - 1) * math.pi
    if xt == None:
      xt = (torch.rand(1) * 2 - 1) 
    if yt == None:
      yt = (torch.rand(1) * 2 - 1) 

    self.transformation = RigidBodyTransformation(angle, xt, yt)

    self.poly_alpha = alpha
    self.color_transform = nn.Parameter(color)