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

cos = torch.cos
sin = torch.sin
def rigid_body_transform(a, xt, yt, anchor_x, anchor_y):
    # a is the angle in radians, xt and yt are translation terms of pixels
    # anchor points are where to rotate around (usually the center of the image)
    # Blessed be Peter Schorn for the anchor point transform https://stackoverflow.com/a/71405577
    A = torch.zeros(1, 3, 3).to(a.device) # transformation matrix
    a = -1.*a
    A[0,0,0] = cos(a)
    A[0,0,1] = -sin(a)
    A[0,0,2] = anchor_x - anchor_x * cos(a) + anchor_y * sin(a) + xt#xt
    A[0,1,0] = sin(a)
    A[0,1,1] = cos(a)
    A[0,1,2] = anchor_y - anchor_x * sin(a) - anchor_y * cos(a) + yt#yt
    A[0,2,0] = 0
    A[0,2,1] = 0
    A[0,2,2] = 1
    return A
    
class RigidBodyTransformation(nn.Module):
    def __init__(self, a, xt, yt):
        super(RigidBodyTransformation, self).__init__()
        self.xt = nn.Parameter(torch.ones(1)*xt)
        self.yt = nn.Parameter(torch.ones(1)*yt)
        self.a = nn.Parameter(torch.ones(1)*a)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        anchor_x, anchor_y = w/2, h/2

        M = rigid_body_transform(self.a[0], self.xt[0]*(w/2), self.yt[0]*(h/2), anchor_x, anchor_y)
        with warnings.catch_warnings(): # suppress annoying torchgeometry warning
            warnings.simplefilter("ignore")
            return torchgeometry.warp_perspective(x, M, dsize=(h,w))

class Polygon(nn.Module):
  def __init__(self, opt, 
               num_sides=None, scale=None, 
               color=None, alpha=None, 
               angle=None, x_pos=None, y_pos=None):
    super().__init__()

    self.MIN_SIDES = opt.MIN_SIDES
    self.MAX_SIDES = opt.MAX_SIDES
    self.MIN_SCALE = opt.MIN_SCALES
    self.MAX_SCALE = opt.MAX_SCALES
    self.MAX_ALPHA = opt.MAX_ALPHA

    if num_sides == None:
      num_sides = torch.randint(low=self.MIN_SIDES, max=self.MAX_SIDES + 1)
    if scale == None:
      scale = (torch.rand(1) * self.MAX_SCALE).clamp(min=0.1)
    if color == None:
      color = (torch.rand(3).to(device) * 0.4) + 0.3
    if angle == None:
      angle = (torch.rand(1) * 2 - 1) * (math.pi / 6) # @henry: clamping angle for equilateral triangles
    if x_pos == None:
      x_pos = (torch.rand(1) * 2 - 1) 
    if y_pos == None:
      y_pos = (torch.rand(1) * 2 - 1) 
    if alpha == None:
      alpha = (torch.rand(1) * 2 - 1) * self.MAX_ALPHA
    
    self.transformation = RigidBodyTransformation(angle, x_pos, y_pos)
    self.num_sides = nn.Parameter(num_sides)
    self.scale = nn.Parameter(scale)
    self.alpha = nn.Paremter(alpha)
    self.color_transform = nn.Parameter(color)

  def forward(self, h, w):
    # Generate polygon
    poly = generate_poly(self.num_sides, self.scale, h, w).unsqueeze(0)

    # Apply x/y position and angle transformation
    x = self.transformation(poly)
    x = torch.cat([x, x, x, x], dim=1)

    # Add color
    x = torch.cat((x[:,:3]*0 + self.color_transform[None,:,None,None], x[:,3:]), dim=1)

    return x