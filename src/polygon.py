import copy
import math
import torch
from torch import nn
import torchgeometry
import torchvision.transforms as T
import warnings
import os
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def rigid_body_transform(a, xt, yt, anchor_x, anchor_y):
    A = torch.zeros(1, 3, 3).to(a.device) # transformation matrix
    a = -1.*a
    A[0,0,0] = torch.cos(a)
    A[0,0,1] = -torch.sin(a)
    A[0,0,2] = anchor_x - anchor_x * torch.cos(a) + anchor_y * torch.sin(a) + xt#xt
    A[0,1,0] = torch.sin(a)
    A[0,1,1] = torch.cos(a)
    A[0,1,2] = anchor_y - anchor_x * torch.sin(a) - anchor_y * torch.cos(a) + yt#yt
    A[0,2,0] = 0
    A[0,2,1] = 0
    A[0,2,2] = 1
    return A
    
class RigidBodyTransformation(nn.Module):
    def __init__(self, a, xt, yt):
        super(RigidBodyTransformation, self).__init__()
        self.x_pos = nn.Parameter(torch.ones(1)*xt)
        self.y_pos = nn.Parameter(torch.ones(1)*yt)
        self.angle = nn.Parameter(torch.ones(1)*a)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        anchor_x, anchor_y = w/2, h/2

        M = rigid_body_transform(self.angle[0], self.x_pos[0]*(w/2), self.y_pos[0]*(h/2), anchor_x, anchor_y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torchgeometry.warp_perspective(x, M, dsize=(h,w))

class Polygon(nn.Module):
  def __init__(self, opt, 
               num_sides=None, scale=None, 
               color=None, alpha=None, 
               angle=None, x_pos=None, y_pos=None):
    super().__init__()

    self.opt = opt
    self.MIN_SIDES = opt.min_sides
    self.MAX_SIDES = opt.max_sides
    self.MIN_SCALE = opt.min_scale
    self.MAX_SCALE = opt.max_scale
    self.MAX_ALPHA = opt.max_alpha
    self.MAX_ANGLE = opt.max_angle # (math.pi / 6)

    # if num_sides == None:
    #   num_sides = torch.randint(low=self.MIN_SIDES, high=self.MAX_SIDES + 1, size=(1,))
    if scale == None:
      scale = torch.rand(1).clamp(min=0.01) # scale: (0.01, 1)
    if color == None:
      color = (torch.rand(3).to(device) * 0.4) + 0.3 # scale: (0.3, 0.7)
    if angle == None:
      angle = (torch.rand(1) * 2 - 1) * self.MAX_ANGLE # @henry: clamping angle for equilateral triangles
    if x_pos == None:
      x_pos = (torch.rand(1) * 2 - 1) 
    if y_pos == None:
      y_pos = (torch.rand(1) * 2 - 1) 
    if alpha == None:
      alpha = (torch.rand(1) * 2 - 1) * self.MAX_ALPHA

    self.transformation = RigidBodyTransformation(angle, x_pos, y_pos)
    # self.num_sides = nn.Parameter(num_sides)
    self.scale = nn.Parameter(scale)
    self.alpha = nn.Parameter(alpha)
    self.color_transform = nn.Parameter(color)

  def forward(self, h, w):
    param2img = PolyParamsToImage()
    param2img.load_state_dict(torch.load(os.path.join(self.opt.cache_dir, self.opt.poly_model), map_location=device))
    param2img.eval()
    param2img.to(device)

    # Generate polygon
    param = torch.tensor([self.scale]).unsqueeze(0)
    param = param.to(device)
    x = Polygon.pad_img(param, h, w, param2img).unsqueeze(1)

    # Apply x/y position and angle transformation
    x = self.transformation(x)
    x = torch.cat([x, x, x, x], dim=1)

    # Add color
    x = torch.cat((x[:,:3]*0 + self.color_transform[None,:,None,None], x[:,3:]), dim=1)

    return x
  
  def pad_img(param, h, w, param2img):
    img = param2img(param)
    pad_top = (h - img.shape[1]) // 2
    pad_bot = pad_top
    pad_left = (w - img.shape[2]) // 2
    pad_right = pad_left
    pad_for_full = T.Pad((pad_left, pad_top, pad_right, pad_bot))
    return pad_for_full(img)

  def make_valid(poly):
    with torch.no_grad():
      # Poly attributes
      poly.scale.data.clamp_(poly.MIN_SCALE, poly.MAX_SCALE)

      # RigidBodyTransformation Attributes
      poly.transformation.x_pos.data.clamp_(-1., 1.)
      poly.transformation.y_pos.data.clamp_(-1., 1.)
      poly.transformation.angle.data.clamp_(-1. * poly.MAX_ANGLE, 1. * poly.MAX_ANGLE)

      # Color Attributes
      poly.color_transform.data.clamp_(0.02,0.85)

class PolyParamsToImage(nn.Module):
  def __init__(self):
    super().__init__()
    self.nh = 20
    self.nc = 20
    self.size_x = 128
    self.size_y = 64
    self.main = nn.Sequential(
        nn.BatchNorm1d(1),
        nn.Linear(1, self.nh),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm1d(self.nh),
        nn.Linear(self.nh, self.size_x*self.size_y),
        nn.LeakyReLU(0.2, inplace=True)
    )
    self.conv = nn.Sequential(
        nn.BatchNorm2d(1),
        nn.Conv2d(1, self.nc, kernel_size=5, padding='same', dilation=1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.BatchNorm2d(self.nc),
        nn.Conv2d(self.nc, 1, kernel_size=5, padding='same', dilation=1),
        nn.Sigmoid()
    )

  def forward(self, x):
    x = self.main(x)
    x = x.view(-1, 1, self.size_y, self.size_x)
    x = self.conv(x)[:,0]
    return x
