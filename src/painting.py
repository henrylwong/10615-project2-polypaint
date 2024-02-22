import torch
from torch import nn
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode 
bicubic = InterpolationMode.BICUBIC
import numpy as np
from brush_stroke import BrushStroke
from param2stroke import get_param2img

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Painting(nn.Module):
  def __init__(self, num_polygons, opt):
    super().__init__()
    self.num_polygons = num_polygons

    self.polygons = nn.ModuleList([Polygon(opt) for _ in range(num_polygons)])
    self.param2img = get_param2img(opt)

  def get_optimizers(self, multiplier=1.0):
    scale = list()
    x_pos = list()
    y_pos = list()
    angle = list()
    color = list()
    
    for n, p in self.named_parameters():
      if "x_pos" in n.split('.')[-1]: x_pos.append(p)
      if "y_pos" in n.split('.')[-1]: y_pos.append(p)
      if "angle" in n.split('.')[-1]: angle.append(p)
      if "color" in n.split('.')[-1]: color.append(p)
    
    position_opt = torch.optim.Adam(x_pos + y_pos, lr=5e-3 * multiplier)
    rotation_opt = torch.optim.Adam(angle, lr=1e-2 * multiplier)
    color_opt = torch.optim.Adam(color, lr=5e-3 * multiplier)
    scale_opt = torch.optim.Adam(scale, lr=1e-2 * multiplier)

  def forward(self, h, w):
    '''Creates canvas; draws and merges each polygon on canvas'''
    canvas = torch.ones((1, 4, h, w)).to(device)

    for poly in self.polygons:
      single_poly = poly(h, w, self.param2img)
      canvas = canvas * (1 - single_poly[:, 3:] * opacity_factor) + single_poly[:, 3:] * opacity_factor * single_poly

    return canvas

  def get_alpha(self, h, w):
    '''Get all alpha (idx 3) vals of all polygons'''
    poly_alphas = list()
    for poly in self.polygons:
      single_poly = poly(h, w)
      poly_alphas.append(single_poly[:, 3:])

    alphas = torch.cat(poly_alphas, dim=1)
    alphas, _ = torch.max(alphas, dim=1)
    return alphas

  def validate(self):
    ''' Make sure all polygons have valid params'''
    for poly in self.polygons:
      Polygon.make_valid(poly)

  def pop(self):
    '''Pop first polygon from plan'''
    if not len(self.polygons):
      return None

    poly = self.polygons[0]
    self.polygons = nn.ModuleList([self.polygons[i] for i in range(1, len(self.brush_strokes))])
    return poly

  def __len__(self):
    return len(self.polygons)