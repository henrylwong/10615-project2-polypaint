import torch
from torch import nn
import torchvision.transforms as T
import numpy as np
from polygon import Polygon

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Painting(nn.Module):
  def __init__(self, opt, num_polygons):
    super().__init__()
    self.num_polygons = num_polygons

    self.polygons = nn.ModuleList([Polygon(opt) for _ in range(num_polygons)])

  def get_optimizers(self, multiplier=1.0):
    scale = list()
    x_pos = list()
    y_pos = list()
    angle = list()
    color = list()
    
    for n, p in self.named_parameters():
      if "scale" in n.split('.')[-1]: scale.append(p)
      if "x_pos" in n.split('.')[-1]: x_pos.append(p)
      if "y_pos" in n.split('.')[-1]: y_pos.append(p)
      if "angle" in n.split('.')[-1]: angle.append(p)
      if "color" in n.split('.')[-1]: color.append(p)
    
    position_opt = torch.optim.Adam(x_pos + y_pos, lr=5e-3 * multiplier)
    rotation_opt = torch.optim.Adam(angle, lr=1e-2 * multiplier)
    color_opt = torch.optim.Adam(color, lr=5e-3 * multiplier)
    scale_opt = torch.optim.Adam(scale, lr=1e-2 * multiplier)

    return position_opt, rotation_opt, color_opt, scale_opt

  def forward(self, h, w):
    '''Creates canvas; draws and merges each polygon on canvas'''
    canvas = torch.ones((1, 4, h, w)).to(device)
    opacity_factor = 0.8

    poly_alphas = list()
    for poly in self.polygons:
      single_poly = poly(h, w)
      poly_alphas.append(single_poly[:,3:])
      canvas = canvas * (1 - single_poly[:, 3:] * opacity_factor) + single_poly[:, 3:] * opacity_factor * single_poly
    alphas = torch.cat(poly_alphas, dim=1)
    alphas = torch.sum(alphas, dim=1)

    return canvas, alphas

  def get_alpha(self, h, w):
    '''Get all alpha (idx 3) vals of all polygons'''
    poly_alphas = list()
    for poly in self.polygons:
      single_poly = poly(h, w)
      poly_alphas.append(single_poly[:, 3:])

    alphas = torch.cat(poly_alphas, dim=1)
    alphas, _ = torch.max(alphas, dim=1)
    return alphas

  def to_csv(self):
    ''' To csv string '''
    csv_str = ''
    for poly in self.polygons:
      x = str(poly.transformation.x_pos[0].detach().cpu().item())
      y = str(poly.transformation.y_pos[0].detach().cpu().item())
      a = str(poly.transformation.angle[0].detach().cpu().item())

      scale = str(poly.scale[0].detach().cpu().item())
      alpha = str(poly.alpha[0].detach().cpu().item())
      color = poly.color_transform.detach().cpu().numpy()

      csv_str += ','.join([x, y, a, scale, alpha, str(color[0]), str(color[1]), str(color[2])]) #  *[str(c) for c in color] 
      csv_str += '\n' 
    csv_str = csv_str[:-1] # remove training newline
    return csv_str

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