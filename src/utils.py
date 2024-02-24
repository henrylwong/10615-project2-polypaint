from PIL import Image
import os
from torch import nn
import torch
import numpy as np
# import colour
# import cv2
import random
from torchvision import transforms

def randomize_polygon_order(painting):
    with torch.no_grad():
        polygons = [poly for poly in painting.polygons]
        random.shuffle(polygons)
        painting.polygons = nn.ModuleList(polygons)
        return painting

def image_to_tensor(image_path):
    if os.path.exists(image_path):
        image = Image.open(image_path)
        image_with_alpha = image.convert("RGBA")
        
        transform = transforms.Compose([transforms.ToTensor(),])
        tensor_image = transform(image_with_alpha)
        
        return tensor_image.unsqueeze(0)
    else:
        print(f"Image path: {image_path} is not valid")

def tensor_to_image(rgba_tensor, filename):
    rgba_image = Image.fromarray((rgba_tensor.squeeze(0).permute(1, 2, 0).mul(255).byte().numpy()), 'RGBA')
    rgba_image.save(filename)

# def sort_brush_strokes_by_color(painting, bin_size=3000):
#     with torch.no_grad():
#         poly = [p for p in painting.polygons]
#         for j in range(0,len(poly), bin_size):
#             poly[j:j+bin_size] = sorted(poly[j:j+bin_size], 
#                 key=lambda x : x.color_transform.mean()+x.color_transform.prod(), 
#                 reverse=True)
#         painting.polygons = nn.ModuleList(poly)
#         return painting
    
# def discretize_color(polygon, discrete_colors):
#     dc = discrete_colors.cpu().detach().numpy()
#     dc = dc[None,:,:]
#     dc = cv2.cvtColor(dc, cv2.COLOR_RGB2Lab)

#     with torch.no_grad():
#         color = polygon.color_transform.detach()
#         c = color[None,None,:].detach().cpu().numpy()
#         c = cv2.cvtColor(c, cv2.COLOR_RGB2Lab)

#         dist = colour.delta_E(dc, c)
#         argmin = np.argmin(dist)

#         return discrete_colors[argmin].clone()

# def discretize_colors(painting, discrete_colors):
#     with torch.no_grad():
#         for poly in painting.polygons:
#             new_color = discretize_color(poly, discrete_colors)
#             poly.color_transform.data *= 0
#             poly.color_transform.data += new_color