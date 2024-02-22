import glob
import json
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode 
bicubic = InterpolationMode.BICUBIC
from torch import nn

import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import copy
import gzip
from torchvision.transforms.functional import affine

def get_param2img(opt, device='cuda'):
    
    w_canvas_m = opt.CANVAS_WIDTH_M
    h_canvas_m = opt.CANVAS_HEIGHT_M
    
    # Load how many meters the param2image model output represents
    with open(os.path.join(opt.cache_dir, 'param2stroke_settings.json'), 'r') as f:
        settings = json.load(f)
        print('Param2Stroke Settings:', settings)
        w_p2i_m = settings['w_p2i_m']
        h_p2i_m = settings['h_p2i_m']
        xtra_room_horz_m = settings['xtra_room_horz_m']
        xtra_room_vert_m = settings['xtra_room_vert_m']
        MAX_BEND = settings['MAX_BEND']
    
    # print(w_p2i_m - xtra_room_horz_m, 0.5*w_canvas_m)
    if (w_p2i_m- xtra_room_horz_m) > (0.5 * w_canvas_m):
        raise Exception("The canvas width is less than two times the max_stroke_length. This makes it really too hard to render. Must use larger canvas.")
    
    param2img = StrokeParametersToImage()
    param2img.load_state_dict(torch.load(
        os.path.join(opt.cache_dir, 'param2img.pt')))
    param2img.eval()
    param2img.to(device)

    def forward(param, h_render_pix, w_render_pix):
        # Figure out what to resize the output of param2image should be based on the desired render size
        w_p2i_render_pix = int((w_p2i_m / w_canvas_m) * w_render_pix)
        h_p2i_render_pix = int((h_p2i_m / h_canvas_m) * h_render_pix)
        res_to_render = transforms.Resize((h_p2i_render_pix, w_p2i_render_pix), bicubic, antialias=True)

        # Pad the output of param2image such that the start of the stroke is directly in the
        # middle of the canvas and the dimensions of the image match the render size
        pad_left_m = 0.5 * w_canvas_m - xtra_room_horz_m
        pad_right_m = w_canvas_m - pad_left_m - w_p2i_m
        pad_top_m = 0.5 * h_canvas_m - MAX_BEND - xtra_room_vert_m
        pad_bottom_m = 0.5 * h_canvas_m - MAX_BEND - xtra_room_vert_m

        pad_left_pix =   int(pad_left_m   * (w_render_pix / w_canvas_m))
        pad_right_pix =  int(pad_right_m  * (w_render_pix / w_canvas_m))
        pad_top_pix =    int(pad_top_m    * (h_render_pix / h_canvas_m))
        pad_bottom_pix = int(pad_bottom_m * (h_render_pix / h_canvas_m))

        pad_for_full = transforms.Pad((pad_left_pix, pad_top_pix, pad_right_pix, pad_bottom_pix))

        return pad_for_full(res_to_render(param2img(param)))
    return forward#param2img#param2imgs, resize