#!/usr/bin/env python3

import numpy as np
import torch
from tqdm import tqdm
import os
from torchvision.utils import save_image
import time
import copy
from utils import *
from painting import *
from options import Options

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_objective_function(objective_type, target, p, weight=1.0):
    if objective_type == "mse":
        return ((p - target)**2).mean() * weight
    raise NotImplementedError()

def plan(opt, n_strokes, h, w):
    painting = Painting(opt, n_strokes).to(device)
    target_img = image_to_tensor(opt.img_path)
    target_img = target_img.to(device)
    
    # # Getting best initial copy
    # init_painting = copy.deepcopy(painting.cpu())
    # best_init_painting, best_init_loss = init_painting, sys.maxsize

    # for attempt in range(opt.n_inits):
    #     painting = copy.deepcopy(init_painting).to(device)
    #     optims = painting.get_optimizers(multiplier=opt.lr_multiplier)
    #     for j in tqdm(range(opt.intermediate_optim_iter), desc="Intermediate Optimization"):
    #         for o in optims: o.zero_grad() if o is not None else None
    #         p, alphas = painting(h, w)
    #         loss = 0
    #         loss = get_objective_function(opt.objective, target_img, p[:,:3])
    #         loss += (1 - alphas).mean() * opt.fill_weight

    #         loss.backward()
    #         for o in optims: o.step() if o is not None else None
    #         painting.validate()

    #     if loss.item() < best_init_loss:
    #         best_init_loss = loss.item()
    #         best_init_painting = copy.deepcopy(painting.cpu())
    #         painting.to(device)
    #         print('Best_painting = {}'.format(attempt))
    # painting = best_init_painting.to(device)

    # Optimizing best initialization
    position_opt, rotation_opt, color_opt, scale_opt = painting.get_optimizers(multiplier=opt.lr_multiplier)
    optims = (position_opt, rotation_opt, color_opt, scale_opt)

    for i in tqdm(range(opt.optim_iter), desc='Optimizing {} Polygons'.format(str(len(painting.polygons)))): 
        for o in optims: o.zero_grad() if o is not None else None

        p, alphas = painting(h, w) 
        loss = get_objective_function(opt.objective, target_img[:, :3], p[:,:3])
        loss += (1 - alphas).mean() * opt.fill_weight
        loss.backward()

        for o in optims: o.step() if o is not None else None
        if i < .8*opt.optim_iter: color_opt.step() if color_opt is not None else None

        painting.validate()

        if i < 0.3 * opt.optim_iter and i % 3 == 0:
            painting = randomize_polygon_order(painting)

        print(f"Epoch {i + 1}: Loss = {loss}")
        if i % 10 == 0:
          tensor_to_image(p.cpu(), os.path.join(opt.cache_dir, f"img/epoch[{i}]_time[{time.time()}].png"))
        if i % 50 == 0:
          torch.save(painting.state_dict(), os.path.join(opt.cache_dir, opt.final_model))
    return painting

if __name__ == '__main__':
    global opt
    opt = Options()
    opt.gather_options()
    
    global h, w, colors, current_canvas, text_features, style_img, sketch
    w = int(opt.max_width)
    h = int(opt.max_height)
    
    painting = plan(opt, opt.num_polygons, h, w)
    
    with torch.no_grad():
        # save_image(painting(h,w), os.path.join(opt.cache_dir, 'init_painting_plan{}.png'.format(str(time.time()))))
        tensor_to_image(painting(h, w)[0].cpu(), os.path.join(opt.cache_dir, f"img/final_time[{time.time()}].png")) 
    
    f = open(os.path.join(opt.cache_dir, "poly_order.csv"), "w")
    f.write(painting.to_csv())
    f.close()
