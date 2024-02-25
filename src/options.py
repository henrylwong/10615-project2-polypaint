import argparse 
import os
import math
import json

class Options(object):
    def __init__(self):
        self.initialized = False
        self.opt = {}

    def initialize(self, parser):
        # Main parameters
        parser.add_argument("--cache_dir", type=str, default='./cache', help='Where to store cached files.')
        parser.add_argument("--image_name", type=str, default='target.png')
        parser.add_argument("--poly_model", type=str, default='tri_train.pt')
        parser.add_argument("--final_model", type=str, default='final.pt')

        # Planning Parameters
        parser.add_argument('--max_height', default=360.0, type=int, help='height of painting')
        parser.add_argument('--max_width', default=640.0, type=int, help='width of painting')
        parser.add_argument('--num_polygons', type=int, default=400)
        parser.add_argument('--fill_weight', type=float, default=0.0, help="Encourage strokes to fill canvas.")

        # Optimization Parameters
        parser.add_argument('--objective', nargs='*', type=str, default="mse", help='text|style|clip_conv_loss|l2|clip_fc_loss')
        parser.add_argument('--objective_weight', nargs='*', type=float, default=1.0)
        parser.add_argument('--n_inits', type=int, default = 0)
        parser.add_argument('--intermediate_optim_iter', type=int, default = 40)
        parser.add_argument('--init_optim_iter', type=int, default=400)
        parser.add_argument('--optim_iter', type=int, default=400)
        parser.add_argument('--lr_multiplier', type=float, default=0.2)
        parser.add_argument('--num_augs', type=int, default=30)

        # Logging Parameters
        parser.add_argument("--tensorboard_dir", type=str, default='./painting_log', help='Where to write tensorboard log to.')
        parser.add_argument('--plan_gif_dir', type=str, default='../outputs/')
        parser.add_argument('--log_frequency', type=int, default=5, help="Log to TB after this many optim. iters.")
        parser.add_argument("--output_dir", type=str, default="../outputs/", help='Where to write output to.')
        
        parser.add_argument("--min_sides", type=int, default=3)
        parser.add_argument("--max_sides", type=int, default = 8)
        parser.add_argument("--min_scale", type=float, default=0.1)
        parser.add_argument("--max_scale", type=float, default = 1.0)
        parser.add_argument("--max_alpha", type=float, default=1.0)
        parser.add_argument("--max_angle", type=float, default = math.pi/6)

        return parser 

    def gather_options(self):
        if not self.initialized:
            self.parser = argparse.ArgumentParser(description="Poly Paint")
            self.parser = self.initialize(self.parser)

        self.opt = vars(self.parser.parse_args())
        self.opt["img_path"] = os.path.join(self.opt["cache_dir"], self.opt["image_name"])

    def __getattr__(self, attr_name):
        return self.opt[attr_name]
