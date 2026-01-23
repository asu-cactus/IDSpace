from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import numpy as np
import math
import json
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from utils import *
import sys
import time
from itertools import islice
import yaml
import copy


def optimization(params, fixed_args):
    testing = fixed_args['testing']
    fixed_args['testing'] = False
    def objective_function(**params):
        return evaluate_parameters(**params, **fixed_args)

    # Initialize Bayesian Optimization
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        verbose=2,
        #random_state=1,
    )
    
    # Define the evaluation function for the optimizer
    optimizer.set_gp_params(normalize_y=True)
    
    # Define the acquisition function
    utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)
    
    # Add the evaluation function to the optimizer
    # Perform the optimization
    optimizer.maximize(
        init_points=1,
        n_iter=1,
        acquisition_function=utility
    )

    # Get the best parameters
    best_params = optimizer.max['params']
    print("Best Parameters:", best_params)
    
    return optimizer
    # Evaluate the best parameters one more time to get the final SSIM and PSNR
    best_sv_pv = evaluate_parameters(**best_params, **fixed_args)
    print("Best Evaluation on validation data:", best_sv_pv)

    # Evaluate on test data
    if testing:
        fixed_args['testing'] = True
        best_sv_pv = evaluate_parameters(**best_params, **fixed_args)
        print("Best Evaluation on testing data:", best_sv_pv)
    return optimizer
def generate_pbounds(font_info, default_margin=10):
    pbounds = {}

    def add_bound(key, initial, margin, clamp=None):
        if initial is None:
            return  # skip undefined values
        if margin is None:
            margin = default_margin  # fallback

        lower = initial - margin
        upper = initial + margin

        if clamp:
            lower = max(clamp[0], lower)
            upper = min(clamp[1], upper)

        pbounds[key] = (lower, upper)

    for key, value in font_info.items():
        if isinstance(value, dict) and "initial" in value:
            # flat field like font_size or text_height
            add_bound(key, value["initial"], value["margin"])
        elif key == "text_color":
            for channel, channel_value in value.items():
                add_bound(channel, channel_value["initial"], channel_value["margin"], clamp=(0, 255))
        elif key == "text_position":
            for axis, axis_value in value.items():
                position_key = "xp" if axis == "x" else "yp"
                add_bound(position_key, axis_value["initial"], axis_value["margin"])

    return pbounds


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Template configuration.")

    parser.add_argument('--target_samples', type=int, default=10, help='Number of target samples to be used')
    parser.add_argument('--with_model', type=int, choices=[0, 1], default=0, help='Whether to use model guided method (1=yes, 0=no)')
    parser.add_argument('--lambda0', type=float, default=1, help='The fractions of similarity score')
    parser.add_argument('--lambda1', type=float, default=1, help='The fractions of consistency score')
    parser.add_argument('--candidate_models', nargs='*', default="resnet50", help='List of candidate model names if using model guided method')
    parser.add_argument('--config_info', type=str, default="data/configures/ALB_initial.yaml", help='Information about the segment and guided model')
    #parser.add_argument('--output_file', type=str, default="ALB_parameters.json", help='Information about the segment and guided model')

    # Parse the arguments
    args = parser.parse_args()

    # Access variables
    target_samples = args.target_samples
    with_model = args.with_model
    candidate_models = args.candidate_models if args.candidate_models else []
    if args.with_model and len(args.candidate_models) == 0:
        print("Error: You should provide model name if you use model guided method")

    #if os.path.exists(args.output_file):
    #    try:
    #        with open(args.output_file, 'r') as f:
    #            configures = json.load(f)
    #    except:
    #        configures = {}
    #else:
    #    configures = {}
    #configures['args'] = vars(args)
    #if 'segments' not in configures:
    #    configures['segments'] = {}

    # Example print (can be removed)
    print(f"Target Samples: {target_samples}")
    print(f"With Model: {with_model}")
    print(f"Candidate Models: {candidate_models}")
    
    print("candidate_models:", candidate_models)
    with open( args.config_info, 'r') as f:
        #confs = json.load(f)
        confs = yaml.safe_load(f)
    configures = copy.deepcopy(confs)
    configures['BO_setings'] = vars(args)
    #if 'segments' not in configures:
    configures['segments'] = {}


    area = confs['area']
    args.area = area
    #configures['area'] = area
    fonts_path = confs['text_fonts_path']
    args.fonts_path = fonts_path
    args.template_path = confs['template_path']
    #configures['fonts_path'] = fonts_path
    #font_list = search_fonts(area, segment, 2)
    font_files = []
    #with open("./google_fonts/label_index_new.json") as f:
    #    label_indexs = json.load(f)
    #for fl in font_list:
    #    font_files.append(os.path.join('./google_fonts/newfonts', label_indexs[str(fl)]))
    for fl in os.listdir(fonts_path):
        if fl.endswith('.ttf'):
            font_files.append(os.path.join(fonts_path, fl))
    confs['font_files'] = font_files

    val_datas = {} 
    with open(confs['training_samples']) as f:
        ts = json.load(f)
    for v in ts['val']:
        if v[1] == 0:
            name = v[0].split('/')[-1]
            if name.split('_')[0].upper() == area:
                val_datas[name] = v[0]
    first_N_items = dict(islice(val_datas.items(), target_samples))
    confs['val_data'] = first_N_items

    val_datas = {}
    for v in ts['test']:
        if v[1] == 0:
            name = v[0].split('/')[-1]
            if name.split('_')[0].upper() == area:
                val_datas[name] = v[0]
    confs['test_data'] = val_datas

    # Define the bounds for the parameters
    key_pairs = confs['key_pairs']
    segments = confs['segments']
    save_quality = []
    for segment, search_parameters in segments.items():
    #search_parameters = segments[segment]
        if 'tunable' not in search_parameters or search_parameters['tunable'] != True:
            configures['segments'][segment] = search_parameters
            continue
        print(f" ### BO search for {segment} start")
        since = time.time()                                                                                                                                                                                                                                                                                                   
        pbounds = generate_pbounds(search_parameters['font_info'])
        pbounds["stroke_width"] = (0, 1)
        pbounds["spacing"] = (-5, 5)
        pbounds["blur_ratio"] = (0.0, 1.0)
        pbounds["font_style_idx"] = (0, len(font_files) - 0.01)
        pbounds["save_quality"] = (60, 100)
        #pbounds["save_quality2"] = (60, 100)

        fixed_args = {
                'area': area,
                'segment': segment,
                'confs': confs,
                'testing': True,
                'candidate_models': candidate_models,
                'with_model': with_model,
                'lambda0': args.lambda0,
                'lambda1': args.lambda1
        }

        def objective_function(**params):
            return evaluate_parameters(**params, **fixed_args)

        #optimizer = optimization(pbounds = pbounds, segment_key = segment_key, confs = confs, testing = False, candidate_models = candidate_models, with_model = with_model, lambda0 = lambda0, lambda1 = lambda1)
        optimizer = optimization(pbounds, fixed_args)
        bps = optimizer.max['params']
        tmp = {}
        tmp['x_position'] = int(bps['xp'])
        tmp['y_position'] = int(bps['yp'])
        tmp['font_size'] = int(bps['font_size'])
        tmp['stroke_width'] = round(bps['stroke_width'])
        tmp['color_r'] = int(bps['r'])
        tmp['color_g'] = int(bps['g'])
        tmp['color_b'] = int(bps['b'])
        tmp['spacing'] = int(bps['spacing'])
        tmp['blur_ratio'] = bps['blur_ratio']
        tmp['font_style'] = font_files[int(bps['font_style_idx'])].split('/')[-1] 
        save_quality.append(int(bps['save_quality']))
        #tmp['save_quality'] = int(bps['save_quality'])

        configures['segments'][key_pairs[segment]] = {'type': 'text', 'font_info': tmp}
        time_elapsed = time.time() - since                                                                                                                                                                                                                                                                                    
        print('Bayesian complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))                                                                                                                                                                                                                         
    configures['save_quality'] = save_quality
    with open(confs['best_seting_file'], 'w') as f:
        json.dump(configures, f, indent = 4)

