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

# NEW: Optuna-based optimizer (drop-in for your current `optimization`)
import optuna
from types import SimpleNamespace

def optimization(pbounds, fixed_args, bo_settings, font_files=None):
    """
    Optuna optimization. Keeps return API compatible with bayes_opt by
    returning an object with `.max['params']`.
    """
    testing = fixed_args.get('testing', False)
    fixed_args['testing'] = False  # evaluate on validation during search

    # heuristics: which params should be integers vs floats
    int_like_keys = {
        'xp','yp','r','g','b','spacing','stroke_width','save_quality','text_height'
    }
    # If you want text_height to be float, remove it from int_like_keys.

    # Trials count: map your BO config to Optuna
    seed = int(bo_settings.get('seed', 1))
    init_points = int(bo_settings.get('init_points', 20))
    n_iter = int(bo_settings.get('n_iter', 50))
    n_trials = init_points + n_iter

    # Sampler & pruner
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=init_points)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, init_points // 2))

    def suggest_param(trial, name, low, high):
        # inclusive integer bounds; floats use suggest_float
        if name.split('#')[-1] in int_like_keys:
            lo = int(math.floor(low))
            hi = int(math.ceil(high))
            return trial.suggest_int(name, lo, hi)
        else:
            return trial.suggest_float(name, float(low), float(high))

    def objective(trial: optuna.Trial):
        # Build parameter dict from pbounds
        params = {}
        for k, (lo, hi) in pbounds.items():
            if k.split('#')[-1] == "font_style_idx":
                # Use categorical over indices to keep the rest of your code unchanged
                assert font_files is not None and len(font_files) > 0, \
                    "font_files must be provided when optimizing font_style_idx"
                params[k] = trial.suggest_categorical(k, list(range(len(font_files))))
            else:
                params[k] = suggest_param(trial, k, lo, hi)

        # Evaluate with your existing function
        return evaluate_parameters_global(**params, **fixed_args)

    # Create & run study (maximize your metric)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_trial.params
    print("Best Parameters:", best_params)

    # Optional: one more eval on validation to mirror your BO code path
    _ = evaluate_parameters_global(**best_params, **fixed_args)

    # Optional testing pass (kept for parity)
    if testing:
        fixed_args['testing'] = True
        best_sv_pv = evaluate_parameters_global(**best_params, **fixed_args)
        print("Best Evaluation on testing data:", best_sv_pv)

    # Return a BO-like object so the rest of your script (bps = optimizer.max['params']) still works
    return SimpleNamespace(max={'params': best_params}, study=study)


def optimization_BO(params, fixed_args, bo_settings):
    testing = fixed_args['testing']
    fixed_args['testing'] = False
    def objective_function(**params):
        return evaluate_parameters_global(**params, **fixed_args)

    # Initialize Bayesian Optimization
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        verbose=2,
        random_state= int(bo_settings.get('seed', 1)),
    )
    
    # Define the evaluation function for the optimizer
    optimizer.set_gp_params(normalize_y=True)
    
    # Define the acquisition function
    utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)
    
    # Add the evaluation function to the optimizer
    # Perform the optimization
    #print(params)
    optimizer.maximize(
        init_points= int(bo_settings.get('init_points', 20)),
        n_iter= int(bo_settings.get('n_iter', 50)),
        acquisition_function=utility
    )

    # Get the best parameters
    best_params = optimizer.max['params']
    print("Best Parameters:", best_params)
    
    return optimizer
    # Evaluate the best parameters one more time to get the final SSIM and PSNR
    best_sv_pv = evaluate_parameters_global(**best_params, **fixed_args)
    print("Best Evaluation on validation data:", best_sv_pv)

    # Evaluate on test data
    if testing:
        fixed_args['testing'] = True
        best_sv_pv = evaluate_parameters_global(**best_params, **fixed_args)
        print("Best Evaluation on testing data:", best_sv_pv)
    return optimizer

def get_bound(initial, margin, clamp=None):
    if initial is None:
        return  # skip undefined values
    if margin is None:
        margin = default_margin  # fallback

    lower = initial - margin
    upper = initial + margin

    if clamp:
        lower = max(clamp[0], lower)
        upper = min(clamp[1], upper)

    #print(f"<<< add bound -> key: {key}, lower: {lower}, upper: {upper}")
    return (lower, upper)

def generate_pbounds(font_info, prefix, default_margin=10):
    pbounds = {}


    for key, value in font_info.items():
        if isinstance(value, dict) and "initial" in value:
            # flat field like font_size or text_height
            pbounds[prefix + key] = get_bound(value["initial"], value["margin"])
        elif key == "text_color":
            for channel, channel_value in value.items():
                pbounds[prefix + channel] = get_bound(channel_value["initial"], channel_value["margin"], clamp=(0, 255))
        elif key == "text_position":
            for axis, axis_value in value.items():
                position_key = "xp" if axis == "x" else "yp"
                pbounds[prefix + position_key] = get_bound(axis_value["initial"], axis_value["margin"])

    return pbounds


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Template configuration.")
    parser.add_argument("--config", type=str, default="template_config.yaml",
                        help="Path to YAML with bo_settings, eval_args, pbounds, output.best_settings_yaml")

    #parser.add_argument('--version', type=str, default="IDSPACE", help='IDSPACE or IDNET')
    #parser.add_argument('--with_model', type=int, choices=[0, 1], default=1, help='Whether to use model guided method (1=yes, 0=no)')
    #parser.add_argument('--target_samples', type=int, default=1, help='Whether to use model guided method (1=yes, 0=no)')
    #parser.add_argument('--lambda0', type=float, default=1, help='The fractions of similarity score')
    #parser.add_argument('--lambda1', type=float, default=1, help='The fractions of consistency score')
    #parser.add_argument('--candidate_models', nargs='*', default="resnet50", help='List of candidate model names if using model guided method')
    #parser.add_argument('--config_info', type=str, default="data/configures/ALB_initial.yaml", help='Information about the segment and guided model')
    #parser.add_argument('--output_file', type=str, default="ALB_parameters.json", help='Information about the segment and guided model')

    # Parse the arguments
    args = parser.parse_args()

    with open(args.config, "r") as f:
        confs = yaml.safe_load(f)

    bo_settings = confs.get("bo_settings", {})
    #bo_settings['init_points'] = 200
    #bo_settings['n_iter'] = 500

    eval_args   = confs.get("eval_args", {})
    out_cfg     = confs.get("output", {})
    best_seting_file   = out_cfg.get("best_settings_file", "best_settings.json")

    # Access variables
    target_samples = eval_args['target_samples']
    with_model = eval_args['with_model']
    #target_samples = args.target_samples
    #with_model = args.with_model
    eval_args['target_samples'] = target_samples
    eval_args['with_model'] = with_model
    candidate_models = eval_args['candidate_models']
    if with_model and len(candidate_models) == 0:
        print("Error: You should provide model name if you use model guided method")
    area = confs['area']
    #best_seting_file = ''
    #if args.version == "IDSPACE":
    #    confs['template_path'] = f'SSIM_compare_with_IDNet/template_new/{area}_template.png'
    #    best_seting_file = f'SSIM_compare_with_IDNet/best_settings_IDSPACE/{area}_best_settings_{str(bo_settings['init_points'])}_{str(bo_settings['n_iter'])}_{str(target_samples)}.json'
    #elif args.version == "IDNET":
    #    confs['template_path'] = f'../Datageneration/templates/{area}_Template.png'
    #    best_seting_file = f'SSIM_compare_with_IDNet/best_settings_IDNET/{area}_best_settings_{str(bo_settings['init_points'])}_{str(bo_settings['n_iter'])}.json'
    #    with_model = 0
    #else:
    #    print("Not supported version: ", args.version)
    #    assert 0
    #confs['target_annotation_path'] = f"annotations/{area}_original_annotation.json"



    # Example print (can be removed)
    print(f"Target Samples: {target_samples}")
    print(f"With Model: {with_model}")
    print(f"Candidate Models: {candidate_models}")
    
    print("candidate_models:", candidate_models)

    configures = copy.deepcopy(confs)
    configures['eval_args'] = confs['eval_args']
    configures['bo_settings'] = confs['bo_settings']
    #if 'segments' not in configures:
    configures['segments'] = {}


    #configures['area'] = area
    fonts_path = confs['text_fonts_path']
    #fonts_path = "SSIM_compare_with_IDNet/fonts_for_IDSPACE"
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
    with open(eval_args['guided_datapaths']) as f:
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
    #key_pairs = confs['key_pairs']
    segments = confs['segments']
    copy_areas = confs['copy_areas']
    color_group = confs['color_group']
    save_quality = []
    font_group = set()
    colors = set()
    selected_segments = []
    pbounds = {}
    since = time.time()                                                                                                                                                                                                                                                                                                   
    for segment, search_parameters in segments.items():
    #search_parameters = segments[segment]
        if 'tunable' not in search_parameters or search_parameters['tunable'] != True:
            configures['segments'][segment] = search_parameters
            continue
        prefix = segment + '#'
        pbounds |= generate_pbounds(search_parameters['font_info'], prefix)
        pbounds[prefix + "spacing"] = (-3, 10)
        fg = search_parameters['font_group']
        font_group.add(fg)
        cg = search_parameters['color_group']
        colors.add(cg)
        selected_segments.append([segment, fg, cg])
        #pbounds[prefix + "blur_ratio"] = (0.0, 1.0)
    for v in font_group:
        pbounds[str(v) + '#' + "font_style_idx"] = (0, len(font_files) - 0.01)
        pbounds[str(v) + "#stroke_width"] = (0, 2)
    for c in colors:
        for k, v in color_group[c].items():
            pbounds[str(c) + '#' + k] = get_bound(v["initial"], v["margin"])
    #pbounds[prefix + "font_style_idx"] = (0, len(font_files) - 0.01)
    pbounds["save_quality1"] = (60, 100)
    pbounds["save_quality2"] = (60, 100)
    #print("###### pbounds:", pbounds)
    print("### pbounds:", pbounds)

    fixed_args = {
            'area': area,
            'selected_segments': selected_segments,
            'copy_areas': copy_areas,
            'confs': confs,
            'testing': True,
            'candidate_models': candidate_models,
            'with_model': with_model,
            'lambda0': eval_args['lambda0'],
            'lambda1': eval_args['lambda1']
    }

    #optimizer = optimization(pbounds = pbounds, segment_key = segment_key, confs = confs, testing = False, candidate_models = candidate_models, with_model = with_model, lambda0 = lambda0, lambda1 = lambda1)
    #optimizer = optimization(pbounds, fixed_args, bo_settings)
    optimizer = optimization(pbounds, fixed_args, bo_settings, font_files=font_files)
    #bps = optimizer.max['params']  # unchanged usage

    bps = optimizer.max['params']
    print(bps)
    for segment, fg, cg in selected_segments:
        prefix_s = segment + '#'
        prefix_f = str(fg) + '#'
        prefix_c = str(cg) + '#'
        tmp = {}
        tmp['x_position'] = int(bps[prefix_s + 'xp'])
        tmp['y_position'] = int(bps[prefix_s + 'yp'])
        tmp['stroke_width'] = int(bps[prefix_f + 'stroke_width'])
        tmp['color_r'] = int(bps[prefix_c + 'r'])
        tmp['color_g'] = int(bps[prefix_c + 'g'])
        tmp['color_b'] = int(bps[prefix_c + 'b'])
        tmp['spacing'] = int(bps[prefix_s + 'spacing'])
        tmp['blur_ratio'] = bps.get(prefix_s + 'blur_ratio', 0)
        font_file = font_files[int(bps[prefix_f + 'font_style_idx'])] 
        target_height = bps[prefix_s + 'text_height']
        font_size, height = find_font_size_for_height(font_file, target_height)
        tmp['font_style'] = font_file.split('/')[-1] 
        tmp['font_size'] = font_size
        #tmp['save_quality1'] = int(bps['save_quality1'])
        #tmp['save_quality1'] = int(bps['save_quality1'])

        configures['segments'][segment] = {'type': 'text', 'font_info': tmp}
    time_elapsed = time.time() - since                                                                                                                                                                                                                                                                                    
    print('Bayesian complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))                                                                                                                                                                                                                         
    configures['save_quality1'] = int(bps['save_quality1'])
    configures['save_quality2'] = int(bps['save_quality2'])
    #save_quality.append(int(bps['save_quality']))
    with open(best_seting_file, 'w') as f:
        json.dump(configures, f, indent = 4)

