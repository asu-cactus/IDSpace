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

def get_configs(area):
    with open(f"data/configures/{area}_parameters.json") as f:
        conf = json.load(f)
    return conf

if __name__ == '__main__':
    
    dataset = sys.argv[1]
    area = "ALB"
    segment_key = area + "_" + "surname"
    confs = get_configs(area)

    val_datas = []
    with open(confs['training_samples']) as f:
        ts = json.load(f)

    for v in ts['train']:
        if v[1] == 0:
            name = v[0].split('/')[-1]
            if name.split('_')[0].upper() == area:
                val_datas.append(name)
    for v in ts['val']:
        if v[1] == 0:
            name = v[0].split('/')[-1]
            if name.split('_')[0].upper() == area:
                val_datas.append(name)
    #confs['val_data'] = val_datas[:target_samples]
    confs['val_data'] = val_datas

    val_datas = []
    for v in ts['test']:
        if v[1] == 0:
            name = v[0].split('/')[-1]
            if name.split('_')[0].upper() == area:
                val_datas.append(name)
    confs['test_data'] = val_datas
    confs['save_path'] = "idnet_new_results.csv"


    # Define the bounds for the parameters

    candidate_models = []
    #best_sv_pv = evaluate_parameters_custom(
    best_sv_pv = evaluate_diffusion(
        segment = segment_key,
        confs = confs,
        testing = True,
        candidate_models = candidate_models,
        with_model = False,
        dataset = dataset
    )
    print("Best Evaluation on testing data:", best_sv_pv)

