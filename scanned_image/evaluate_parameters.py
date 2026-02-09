#!/usr/bin/env python
# coding: utf-8

import os
import math
import json
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import random
import math
import tempfile
from utils import *


#def oval_mask_in_corner(shape1, shape2, corner='top-left'):
#    height, width = shape1
#    mask = np.zeros((height, width), dtype=np.uint8)
#
#    # Define the size of the ellipse (can adjust as needed)
#    ellipse_height, ellipse_width = shape2
#
#    # Create coordinate grid
#    Y, X = np.ogrid[:ellipse_height, :ellipse_width]
#
#    # Ellipse equation: (x/a)^2 + (y/b)^2 <= 1
#    a = ellipse_width // 2 
#    b = ellipse_height // 2 
#    ellipse = ((X - a)**2 / (a**2 + 0.0001) + (Y - b)**2 / (b**2 + 0.0001)) >= 1
#    # Place the ellipse in the specified corner
#    if corner == 'top-left':
#        mask[:b, :a] = np.logical_or(mask[:b, :a], ellipse[:b, :a])
#    elif corner == 'top-right':
#        mask[:b, -a:] = np.logical_or(mask[:b, -a:], ellipse[:b, -a:])
#    elif corner == 'bottom-left':
#        mask[-b:, :a] = np.logical_or(mask[-b:, :a], ellipse[-b:, :a])
#    elif corner == 'bottom-right':
#        mask[-b:, -a:] = np.logical_or(mask[-b:, -a:], ellipse[-b:, -a:])
#    else:
#        raise ValueError("corner must be one of: 'top-left', 'top-right', 'bottom-left', 'bottom-right'")
#
#    return mask
#    #return ellipse
#
#def get_mask(img, param):
#    img_np = np.array(img)
#    shape1 = img.size
#    mask = np.zeros(shape1, dtype=np.uint8)
#    mask = np.logical_or(mask, oval_mask_in_corner(shape1, param['bottom_left'], corner='bottom-left'))
#    mask = np.logical_or(mask, oval_mask_in_corner(shape1, param['bottom_right'], corner='bottom-right'))
#    mask = np.logical_or(mask, oval_mask_in_corner(shape1, param['top_right'], corner='top-right'))
#    mask = np.logical_or(mask, oval_mask_in_corner(shape1, param['top_left'], corner='top-left'))
#
#    mask_img = Image.fromarray(((~mask).astype(np.uint8) * 255).transpose()).convert("L")
#    return mask_img          # <- ready for Image.paste
#
#def create_shadow(ori_img, id_img, paper_img, param):
#
#    # Load the santa hat
#    #bbox = get_mask(ori_img, param['corner_threshold'])
#    bbox = get_mask(id_img, param)
#    # copy bbox left corner
#    shadow = Image.new("RGBA", id_img.size, color=param['shadow_color'])
#    # Coordinates at which to draw the hat and shadow
#    w1, h1 = id_img.size
#    w2, h2 = paper_img.size
#    left_w = int((w2 - w1)/2)
#    left_h = int((h2 - h1) /2)
#    (o1, o2) = param['shadow_offset']
#    id_coords = (left_w + o1, left_h + o2)
#    shadow_coords = (left_w, left_h)
#    
#    # Custom-mask the shadow so it has the same shape as the santa hat
#    paper_img.paste(shadow, shadow_coords, mask=bbox)
#    paper_img = paper_img.filter(ImageFilter.GaussianBlur(radius=param['shadow_blur_radius']))
#    
#    # Apply an unsharp mask
#    #paper_img = paper_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=2))
#    
#    # add blur here
#    # Now paste the hat on top of the shadow
#    #paper_img.paste(id_img, box=id_coords, mask=bbox)
#    paper_img.paste(id_img, id_coords, mask=bbox)
#    return paper_img

def simulate_scan(params, image_path):
    img_pil = Image.open(image_path).convert("RGBA") # Keep alpha for resizing and pasting
    img_np = np.array(img_pil)
    original_width, original_height = img_pil.size

    ori_img = img_pil
    # Adjust brightness and contrast
    enhancer_brightness = ImageEnhance.Brightness(img_pil)
    img_pil = enhancer_brightness.enhance(params['brightness'])
    enhancer_contrast = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer_contrast.enhance(params['contrast'])

    # Sharpen
    if params['sharpness_factor'] != 1.0:
        enhancer_sharpness = ImageEnhance.Sharpness(img_pil)
        img_pil = enhancer_sharpness.enhance(params['sharpness_factor'])

    img_np = np.array(img_pil)

    # Add noise
    if params['noise_std'] > 0:
        noise = np.zeros_like(img_np, np.uint8)
        cv2.randn(noise, 0, params['noise_std'])
        img_np = cv2.add(img_np, noise).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

    # Blur
    if params['blur_radius'] > 0:
        img_np = cv2.GaussianBlur(img_np, (0, 0), sigmaX=params['blur_radius'], sigmaY=params['blur_radius'])
        img_pil = Image.fromarray(img_np)

    if params.get('paper_texture_path'):
        #try:
        if True:
            paper = Image.open(params['paper_texture_path']).convert("RGBA")
            #paper = paper.resize((params['paper_width'], params['paper_height']))
            paper_width, paper_height = paper.size
            id_width, id_height = img_pil.size
            if params.get('id_resized_shape'):
                (target_width, target_height) = params.get('id_resized_shape') 

                resized_id = img_pil.resize((target_width, target_height))
                id_img = ori_img.resize((target_width, target_height))
                #length = math.sqrt( target_width **2 + target_height ** 2)
                #start1 = random.randint(300, 1000)
                #start2 = random.randint(300, 1000)
            else:
                resized_id = img_pil
                id_img = ori_img
            start1, start2 = 600, 600
            paper_tmp = paper.crop((start1, start2, start1 + 1300, start2 + 1300))

            resized_id = create_shadow(ori_img, id_img, paper_tmp, params)

            resized_id_width, resized_id_height = resized_id.size
            #position = (x_off, y_off)
            #position = (random_bbox[:2])
            

            resized_id = resized_id.rotate(params['rotate'])
            # Composite resized ID onto paper
            if resized_id.mode in ('RGBA', 'LA'):
                paper.paste(resized_id, (start1, start2), resized_id.split()[3])
            else:
                paper.paste(resized_id, (start1, start2))
            final_image = paper.convert("RGB")
    return final_image


# Defined best parameters
bps = {}
bps['brightness'] = 1.2
bps['contrast'] = 0.8
bps['sharpness_factor'] = 1.2
bps['noise_std'] = 0.5
bps['blur_radius'] = 0.5
bps['shadow_offset'] = (2, 3)
bps['shadow_color'] = 128
bps['shadow_blur_radius'] = 2
bps['id_resized_shape'] = (1013, 641)
bps['top_left'] = (60, 60)
bps['top_right'] = (65, 65)
bps['bottom_left'] = (65, 65)
bps['bottom_right'] = (65, 60)
bps['rotate'] = 0
bps['save_quality1'] = 95
bps['save_quality2'] = 70


def evaluate_parameters(brightness, contrast, sharpness_factor, noise_std, blur_radius, shadow_offset1, shadow_offset2, 
                  shadow_color, shadow_blur_radius, id_resized_shape1, id_resized_shape2, 
                        top_left1, top_left2, top_right1, top_right2, bottom_left1, bottom_left2, bottom_right1, bottom_right2,
                        save_quality1, save_quality2, lambda0, lambda1,
                       target_samples, testing, candidate_models, with_model, paper_path, guided_datapaths, models ):
#def evaluate_parameters(**params):
#        brightness, contrast, sharpness_factor, noise_std, blur_radius, shadow_offset1, shadow_offset2, 
#                  shadow_color, shadow_blur_radius, id_resized_shape1, id_resized_shape2, 
#                        top_left1, top_left2, top_right1, top_right2, bottom_left1, bottom_left2, bottom_right1, bottom_right2,
#                        save_quality1, save_quality2,
#                       target_samples, testing, candidate_models, with_model):

    params = {}
    params['brightness'] = brightness #1.0 + random.uniform(-0.5, 0.5)
    params['contrast'] = contrast #1.0 + random.uniform(-0.5, 0.5)
    params['sharpness_factor'] = sharpness_factor #1.0 + random.uniform(-0.5, 0.5)
    params['noise_std'] = noise_std #random.uniform(0, 5)
    params['blur_radius'] = blur_radius #random.uniform(0, 1)
    params['rotate'] = 0
    params['shadow_offset'] = (int(shadow_offset1), int(shadow_offset2))
    params['shadow_color'] = (0, 0, 0, int(shadow_color))
    params['shadow_blur_radius'] = shadow_blur_radius #random.uniform(0, 5)
    params['id_resized_shape'] = (int(id_resized_shape1), int(id_resized_shape2))
    params['top_left'] = (int(top_left1), int(top_left2))
    params['top_right'] = (int(top_right1), int(top_right2))
    params['bottom_left'] = (int(bottom_left1), int(bottom_left2))
    params['bottom_right'] = (int(bottom_right1), int(bottom_right2))
    params['save_quality1'] = int(save_quality1)
    params['save_quality2'] = int(save_quality2)

    confs = {'models': models}
        
    prefix = '_'.join(candidate_models)
    #testing = params['testing']
    #target_samples = params['target_samples']
    #candidate_models = params['candidate_models']
    #with_model = params['with_model']
    generated_paths = []
    real_paths = []
    ssims = []
    with open(guided_datapaths, 'r') as f:
        guided_data = json.load(f)

    if not testing:
        choiced_images = guided_data['val'][:target_samples]
    else:
        choiced_images = guided_data['test']
    params['paper_texture_path'] = os.path.join(paper_path, os.listdir(paper_path)[0])
    bps['paper_texture_path'] = params['paper_texture_path']
    count = 0

    with tempfile.TemporaryDirectory() as tmpdirname:

        for choiced, label in tqdm(choiced_images):
            count += 1
            tar_path = os.path.join(tmpdirname, f"{choiced.split('/')[-1][:-4]}_tar_{os.getpid()}_{count}.jpg")
            syn_path = os.path.join(tmpdirname, f"{choiced.split('/')[-1][:-4]}_syn_{os.getpid()}_{count}.jpg")
            syn_result = simulate_scan(params, choiced)
            if label == 0:
                syn_result.convert("RGB").save(syn_path, quality=params['save_quality1'])
            else:
                syn_result.convert("RGB").save(syn_path, quality=params['save_quality2'])

            tar_result = simulate_scan(bps, choiced)
            if label == 0:
                tar_result.convert('RGB').save(tar_path, quality=bps['save_quality1'])
            else:
                tar_result.convert('RGB').save(tar_path, quality=bps['save_quality2'])

            #sample_np = np.array(tar_result.crop((500, 500, 2000, 2000)))
            #generated_np = np.array(syn_result.crop((500, 500, 2000, 2000)))
            sample_np = np.array(tar_result)
            generated_np = np.array(syn_result)

            real_paths.append([tar_path, label])
            generated_paths.append([syn_path, label])
            sv, _ = ssim(sample_np, generated_np, full=True, multichannel=True, channel_axis=-1)
            ssims.append(sv)

        score = 0
        if with_model:
            all_tests = eval_models(generated_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
            all_samples = eval_models(real_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
            accs = [accuracy_score(all_samples[key][0], all_tests[key][0]) for key in all_tests.keys()]
            score1 = sum(accs)/len(accs)
            score2 = (sum(ssims) / len(ssims))
            score = lambda0 * score1 + lambda1 * score2
            if testing:
                print(f"Test results for ssim + consistency -> Models:{all_tests.keys()}, Model consistency: {accs}, Evaluation score -> SSIM: {score2}, Consistency: {score1}")
            else:
                print(f"SSIM + Consistency -> Models:{all_tests.keys()}, Model consistency: {accs}, Evaluation score -> SSIM: {score2}, Consistency: {score1}")
        else:
            score = sum(ssims) / len(ssims)
            if testing:
                all_tests = eval_models(generated_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
                all_samples = eval_models(real_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
                accs = [accuracy_score(all_samples[key][0], all_tests[key][0]) for key in all_tests.keys()]
                score1 = sum(accs)/len(accs)
                score2 = (sum(ssims) / len(ssims))
                print(f"Test results for only ssim -> Models:{all_tests.keys()}, Model consistency: {accs}, Evaluation score -> SSIM: {score2}, Consistency: {score1}")
            else:
                print(f"SSIM, Evaluation score -> SSIM: {score}")
    return score


