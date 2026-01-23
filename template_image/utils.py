#!/usr/bin/env python
# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import cv2
import copy
import math
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import json
from tqdm import tqdm
import time                                                                                                                                                                                             
import torch                                                                                                                                                                                            
from torch.utils.data import Dataset, DataLoader                                                                                                                                                        
from torchvision import transforms                                                                                                                                                                      
from sklearn.metrics import accuracy_score
from rembg import remove
import random
import tempfile
import shutil


from fontTools.ttLib import TTFont
import uharfbuzz as hb
from io import BytesIO


from pathlib import Path
from typing import List, Dict, Any, Union
import json
import yaml
import pandas as pd
from personal_info_generation import pii_generation


ReturnType = Union[List[Dict[str, Any]], Dict[str, Any]]


def load_pii_info(file_path: str) -> ReturnType:
    path = Path(file_path)

    # ---- Basic checks ----
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"File is empty: {path}")

    suffix = path.suffix.lower()

    # ---- CSV (pandas) ----
    if suffix == ".csv":
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"Failed to read CSV file {path}: {exc}") from exc

        if df.empty:
            raise ValueError(f"CSV file has no rows: {path}")

        # List[Dict[str, Any]]
        return df.to_dict(orient="records")

    # ---- JSONL ----
    elif suffix in {".jsonl", ".ndjson"}:
        rows: List[Dict[str, Any]] = []

        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {lineno} of {path}: {exc}"
                    ) from exc

                if not isinstance(obj, dict):
                    raise TypeError(
                        f"Expected JSON object on line {lineno} of {path}, "
                        f"got {type(obj).__name__}"
                    )

                rows.append(obj)

        if not rows:
            raise ValueError(f"JSONL file contains no valid records: {path}")

        return rows

    # ---- YAML ----
    elif suffix in {".yml", ".yaml"}:
        with path.open("r", encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise ValueError(f"Failed to parse YAML file {path}: {exc}") from exc

        if data is None:
            raise ValueError(f"YAML file is empty: {path}")

        if not isinstance(data, dict):
            raise TypeError(
                f"Expected top-level YAML dict in {path}, "
                f"got {type(data).__name__}"
            )

        return pii_generation(data)

    # ---- Unsupported ----
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def font_supports_text(font_path: str, text: str) -> bool:
    """
    Return True if the font at font_path supports all characters in `text`,
    including shaping (no .notdef glyphs).
    """

    # ---------- 1) Check Unicode coverage (cmap) ----------
    try:
        ttfont = TTFont(font_path, recalcBBoxes=False, recalcTimestamp=False)
    except Exception:
        return False

    supported_codepoints = set()
    if "cmap" in ttfont:
        for table in ttfont["cmap"].tables:
            supported_codepoints.update(table.cmap.keys())

    for ch in text:
        if ord(ch) not in supported_codepoints:
            ttfont.close()
            return False

    # ---------- 2) Check shaping with HarfBuzz ----------
    buf = BytesIO()
    try:
        ttfont.save(buf)
        font_bytes = buf.getvalue()
    except Exception:
        ttfont.close()
        return False
    finally:
        ttfont.close()

    try:
        face = hb.Face(font_bytes)
        font = hb.Font(face)
        hb_buf = hb.Buffer()
        hb_buf.add_str(text)
        hb_buf.guess_segment_properties()
        hb.shape(font, hb_buf)
    except Exception:
        return False

    # HarfBuzz returns glyph id 0 for ".notdef" (missing glyph)
    for info in hb_buf.glyph_infos:
        if info.codepoint == 0:
            return False

    return True


def find_font_size_for_bbox(font_path, target_bbox, sample_text="SAMPLE"):
    t_height = target_bbox[3] - target_bbox[1]
    t_width = target_bbox[2] - target_bbox[0]
    t_height = random.randint(int(t_height/2), t_height)
    t_width = random.randint(int(t_width/2), t_width)

    for size in range(20, 200):
        font = ImageFont.truetype(font_path, size)
        bbox = font.getbbox(sample_text)
        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]
        if height >= t_height or width >= t_width:
            return size
    return size - 1

def find_font_size_for_height(font_path, target_height, sample_text="SAMPLE"):
    for size in range(1, 200):
        font = ImageFont.truetype(font_path, size)
        bbox = font.getbbox(sample_text)
        height = bbox[3] - bbox[1]
        if height >= target_height:
            return size, height
    return size - 1, height

def write_with_spacing1(imp, xx, yy, content, textcolor, test_mf, spacing):
    x_cursor = xx  # Starting x position
    for char in content:
        imp.text((x_cursor, yy), char, textcolor, font=test_mf)
        # Advance x_cursor based on character width + extra spacing

        left, top, right, bottom = test_mf.getbbox(char)
        char_width = right - left #, _ = imp.textsize(char, font=test_mf)
        x_cursor += char_width + spacing  # spacing is a value you define, e.g., 5
    return x_cursor - spacing
def write_with_spacing(imp, xx, yy, content, textcolor, test_mf, stroke_width, spacing):
    x_cursor = xx  # Starting x position
    for char in content:
        imp.text((x_cursor, yy), char, textcolor, font=test_mf, stroke_width=int(stroke_width))
        # Advance x_cursor based on character width + extra spacing

        left, top, right, bottom = test_mf.getbbox(char)
        char_width = right - left #, _ = imp.textsize(char, font=test_mf)
        x_cursor += char_width + spacing  # spacing is a value you define, e.g., 5
    return x_cursor - spacing

def resize_image(source_image, bbox):
    width, height = source_image.size
    left, top, right, bottom = bbox
    newsize = (right - left, bottom - top)
    #newsize = new_bgsize
    r1 = newsize[0]/width
    r2 = newsize[1]/height
    r = max(r1, r2)
    w1 = round(width * r)
    h1 = round(height * r)
    if r1 > r2:
        crop_box = (0,(h1-newsize[1])//2, w1, (h1 + newsize[1])//2 )
    else:
        crop_box = ((w1-newsize[0])//2, 0, (w1 + newsize[0])//2, h1 )
    tmpsize = (round(width * r), round(height * r))
    source_image = source_image.resize(tmpsize)
    source_image = source_image.crop(crop_box)
    return source_image


def write_bg(template, source_image, bbox):

    bluebg = source_image.convert("RGBA")
    left, top, right, bottom = bbox
    newsize = (right - left, bottom - top)
    bluebg = bluebg.resize(newsize)
    enhancer = ImageEnhance.Contrast(bluebg)
    img_low_contrast = enhancer.enhance(0.7)
    img_with_alpha = img_low_contrast.convert("RGBA")
    alpha_value = 100  # 0 is fully transparent, 255 is fully opaque
    alpha = Image.new("L", img_with_alpha.size, alpha_value)
    img_with_alpha.putalpha(alpha)
    bluebg = img_with_alpha

    template.paste(bluebg, bbox[:2], mask=bluebg)

def write_ghost(template, source_image, bbox):

    left, top, right, bottom = bbox
    ghostsize = (right - left, bottom - top)
    ghostimg = source_image.resize(ghostsize)

    ghostimg2 = ghostimg.copy()
    ghostimg2 = ghostimg2.convert("RGBA")
    r,g,b,a = ghostimg2.split()
    bluetint = a.point(lambda x: x * 0.5)
    ghostimg = Image.merge("RGBA", (r,g,b,bluetint))
    template.paste(ghostimg, bbox[:2], mask=ghostimg)

def write_portrait(template, source_image, bbox, remove_bg = False, change2gray = False):
    #source_image = Image.open(image_path).convert('RGBA')
    source_image = resize_image(source_image, bbox)
    if remove_bg:
        source_image = remove(source_image)

    if change2gray:
        white_bg = Image.new("RGBA", source_image.size, (255,255,255))
        white_bg.putalpha(200)
        source_image = Image.alpha_composite(white_bg,source_image)
        source_image = source_image.convert("L")
        source_image = remove(source_image)
    template.paste(source_image, bbox[:2], mask=source_image)
    return source_image 

def write_images(template, keys, image_path, innerdict):
    if 'portrait_background' in keys:
        innerdict['portrait_background'] = keys['portrait_background']
    if 'portrait' in keys:
        image_name = keys['portrait']['value']
        image_path = os.path.join(image_path, image_name)
        source_image = Image.open(image_path).convert('RGBA')
        params = keys['portrait']
        source_image = write_portrait(template, source_image, params['bbox'], params.get('remove_bg', False), params.get('change2gray', False))
        innerdict['portrait'] = keys['portrait']
        #innerdict['portrait']['image_path'] = image_path
    if 'ghost' in keys:
        params = keys['ghost']
        write_ghost(template, source_image, params['bbox'])
        innerdict['ghost'] = params

def write_signature(template, signature_fonts, signature_keys, innerdict):
    signaturetext = signature_keys['value']
    bbox = signature_keys['bbox']
    signature_font = signature_keys.get('font_style', None)
    if signature_font:
        font_path = os.path.join(signature_fonts, signature_font)
    else:
        signatureFontFiles = os.listdir(signature_fonts)
        random.shuffle(signatureFontFiles)
        signature_font = random.choice(signatureFontFiles)
        font_path = os.path.join(signature_fonts, signature_font)
    text_color = signature_keys.get('text_color', None)
    if text_color is not None:
        textcolor = (text_color['r'], text_color['g'], text_color['b'])
    else:
        textcolor = (0, 0, 0)
    im = ImageDraw.Draw(template)
    #signaturetext = signaturetext[:6]
    fontsize =  find_font_size_for_bbox(font_path, bbox, sample_text=signaturetext)
    mf = ImageFont.truetype(font_path, fontsize)
    real_bbox = mf.getbbox(signaturetext)
    delta_x = round((bbox[2] - bbox[0] - real_bbox[2] + real_bbox[0])/2)
    delta_y = round((bbox[3] - bbox[1] - real_bbox[3] + real_bbox[1])/2)
    x_position = bbox[0] + delta_x
    y_position = bbox[1] + delta_y
    textpos = (x_position, y_position)
    
    im.text(textpos, signaturetext, textcolor, font=mf)
    innerdict['signature'] = {
        'type': 'signature',
        'x_position': x_position,
        'y_position': y_position,
        "font_size": fontsize,
        "stroke_width": 0,
        "color_r": textcolor[0],
        "color_g": textcolor[1],
        "color_b": textcolor[2],
        "spacing": 0,
        "blur_ratio": 0,
        'font_style': signature_font,
        "value": signaturetext,
        "bbox": [x_position + real_bbox[0], y_position + real_bbox[1], x_position + real_bbox[2], y_position + real_bbox[3]]
    }



def write_parameters(x_position, y_position, font_size, stroke_width, color_r, color_g, color_b,                                                                                                                                                                                                 
                    font_style, spacing, blur_ratio, content, template, text_font_paths):                                                                                                                                                                                                          
    font_file = os.path.join(text_font_paths, font_style)                                                                                                                                                                                                                              
    target_sample_image = template
    img_width, img_height = target_sample_image.size
    #content = values[segment_key]['value']
    test_mf = ImageFont.truetype(font_file, font_size)
    left, top, right, bottom = test_mf.getbbox(content)
    text_width = right - left
    text_height = bottom - top
    if spacing and spacing > 0:
        text_width = right - left + text_width * spacing

    bbox = [max(0, x_position - 20), 
            max(0, y_position - 20), 
            min(img_width, x_position + text_width + 20), 
            min(img_height, y_position + text_height + 20)] 
    sample_np = np.array(target_sample_image.crop(bbox))
    #region = template_image.crop(bbox)
    #target_sample_image.paste(region, bbox)

    textcolor = (color_r, color_g, color_b)
    imp = ImageDraw.Draw(target_sample_image)
    if spacing:
        real_right = write_with_spacing(imp, x_position, y_position, content, textcolor, test_mf, stroke_width, spacing)
        real_bbox = [x_position + left, y_position + top, real_right, y_position + bottom]
    else:
        imp.text((x_position, y_position), content, textcolor, font=test_mf, stroke_width=int(stroke_width))
        real_bbox = [x_position + left, y_position + top, x_position + right, y_position + bottom]

    if blur_ratio:
        im2 = target_sample_image.crop(bbox)
        im2 = im2.filter(ImageFilter.GaussianBlur(radius=blur_ratio))
        Image.Image.paste(target_sample_image, im2, (bbox[0], bbox[1]))
    return real_bbox

def region_attributes(value = None, font_style = None, font_size = None, font_color = None, bbox = None):
    attributes = {}
    attributes['value'] = value
    attributes['font_style'] = font_style
    attributes['font_size'] = font_size
    attributes['font_color'] = font_color
    attributes['bbox'] = bbox
    return attributes


def read_json(path: str):
    with open(path) as f:
        return json.load(f)
def write_json(data:dict, path:str):
    with open(path, "w", encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

class ImageCSVDataset(Dataset):                                                                                                                                                                         
    def __init__(self, input_list, transform=None):                                                                                                                                                     
        self.input = input_list                                                                                                                                                                         
        self.transform = transform                                                                                                                                                                      
                                                                                                                                                                                                        
    def __len__(self):                                                                                                                                                                                  
        return len(self.input)                                                                                                                                                                          
                                                                                                                                                                                                        
    def __getitem__(self, idx):                                                                                                                                                                         
        img_path = self.input[idx][0]                                                                                                                                                                   
        image = Image.open(img_path).convert('RGB')                                                                                                                                                     
        label = self.input[idx][1]                                                                                                                                                                      
                                                                                                                                                                                                        
        if self.transform:                                                                                                                                                                              
            image = self.transform(image)                                                                                                                                                               
                                                                                                                                                                                                        
        return image, label, img_path                                                                                                                                                                   

def get_font_styles(base_walk_path):
    results = []
    for r, d, f in os.walk(base_walk_path):
        f.sort()
        for file in f:
            if file.endswith('.ttf'):
                results.append(base_walk_path + file)
    return results

def load_all_templates(input_folder, area):                                                                                                                                                                                                                                                                                             
    input_paths = {}                                                                                                                                                                                                                                                                                                      
    for image_name in os.listdir(input_folder):                                                                                                                                                                                                                                                                           
        if image_name[:3] == area.lower():                                                                                                                                                                                                                                                                                
            input_path = os.path.join(input_folder, image_name)                                                                                                                                                                                                                                                           
            input_paths[image_name] = input_path                                                                                                                                                                                                                                                                                
    return input_paths 

def load_annotations(annotation_path):
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    return annotations

def save_test_files(file_list, destination_dir = "data/outputs/template_test/"):
    """
    Copies a list of files from a source directory to a destination directory.

    Args:
        file_list (list): A list of filenames (strings) to be copied.
        source_dir (str): The path to the source directory.
        destination_dir (str): The path to the destination directory.
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)  # Create destination directory if it doesn't exist

    for filename in file_list:
        if os.path.exists(filename[0]):
            shutil.copy(filename[0], destination_dir)
        else:
            print(f"Warning: File '{filename}' not found in '{source_dir}'")


def eval_models(test_paths, confs, testing, candidate_models):                                                                                                                                                                                                                                                                
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    since = time.time()                                                                                                                                                                                                                                                                                                   
    for M in confs['models']:
        name = M['name']
        if not testing:
            if name not in candidate_models:
                continue
        model_path = M['path']
        im_size = M['im_size']
        transform = transforms.Compose([                                                                                                                                                                        
            transforms.Resize((im_size, im_size)),                                                                                                                                                                      
            transforms.ToTensor(),                                                                                                                                                                              
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])                                                                                                                         
        ])                                                                                                                                                                                                      
        test_dataset = ImageCSVDataset(test_paths, transform=transform)                                                                                                                                        
        test_loader = DataLoader(test_dataset, batch_size= 32, num_workers = 8, shuffle=False) 
        acc_history = []                                                                                                                                                                                                                                                                                                      
        best_acc = 0.0                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                              
        model = torch.load(model_path, weights_only = False, map_location = device)                                                                                                                                                                                                                                                                                       
        model.eval()                                                                                                                                                                                                                                                                                                          
        model.to(device)                                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                              
        running_corrects = 0                                                                                                                                                                                                                                                                                                  
        All_labels = []
        All_preds = []
                                                                                                                                                                                                                                                                                                                              
        for inputs, labels, filenames in tqdm(test_loader):                                                                                                                                                                                                                                                                   
            inputs = inputs.to(device)                                                                                                                                                                                                                                                                                        
            labels = labels.to(device)                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                              
            with torch.no_grad():                                                                                                                                                                                                                                                                                             
                outputs = model(inputs)                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                                                              
            _, preds = torch.max(outputs, 1)                                                                                                                                                                                                                                                                                  
            All_labels.extend(labels.data.cpu().numpy().tolist())
            All_preds.extend(preds.cpu().numpy().tolist())
            values, indices = torch.sort(outputs, dim=1, descending=True)                                                                                                                                                                                                                                                     
            running_corrects += torch.sum(preds == labels.data)                                                                                                                                                                                                                                                               
        epoch_acc = running_corrects.double() / len(test_loader.dataset)                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                          
        print('Model: {}, Acc: {:.4f}'.format(name, epoch_acc))                                                                                                                                                                                                                                                                                
        results[name]= [All_preds, All_labels]
                                                                                                                                                                                                                                                                                                                          
    time_elapsed = time.time() - since                                                                                                                                                                                                                                                                                    
    print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                          
    return results                                                                                                                                                                                                                                                                                                    

def coord_to_shape(bbox):
    """This function convert the kin of the shape from bbox rectangle x0,y0 + heigh and weight to the polygon coordenades.

    Returns:
        _type_: _description_
    """

    x, y, x_f, y_f = bbox
    c1, c2, c3, c4 = [x, y], [x_f, y], [x_f, y_f], [x, y_f]

    return [c1, c2, c3, c4]

def mask_from_info(img:np.ndarray, shape:np.ndarray):

    """"
        This f(x) extract the  ROI that will be inpainted

    """
    def midpoint(x1, y1, x2, y2):
        x_mid = int((x1 + x2) / 2)
        y_mid = int((y1 + y2) / 2)
        return (x_mid, y_mid)

    x0, x1, x2, x3 = shape[0][0], shape[1][0], shape[2][0], shape[3][0]
    y0, y1, y2, y3 = shape[0][1], shape[1][1], shape[2][1], shape[3][1]


    xmid0, ymid0 = midpoint(x1, y1, x2, y2)
    xmid1, ymid1 = midpoint(x0, y0, x3, y3)

    thickness = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.line(mask, (xmid0, ymid0), (xmid1, ymid1), 255, thickness)

    masked = cv2.bitwise_and(img, img, mask=mask)

    return mask, masked

def coord_to_coord1(bbox):
    """This function convert the kin of the shape from bbox rectangle x0,y0 + heigh and weight to the polygon coordenades.

    Returns:
        _type_: _description_
    """

    x, y, x_f, y_f = bbox

    return [x, y, x_f - x, y_f - y]

def get_font_scale(inner_path: str = os.path.join(os.getcwd(), "data", "fake_fonts", "TTF")):

    ## TODO solve
    try:
        deja = [i for i in os.listdir(inner_path) if "DejaVu" in i]

    except FileNotFoundError:
        for root, dirs, files in os.walk(os.getcwd()):
            for name in dirs:
                if "TTF" == name:
                    inner_path = os.path.join(root, name)
                    break

        deja = [i for i in os.listdir(inner_path) if "DejaVu" in i]

    selected = random.choice(deja)

    return os.path.join(inner_path, selected)

def get_optimal_font_scale(text, width):
    fontsize = 1  # starting font size
    sel_font =  get_font_scale()
    stop  = False  # portion of image width you want text width to be
    img_fraction = 1
    try:
        font = ImageFont.truetype(font=sel_font, size=fontsize ,encoding="unic")
    except:
        sel_font = "data/fake_fonts/TTF/DejaVuSans.ttf"
        font = ImageFont.truetype(font=sel_font, size=fontsize ,encoding="unic")

    bbox = font.getbbox(text)
    b0 = bbox[2] - bbox[0]
    b1 = bbox[3] - bbox[1]
    while (b0 < img_fraction*width) and (stop == False):
        # iterate until the text size is just larger than the criteria
        if b0 == 0:
            sel_font =  "data/fake_fonts/TTF/DejaVuSans.ttf"

            if b1 == 0:
                stop = True
                break

        fontsize += 1
        font = ImageFont.truetype(sel_font, fontsize ,encoding="unic")
        bbox = font.getbbox(text)
        b0 = bbox[2] - bbox[0]
        b1 = bbox[3] - bbox[1]

    # optionally de-increment to be sure it is less than criteria
    fontsize -= 1
    font = ImageFont.truetype(sel_font, fontsize ,encoding="unic")

    return font

def inpaint_image(img: np.ndarray, coord:np.ndarray, mask: np.ndarray, text_str: str):                                                                                                                                                                                                                                    
    """                                                                                                                                                                                                                                                                                                                   
    Inpaints the masked region in the input image using the TELEA algorithm and adds text to it.                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                          
    Args:                                                                                                                                                                                                                                                                                                                 
        img (np.ndarray): Input image.                                                                                                                                                                                                                                                                                    
        coord (np.ndarray[int, ...]): An array of integers representing the (x,y) coordinates of the top-left corner,                                                                                                                                                                                                     
            as well as the width and height of the region where the text will be added.                                                                                                                                                                                                                                   
        mask (np.ndarray): A binary mask with the same shape as `img`, where the masked pixels have value 0.                                                                                                                                                                                                              
        text_str (str): The text to be added to the inpainted region.                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                                          
    Returns:                                                                                                                                                                                                                                                                                                              
        np.ndarray: A numpy array representing the inpainted image with the text added to it.                                                                                                                                                                                                                             
    """                                                                                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                                                          
    inpaint = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)                                                                                                                                                                                                                                                                
    fake_text_image = copy.deepcopy(inpaint)                                                                                                                                                                                                                                                                              
    x0, y0, w, h = coord                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                          
    color = (0, 0, 0)                                                                                                                                                                                                                                                                                                     
    font  = get_optimal_font_scale(text_str, w)                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                                                          
    img_pil = Image.fromarray(fake_text_image)                                                                                                                                                                                                                                                                            
    draw = ImageDraw.Draw(img_pil)                                                                                                                                                                                                                                                                                        
    draw.text(((x0, y0)), text_str, font=font, fill=color)                                                                                                                                                                                                                                                                
    fake_text_image = np.array(img_pil)                                                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                                                          
    return fake_text_image

def get_data_range(img):
    img = img.astype("float64")
    return img.max() - img.min()

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import tempfile
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.filters import sobel
import math

def evaluate_parameters_global2(**kargs):
    save_quality = int(kargs['save_quality'])
    confs = kargs['confs']
    testing = kargs['testing']
    candidate_models = kargs['candidate_models']
    with_model = kargs['with_model']
    lambda0 = kargs['lambda0']
    lambda1 = kargs['lambda1']
    area = kargs['area']

    quality = int(save_quality)
    annotation_path = confs['target_annotation_path']
    template_path = confs['template_path']
    font_path = confs["text_fonts_path"]

    if 'font_files' in confs:
        font_styles = confs['font_files']
    else:
        font_styles = get_font_styles(font_path)

    template_image = Image.open(template_path).convert("RGB")
    if testing:
        val_datas = confs['test_data']
    else:
        val_datas = confs['val_data']
    annotations = load_annotations(annotation_path)

    test_paths = []
    ssims = []
    ssims_all = []
    psnrs = []
    sample_paths = []
    outpath_prefix = ''

    # parameters for combining intensity-SSIM and edge-SSIM
    alpha_edge_weight = 0.5   # weight for edge-SSIM; intensity weight = 1 - this
    edge_alpha_threshold = 16  # alpha threshold from rendered layer to consider pixel rendered

    # helper: robust ssim full call (returns mean_ssim, ssim_map)
    def ssim_full(a, b):
        """Return (mean_ssim, ssim_map). Accepts color images HxWx3 or grayscale HxW"""
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        # Try channel_axis variant first
        try:
            mean_val, ssim_map = ssim_fn(a, b, full=True, channel_axis=-1)
        except TypeError:
            # older versions use multichannel keyword
            try:
                mean_val, ssim_map = ssim_fn(a, b, full=True, multichannel=True)
            except TypeError:
                # fallback to grayscale
                if a.ndim == 3 and a.shape[2] >= 3:
                    def to_gray(arr):
                        return 0.2989*arr[...,0] + 0.5870*arr[...,1] + 0.1140*arr[...,2]
                    A = to_gray(a)
                    B = to_gray(b)
                    mean_val, ssim_map = ssim_fn(A, B, full=True)
                else:
                    mean_val, ssim_map = ssim_fn(a, b, full=True)
        ssim_map = np.asarray(ssim_map, dtype=np.float32)
        if ssim_map.ndim == 3:
            # per-channel maps: average channels to get per-pixel map
            ssim_map = np.mean(ssim_map, axis=-1)
        return float(mean_val), ssim_map

    # helper: compute edge ssim map (on luminance)
    def edge_ssim_map(img_gt, img_pred):
        """Return per-pixel SSIM map of Sobel edge magnitudes (2D)."""
        # convert to gray if needed (luminosity)
        def to_gray(arr):
            if arr.ndim == 3 and arr.shape[2] >= 3:
                return 0.2989*arr[...,0] + 0.5870*arr[...,1] + 0.1140*arr[...,2]
            return arr
        A = to_gray(img_gt).astype(np.float64)
        B = to_gray(img_pred).astype(np.float64)
        eA = sobel(A)
        eB = sobel(B)
        # adapt ssim call (edges are single-channel)
        try:
            mean_e, map_e = ssim_fn(eA, eB, full=True, channel_axis=None)
        except TypeError:
            mean_e, map_e = ssim_fn(eA, eB, full=True, multichannel=False)
        map_e = np.asarray(map_e, dtype=np.float32)
        if map_e.ndim == 3:
            map_e = np.mean(map_e, axis=-1)
        return map_e

    # Create a temporary directory
    real_img = ''
    fake_img = ''
    test_bbox = ''
    test_content = ''
    with tempfile.TemporaryDirectory() as tmpdirname:
        for filename, values in annotations.items():
            # replicate your earlier filtering
            if filename.split('_')[-1] == "00.jpg":
                target_sample = "data/templates/Images/reals/" + filename
                target_sample_image = Image.open(target_sample).convert("RGB")
                sample_np = np.array(target_sample_image)
                real_img = sample_np
                img_width, img_height = target_sample_image.size

                # Fresh copy of template for each sample
                working_template = template_image.copy()

                # prepare a transparent RGBA layer for masks (same size as template)
                layer = Image.new("RGBA", working_template.size, (0, 0, 0, 0))
                layer_draw = ImageDraw.Draw(layer)

                # draw on the working_template's RGB and on the layer (alpha) in parallel
                imp = ImageDraw.Draw(working_template)

                # collect exact masks per segment (full-image boolean masks)
                segment_masks = []  # list of HxW boolean arrays (True where rendered)
                segment_areas = []  # integer area for each segment (for weighting)
                # also collect positions to allow blur composition later
                segment_blurs = []  # store tuples (bbox, blur_radius) if blur needed

                for segment_key, fg, cg in kargs['segments']:
                    try:
                        prefix_s = segment_key + '#'
                        prefix_f = str(fg) + '#'
                        prefix_c = str(cg) + '#'
                        xx = int(kargs[prefix_s + 'xp'])
                        yy = int(kargs[prefix_s + 'yp'])
                        font_size = kargs.get(prefix_s + 'font_size', None)
                        if font_size:
                            font_size = int(font_size)
                        target_height = kargs.get(prefix_s + 'text_height', None)
                        blur_ratio = kargs.get(prefix_s + 'blur_ratio', None)
                        spacing = kargs.get(prefix_s + 'spacing', None)
                        if spacing:
                            spacing = int(spacing)
                        font_style_idx = int(kargs[prefix_f + 'font_style_idx'])
                        xc = int(kargs[prefix_c + 'r'])
                        yc = int(kargs[prefix_c + 'g'])
                        zc = int(kargs[prefix_c + 'b'])
                        outpath_prefix = "IDSPACE"
                    except:
                        prefix_s = segment_key + '#'
                        prefix_f = prefix_s
                        prefix_c = prefix_s
                        xx = int(kargs[prefix_s + 'xp'])
                        yy = int(kargs[prefix_s + 'yp'])
                        font_size = kargs.get(prefix_s + 'font_size', None)
                        if font_size:
                            font_size = int(font_size)
                        target_height = kargs.get(prefix_s + 'text_height', None)
                        blur_ratio = kargs.get(prefix_s + 'blur_ratio', None)
                        spacing = kargs.get(prefix_s + 'spacing', None)
                        if spacing:
                            spacing = int(spacing)
                        font_style_idx = int(kargs[prefix_f + 'font_style_idx'])
                        xc = int(kargs[prefix_c + 'r'])
                        yc = int(kargs[prefix_c + 'g'])
                        zc = int(kargs[prefix_c + 'b'])
                        outpath_prefix = "IDNET"

                    font_file = font_styles[int(font_style_idx)]
                    #bbox = confs['segments'][segment_key].get('bbox', None)
                    bbox = values[segment_key]['bbox'] #confs['segments'][segment_key].get('bbox', None)
                    content = values[segment_key]['value']
                    test_bbox = bbox
                    test_content = content

                    if not font_supports_text(font_file, content):
                        print(f"{font_file} does not support {content}")
                        return 0

                    if target_height:
                        font_size, height = find_font_size_for_height(font_file, target_height)
                    test_mf = ImageFont.truetype(font_file, font_size)
                    left, top, right, bottom = test_mf.getbbox(content)

                    if not bbox:
                        text_width = right - left
                        text_height = bottom - top
                        if spacing and spacing > 0:
                            text_width = right - left + text_width * spacing

                        bbox = [max(0, xx - 30),
                                max(0, yy - 30),
                                min(img_width, xx + text_width + 30),
                                min(img_height, yy + text_height + 30)]

                    # Render text on both the working template (RGB) and the transparent layer (alpha)
                    textcolor = (xc, yc, zc)
                    if spacing:
                        # draw with spacing onto both surfaces
                        write_with_spacing1(imp, xx, yy, content, textcolor, test_mf, spacing)
                        write_with_spacing1(layer_draw, xx, yy, content, (255,255,255,255), test_mf, spacing)
                    else:
                        imp.text((xx, yy), content, textcolor, font=test_mf)
                        layer_draw.text((xx, yy), content, fill=(255,255,255,255), font=test_mf)

                    # If blur is required, save bbox & blur radius to apply after compositing
                    if blur_ratio:
                        # ensure bbox ints in image coords
                        x0 = max(0, int(bbox[0])); y0 = max(0, int(bbox[1]))
                        x1 = min(img_width, int(bbox[2])); y1 = min(img_height, int(bbox[3]))
                        segment_blurs.append(((x0, y0, x1, y1), blur_ratio))





                # Apply blur operations on working_template if any (same as your prior code)
                for (x0, y0, x1, y1), br in segment_blurs:
                    im2 = working_template.crop((x0, y0, x1, y1))
                    im2 = im2.filter(ImageFilter.GaussianBlur(radius=br))
                    Image.Image.paste(working_template, im2, (x0, y0))

                # Composite the layer into the working_template (so generated image equals previous behavior)
                # We do this after blur so blur applies to text region as before.
                working_rgba = working_template.convert("RGBA")
                working_rgba.alpha_composite(layer)
                generated = working_rgba.convert("RGB")
                generated_np = np.array(generated)
                fake_img = np.array(working_template)


                timestamp = time.strftime("%Y%m%d_%H%M%S")
                real_name = f"{tmpdirname}/newBO_{outpath_prefix}_{timestamp}_{filename}"
                # save working_rgba composite result (same as original saved output)
                working_template.save(real_name, format='JPEG', subsampling=0, quality=quality)

                # Build exact rendered mask: alpha channel of `layer` (> threshold = rendered)
                alpha_arr = np.asarray(layer.split()[-1]).astype(np.uint8)  # HxW
                rendered_mask = (alpha_arr > edge_alpha_threshold)  # True where text rendered

                # If no rendered pixels (mask empty), fallback to bbox-based masking to avoid NaNs
                # But here we prefer exact mask
                # Now compute SSIM map (intensity) and edge-SSIM map
                try:
                    _, ssim_map = ssim_full(sample_np, generated_np)
                except Exception:
                    # fallback: mean ssim and uniform map
                    try:
                        mean_val = float(ssim_fn(sample_np, generated_np))
                        ssim_map = np.ones((sample_np.shape[0], sample_np.shape[1]), dtype=np.float32) * mean_val
                    except Exception:
                        ssim_map = np.zeros((sample_np.shape[0], sample_np.shape[1]), dtype=np.float32)

                # compute edge ssim map
                try:
                    edge_map = edge_ssim_map(sample_np, generated_np)
                except Exception:
                    edge_map = np.zeros_like(ssim_map)

                # For each segment, compute masked means using the exact mask clipped to segment bbox
                # We need to recompute segment bboxes based on the drawn text extents.
                # We'll use confs['segments'][segment_key].get('bbox', computed earlier) approach:
                per_region_means = []
                per_region_areas = []

                # Rebuild segment bbox list in the same order as drawn. We can use the same code to compute bbox.
                # For simplicity, iterate again through segments to compute bboxes (deterministic order).
                for segment_key, fg, cg in kargs['segments']:
                    try:
                        prefix_s = segment_key + '#'
                        xx = int(kargs[prefix_s + 'xp'])
                        yy = int(kargs[prefix_s + 'yp'])
                        font_size = kargs.get(prefix_s + 'font_size', None)
                        if font_size:
                            font_size = int(font_size)
                        spacing = kargs.get(prefix_s + 'spacing', None)
                        if spacing:
                            spacing = int(spacing)
                        target_height = kargs.get(prefix_s + 'text_height', None)
                        font_style_idx = int((str(fg) + '#') + 'font_style_idx') if False else None
                        # we won't recompute font_style_idx here; instead rely on bbox in confs if present
                        bbox = confs['segments'][segment_key].get('bbox', None)
                    except:
                        bbox = confs['segments'][segment_key].get('bbox', None)

                    # If bbox not present, attempt to estimate via getbbox using the font again:
                    # We recompute font and content to be safe
                    content = values[segment_key]['value']
                    bbox = values[segment_key]['bbox'] #confs['segments'][segment_key].get('bbox', None)
                    # attempt to get font and font_size again (best-effort)
                    try:
                        prefix_s = segment_key + '#'
                        prefix_f = str(fg) + '#'
                        font_style_idx = int(kargs[prefix_f + 'font_style_idx'])
                        font_file = font_styles[int(font_style_idx)]
                        font_size = kargs.get(prefix_s + 'font_size', None)
                        if font_size:
                            font_size = int(font_size)
                        else:
                            # fallback
                            font_size = 12
                        test_mf = ImageFont.truetype(font_file, font_size)
                        left, top, right, bottom = test_mf.getbbox(content)
                        text_width = right - left
                        text_height = bottom - top
                        spacing = kargs.get(prefix_s + 'spacing', None)
                        if spacing and int(spacing) > 0:
                            text_width = right - left + text_width * int(spacing)
                        xx = int(kargs[prefix_s + 'xp'])
                        yy = int(kargs[prefix_s + 'yp'])
                        bbox_est = [max(0, xx - 30),
                                    max(0, yy - 30),
                                    min(img_width, xx + text_width + 30),
                                    min(img_height, yy + text_height + 30)]
                        x0 = max(0, int(bbox_est[0])); y0 = max(0, int(bbox_est[1]))
                        x1 = min(img_width, int(bbox_est[2])); y1 = min(img_height, int(bbox_est[3]))
                    except Exception:
                        # fallback to whole image
                        x0, y0, x1, y1 = 0, 0, img_width, img_height

                    # clip the rendered mask to this bbox to produce exact per-segment mask
                    seg_mask = rendered_mask[y0:y1, x0:x1]
                    if seg_mask.size == 0 or seg_mask.sum() == 0:
                        per_region_means.append(np.nan)
                        per_region_areas.append(0)
                    else:
                        # masked means on ssim_map and edge_map within bbox
                        ssim_crop = ssim_map[y0:y1, x0:x1]
                        edge_crop = edge_map[y0:y1, x0:x1]

                        # select only rendered pixels inside bbox
                        sel = seg_mask
                        if sel.sum() == 0:
                            per_region_means.append(np.nan)
                            per_region_areas.append(0)
                        else:
                            mean_ssim_int = float(np.nanmean(ssim_crop[sel]))
                            mean_ssim_edge = float(np.nanmean(edge_crop[sel]))
                            # combined per-region
                            comb = (1.0 - alpha_edge_weight) * mean_ssim_int + alpha_edge_weight * mean_ssim_edge
                            per_region_means.append(comb)
                            per_region_areas.append(int(sel.sum()))

                # compute area-weighted overall SSIM for this sample
                areas_arr = np.array(per_region_areas, dtype=float)
                means_arr = np.array([0.0 if np.isnan(m) else m for m in per_region_means], dtype=float)
                total_area = areas_arr.sum()
                if total_area <= 0:
                    # fallback: global mean SSIM over the whole image (from ssim_map)
                    sample_ssim = float(np.nanmean(ssim_map))
                else:
                    sample_ssim = float((means_arr * areas_arr).sum() / total_area)

                # compute PSNR as before (whole-image)
                psnr_val = psnr_fn(sample_np, generated_np, data_range=get_data_range(sample_np))

                sv, _ = ssim(sample_np, generated_np, full=True, multichannel=True, channel_axis=-1)

                ssims.append(sample_ssim)
                ssims_all.append(sv)
                psnrs.append(psnr_val)

                test_paths.append([real_name, 0])
                sample_paths.append([target_sample, 0])

                break  # original loop breaks after first matching filename

        # after all samples processed
        if with_model:
            all_tests = eval_models(test_paths, confs, testing, candidate_models)
            all_samples = eval_models(sample_paths, confs, testing, candidate_models)
            accs = [accuracy_score(all_samples[key][0], all_tests[key][0]) for key in all_tests.keys()]
            score = lambda0 * (sum(accs)/len(accs)) + lambda1 * ((sum(ssims) / len(ssims)))
        else:
            ssim_av = sum(ssims) / len(ssims) if ssims else 0.0
            ssim_all = sum(ssims_all) / len(ssims_all) if ssims_all else 0.0
            psnr_av = sum(psnrs) / len(psnrs) if psnrs else 0.0
            score = 0.99 * ssim_av + 0.01 * (psnr_av / 40.0)

        if testing:
            real_shape = coord_to_shape(test_bbox)
            real_mask, _ = mask_from_info(real_img, real_shape)
            real_coord1 = coord_to_coord1(test_bbox)
            fake_real_image =  inpaint_image(img=real_img, coord=real_coord1, mask=real_mask, text_str=test_content)

            fake_image =  inpaint_image(img=fake_img, coord=real_coord1, mask=real_mask, text_str=test_content)

            psnr_val = psnr_fn(fake_real_image, fake_image, data_range=get_data_range(fake_real_image))
            sv, _ = ssim(fake_real_image, fake_image, full=True, multichannel=True, channel_axis=-1)



            save_test_files(test_paths, destination_dir = "data/outputs/template_test/")
            all_tests = eval_models(test_paths, confs, testing, candidate_models)
            all_samples = eval_models(sample_paths, confs, testing, candidate_models)
            accs = [accuracy_score(all_samples[key][0], all_tests[key][0]) for key in all_tests.keys()]
            print(f"Evaluation score: {score}, ssim: {ssim_av}, psnr: {psnr_av}, ssim with whole picture: {ssim_all}")
            print(f"Evaluation fake: ssim: {sv}, psnr: {psnr_val}")

    return score




def evaluate_parameters_global(**kargs):
    save_quality1 = int(kargs['save_quality1'])
    save_quality2 = int(kargs['save_quality2'])
    confs = kargs['confs']
    testing = kargs['testing']
    candidate_models = kargs['candidate_models']
    with_model = kargs['with_model']
    lambda0 = kargs['lambda0']
    lambda1 = kargs['lambda1']
    area = kargs['area']

    #quality = int(save_quality)
    annotation_path = confs['target_annotation_path']
    template_path = confs['template_path']
    font_path = confs["text_fonts_path"]
    
    if 'font_files' in confs:
        font_styles = confs['font_files']
    else:
        font_styles = get_font_styles(font_path)

    if testing:
        val_datas = confs['test_data']
    else:
        val_datas = confs['val_data']
    annotations = load_annotations(annotation_path)




    generated_paths = []
    ssims = []
    psnrs = []
    ssims_inpaint = []
    psnrs_inpaint = []
    sample_paths = []
    outpath_prefix = ''
    panalty = 0

    # Create a temporary directory
    candidate_segments = [k[0] for k in kargs['selected_segments']]
    with tempfile.TemporaryDirectory() as tmpdirname:
        for filename, values in annotations.items():
            if filename in val_datas:
            #if filename.split('_')[-1]  == "00.jpg":
                target_sample = "data/templates/Images/reals/" + filename
                #print(target_sample)
                #target_sample = val_datas[filename]
                template_image = Image.open(template_path).convert("RGB")
                target_sample_image = Image.open(target_sample).convert("RGB")
                sample_np = np.array(target_sample_image)
                img_width, img_height = target_sample_image.size
                imp = ImageDraw.Draw(template_image)

                choiced_inpaint_segment = random.choice(candidate_segments)

                for segment_key, fg, cg in kargs['selected_segments']:
                    #try:
                    prefix_s = segment_key + '#'
                    prefix_f = str(fg) + '#'
                    prefix_c = str(cg) + '#'
                    outpath_prefix = "IDSPACE"

                    xx = int(kargs[prefix_s + 'xp'])
                    yy = int(kargs[prefix_s + 'yp'])
                    font_size = kargs.get(prefix_s + 'font_size', None)
                    if font_size:
                        font_size = int(font_size)
                    target_height = kargs.get(prefix_s + 'text_height', None)
                    blur_ratio = kargs.get(prefix_s + 'blur_ratio', None)
                    spacing = kargs.get(prefix_s + 'spacing', None)
                    if spacing:
                        spacing = int(spacing)
                    stroke_width = int(kargs.get(prefix_f + 'stroke_width', 0))
                    font_style_idx = int(kargs[prefix_f + 'font_style_idx'])
                    xc = int(kargs[prefix_c + 'r'])
                    yc = int(kargs[prefix_c + 'g'])
                    zc = int(kargs[prefix_c + 'b'])


                    font_file = font_styles[int(font_style_idx)]

                    #area, segment_key = segment.split('_', 1)
                    #bbox = confs['segments'][segment_key].get('bbox', None)

                    try:
                        content = values[segment_key]['value']
                        bbox = values[segment_key]['bbox']
                    except:
                        continue

                    if choiced_inpaint_segment == segment_key:
                        choiced_content = content
                        choiced_bbox = bbox


                    if not font_supports_text(font_file, content) and not testing:
                        print(f"{font_file} does not support {content}")
                        return 0

                    if target_height:
                        font_size, height = find_font_size_for_height(font_file, target_height)
                    test_mf = ImageFont.truetype(font_file, font_size)
                    left, top, right, bottom = test_mf.getbbox(content)
                    spacing = int((-right + left + bbox[2] - bbox[0]) / len(content))
                    kargs[prefix_s + 'spacing'] = spacing
                    #if not bbox:
                    #    text_width = right - left
                    #    text_height = bottom - top
                    #    if spacing and spacing > 0:
                    #        text_width = right - left + text_width * spacing

                    #    bbox = [max(0, xx - 30), 
                    #            max(0, yy - 30), 
                    #            min(img_width, xx + text_width + 30), 
                    #            min(img_height, yy + text_height + 30)] 
                    #region = template_image.crop(bbox)
                    #target_sample_image.paste(region, bbox)

                    textcolor = (xc, yc, zc)
                    if spacing:
                        write_with_spacing(imp, xx, yy, content, textcolor, test_mf, stroke_width, spacing)
                        #write_with_spacing1(imp, xx, yy, content, textcolor, test_mf, spacing)
                    else:
                        imp.text((xx, yy), content, textcolor, font=test_mf, stroke_width=int(stroke_width))
                        #imp.text((xx, yy), content, textcolor, font=test_mf)

                    if blur_ratio:
                        im2 = template_image.crop(bbox)
                        im2 = im2.filter(ImageFilter.GaussianBlur(radius=blur_ratio))
                        Image.Image.paste(template_image, im2, (bbox[0], bbox[1]))
                copy_areas = kargs.get('copy_areas', None)
                if copy_areas:
                    for ca, ca_value in copy_areas.items():
                        bbox = ca_value['bbox']
                        im = target_sample_image.crop(bbox)
                        Image.Image.paste(template_image, im, (bbox[0], bbox[1]))

                generated_np = np.array(template_image)
                sv, _ = ssim(sample_np, generated_np, full=True, multichannel=True, channel_axis=-1)

                psnr_val = psnr(
                        sample_np, generated_np, data_range=get_data_range(sample_np)
                )
                #if sv < 0.6:
                #    continue
                ssims.append(sv)
                psnrs.append(psnr_val)
                if with_model or testing: 
                    generated_name = f"{tmpdirname}/{outpath_prefix}_generated_{filename}"
                    generated_inpaint_name = f"{tmpdirname}/{outpath_prefix}_generated_inpaint_{filename}"
                    target_inpaint_name = f"{tmpdirname}/{outpath_prefix}_real_inpaint_{filename}"
                    template_image.save(generated_name, format='JPEG', subsampling=0, quality=save_quality1)

                    real_shape = coord_to_shape(choiced_bbox)
                    real_mask, _ = mask_from_info(sample_np, real_shape)
                    real_coord1 = coord_to_coord1(choiced_bbox)
                    target_inpaint_image =  inpaint_image(img=sample_np, coord=real_coord1, mask=real_mask, text_str=choiced_content)

                    generated_inpaint_image =  inpaint_image(img=generated_np, coord=real_coord1, mask=real_mask, text_str=choiced_content)

                    Image.fromarray(target_inpaint_image).save(target_inpaint_name)
                    Image.fromarray(generated_inpaint_image).save(generated_inpaint_name, format='JPEG', subsampling=0, quality=save_quality2)

                    generated_paths.append([generated_name, 0])
                    generated_paths.append([generated_inpaint_name, 1])
                    sample_paths.append([target_sample, 0])
                    sample_paths.append([target_inpaint_name, 1])

                if testing:
                    inpaint_generated_np = np.array(generated_inpaint_image)
                    inpaint_target_np = np.array(target_inpaint_image)
                    sv_inpaint, _ = ssim(inpaint_target_np, inpaint_generated_np, full=True, multichannel=True, channel_axis=-1)

                    psnr_inpaint = psnr(
                            inpaint_target_np, inpaint_generated_np, data_range=get_data_range(sample_np)
                    )
                    ssims_inpaint.append(sv_inpaint)
                    psnrs_inpaint.append(psnr_inpaint)


        ssim_av = sum(ssims) / len(ssims)
        if with_model:
            all_tests = eval_models(generated_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
            all_samples = eval_models(sample_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
            accs = [accuracy_score(all_samples[key][0], all_tests[key][0]) for key in all_tests.keys()]
            score = lambda0 * (sum(accs)/len(accs)) + lambda1 * ssim_av
        else:
            #score = 0.5 * ssim_av + 0.5 * (psnr_av / 40.0)
            score = ssim_av 
        if testing:
            psnr_av = sum(psnrs) / len(psnrs)
            ssim_av_inpaint = sum(ssims_inpaint) / len(ssims_inpaint)
            psnr_av_inpaint = sum(psnrs_inpaint) / len(psnrs_inpaint)
            print("SSIM values: ", ssims)
            print("PSNR values: ", psnrs)
            print("SSIM inpaint values: ", ssims_inpaint)
            print("PSNR inpaint values: ", psnrs_inpaint)

            save_test_files(generated_paths, destination_dir = "data/outputs/template_test/")
            all_tests = eval_models(generated_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
            all_samples = eval_models(sample_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
            accs = [accuracy_score(all_samples[key][0], all_tests[key][0]) for key in all_tests.keys()]
            print(f"Models:{all_tests.keys()}, Model consistency: {accs}, Evaluation score: {score}, ssim: {ssim_av}, psnr: {psnr_av}")
            print(f"Evaluation score on inpaint:, ssim: {ssim_av_inpaint}, psnr: {psnr_av_inpaint}")
    return score

def evaluate_parameters(**kargs):
    xx = int(kargs['xp'])
    yy = int(kargs['yp'])
    font_size = kargs.get('font_size', None)
    if font_size:
        font_size = int(font_size)
    target_height = kargs.get('text_height', None)
    blur_ratio = kargs.get('blur_ratio', None)
    spacing = kargs.get('spacing', None)
    if spacing:
        spacing = int(spacing)
    stroke_width = round(kargs['stroke_width'])
    font_style_idx = int(kargs['font_style_idx'])
    xc = int(kargs['r'])
    yc = int(kargs['g'])
    zc = int(kargs['b'])
    save_quality = int(kargs['save_quality'])
    segment_key = kargs['segment']
    confs = kargs['confs']
    testing = kargs['testing']
    candidate_models = kargs['candidate_models']
    with_model = kargs['with_model']
    lambda0 = kargs['lambda0']
    lambda1 = kargs['lambda1']
    area = kargs['area']



    quality = int(save_quality)
    annotation_path = confs['target_annotation_path']
    template_path = confs['template_path']
    font_path = confs["text_fonts_path"]
    
    if 'font_files' in confs:
        font_styles = confs['font_files']
    else:
        font_styles = get_font_styles(font_path)
    font_file = font_styles[int(font_style_idx)]

    template_image = Image.open(template_path).convert("RGB")
    #area, segment_key = segment.split('_', 1)
    bbox = confs['segments'][segment_key].get('bbox', None)
    if testing:
        val_datas = confs['test_data']
    else:
        val_datas = confs['val_data']
    annotations = load_annotations(annotation_path)
    test_paths = []
    ssims = []
    sample_paths = []

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        for filename, values in annotations.items():
            #if filename in val_datas:
            if filename.split('_')[-1]  == "00.jpg":
                target_sample = "data/templates/Images/reals/" + filename
                #target_sample = val_datas[filename]
                target_sample_image = Image.open(target_sample).convert("RGB")
                img_width, img_height = target_sample_image.size
                content = values[segment_key]['value']
                if target_height:
                    font_size, height = find_font_size_for_height(font_file, target_height)
                test_mf = ImageFont.truetype(font_file, font_size)
                left, top, right, bottom = test_mf.getbbox(content)
                if not bbox:
                    text_width = right - left
                    text_height = bottom - top
                    if spacing and spacing > 0:
                        text_width = right - left + text_width * spacing

                    bbox = [max(0, xx - 30), 
                            max(0, yy - 30), 
                            min(img_width, xx + text_width + 30), 
                            min(img_height, yy + text_height + 30)] 
                sample_np = np.array(target_sample_image.crop(bbox))
                region = template_image.crop(bbox)
                target_sample_image.paste(region, bbox)

                textcolor = (xc, yc, zc)
                imp = ImageDraw.Draw(target_sample_image)
                if spacing:
                    write_with_spacing(imp, xx, yy, content, textcolor, test_mf, stroke_width, spacing)
                else:
                    imp.text((xx, yy), content, textcolor, font=test_mf, stroke_width=int(stroke_width))

                if blur_ratio:
                    im2 = target_sample_image.crop(bbox)
                    im2 = im2.filter(ImageFilter.GaussianBlur(radius=blur_ratio))
                    Image.Image.paste(target_sample_image, im2, (bbox[0], bbox[1]))

                generated_np = np.array(target_sample_image.crop(bbox))
                sv, _ = ssim(sample_np, generated_np, full=True, multichannel=True, channel_axis=-1)
                ssims.append(sv)
                real_name = f"{tmpdirname}/newBO_{segment_key}_{filename}"
                target_sample_image.save(real_name, format='JPEG', subsampling=0, quality=quality)
                test_paths.append([real_name, 0])
                sample_paths.append([target_sample, 0])
        if with_model:
            all_tests = eval_models(test_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
            all_samples = eval_models(sample_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
            accs = [accuracy_score(all_samples[key][0], all_tests[key][0]) for key in all_tests.keys()]
            score = lambda0 * (sum(accs)/len(accs)) + lambda1 * ((sum(ssims) / len(ssims)))
        else:
            score = sum(ssims) / len(ssims)
        if testing:
            save_test_files(test_paths, destination_dir = "data/outputs/template_test/")
            all_tests = eval_models(test_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
            all_samples = eval_models(sample_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
            accs = [accuracy_score(all_samples[key][0], all_tests[key][0]) for key in all_tests.keys()]
            print(f"Models:{all_tests.keys()}, Model consistency: {accs}, Evaluation score: {score}")
    return score

def evaluate_best_parameters(**kargs):
    xx = int(kargs['x_position'])
    yy = int(kargs['y_position'])
    font_size = kargs.get('font_size', None)
    if font_size:
        font_size = int(font_size)
    target_height = kargs.get('text_height', None)
    blur_ratio = kargs.get('blur_ratio', None)
    spacing = kargs.get('spacing', None)
    if spacing:
        spacing = int(spacing)
    stroke_width = round(kargs['stroke_width'])
    xc = int(kargs['color_r'])
    yc = int(kargs['color_g'])
    zc = int(kargs['color_b'])
    segment_key = kargs['segment']
    confs = kargs['confs']
    testing = kargs['testing']



    annotation_path = confs['target_annotation_path']
    template_path = confs['template_path']
    font_path = confs["text_fonts_path"]
    
    font_file = os.path.join(font_path, kargs['font_style'])

    template_image = Image.open(template_path).convert("RGB")
    #area, segment_key = segment.split('_', 1)
    bbox =  None #confs['segments'][segment_key].get('bbox', None)
    annotations = load_annotations(annotation_path)
    test_paths = []
    ssims = []
    sample_paths = []

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        for filename, values in annotations.items():
            #if filename in val_datas:
            if filename.split('_')[-1]  == "00.jpg":
                target_sample = "data/templates/Images/reals/" + filename
                #target_sample = val_datas[filename]
                target_sample_image = Image.open(target_sample).convert("RGB")
                img_width, img_height = target_sample_image.size
                content = values[segment_key]['value']
                if target_height:
                    font_size, height = find_font_size_for_height(font_file, target_height)
                test_mf = ImageFont.truetype(font_file, font_size)
                left, top, right, bottom = test_mf.getbbox(content)
                if not bbox:
                    text_width = right - left
                    text_height = bottom - top
                    if spacing and spacing > 0:
                        text_width = right - left + len(content) * spacing

                    bbox = [max(0, xx - 30), 
                            max(0, yy - 30), 
                            min(img_width, xx + text_width + 30), 
                            min(img_height, yy + text_height + 30)] 
                target_segment = target_sample_image.crop(bbox)
                combined = target_sample_image.copy()
                cimp = ImageDraw.Draw(combined)

                sample_np = np.array(target_segment)
                region = template_image.crop(bbox)
                target_sample_image.paste(region, bbox)

                textcolor = (xc, yc, zc)
                imp = ImageDraw.Draw(target_sample_image)
                if spacing:
                    write_with_spacing(imp, xx, yy, content, textcolor, test_mf, stroke_width, spacing)
                    write_with_spacing(cimp, xx, yy, content, textcolor, test_mf, stroke_width, spacing)
                else:
                    imp.text((xx, yy), content, textcolor, font=test_mf, stroke_width=int(stroke_width))
                    cimp.text((xx, yy), content, textcolor, font=test_mf, stroke_width=int(stroke_width))

                if blur_ratio:
                    im2 = target_sample_image.crop(bbox)
                    im2 = im2.filter(ImageFilter.GaussianBlur(radius=blur_ratio))
                    Image.Image.paste(target_sample_image, im2, (bbox[0], bbox[1]))

                generated_segment = target_sample_image.crop(bbox)
                generated_np = np.array(target_sample_image.crop(bbox))
                combined_segment = combined.crop(bbox)
                sv, _ = ssim(sample_np, generated_np, full=True, multichannel=True, channel_axis=-1)
                print(sv)
                return target_segment, generated_segment, combined_segment

