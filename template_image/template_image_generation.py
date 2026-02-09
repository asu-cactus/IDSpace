# Copyright 2025, Arizona State University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from utils import *

from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageOps
from rembg import remove
import json
import os
import random
import math
import tqdm
import argparse
import os
import copy

parser = argparse.ArgumentParser(description="Configurable paths for document generation")


parser.add_argument('--best_setting_path', type=str, default="data/outputs/SVK/template_best_settings.json", help='Path to parameters JSON')

args = parser.parse_args()

with open( args.best_setting_path, 'r') as f:
    confs = json.load(f)

text_font_paths = confs['text_fonts_path']
templatepath = confs['template_path']
segments = confs['segments']
annotationfile = confs['output']['synthetic_images_annotation_path']

personal_info = confs['personal_info']
portrait_images_path = confs['portrait_images_path']
signature_fonts_path = confs['signature_fonts']
oppath = confs['output']['synthetic_images_path']  

id_number = 1
piis = load_pii_info(personal_info)
opfiles = os.listdir(oppath)


outerdict = {}


ctr = 1
clean_template = Image.open(templatepath)
if 'portrait_background' in segments:
    keys = segments['portrait_background']
    bbox = keys['bbox']
    image_path = keys['image_path']
    source_image = Image.open(image_path)
    write_bg(clean_template, source_image, bbox)
    
for data in piis:

    innerdict = {}
    tempfilename = f"generated_{str(ctr)}.png" 
    print(f"{ctr} - {tempfilename}")
    ctr += 1

        
    template = copy.deepcopy(clean_template)
    
    innerdict["fraud"] = "False"

    image_keys = {}
    signature_keys = {}
    for key in segments:                                                                                                                                                                                                                                                
        segment_type = segments[key]['type']
        if segment_type == 'text':
            parameters = copy.deepcopy(segments[key]['font_info'])
            content = data[key]                                                                                                                                                                                                                                                   
            coord = write_parameters(**parameters, content=content, template = template, text_font_paths = text_font_paths)                                                                                                                                                             
            parameters['value'] = content
            parameters['bbox'] = list(coord)
            innerdict[key] = parameters
        elif segment_type == 'image':
            image_keys[key] = segments[key]
            image_keys[key]['value'] = data[key]
        elif segment_type == 'signature':
            signature_keys = segments[key]
            signature_keys['value'] = data[key]
        else:
            print("Unsupported type! Please check again!")
    write_images(template, image_keys, portrait_images_path, innerdict)
        
    id_number += 1
    signaturetext = data['signature']

    write_signature(template, signature_fonts_path, signature_keys, innerdict)

    save_quality = confs.get('save_quality',(confs.get('save_quality1', 100)))
    template.save(os.path.join(oppath, tempfilename), "JPEG", quality=save_quality)
    outerdict[tempfilename] = innerdict


write_json(outerdict, annotationfile)

