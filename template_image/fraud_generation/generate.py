#!/usr/bin/env python
# coding=utf-8
import sys

from data.DataGenerator.Midv2020 import Template_Generator
from data.DataLoader.Datasets import *

## Downloadign our data just as an example
#data = SIDTD(download_original=False).download_dataset("templates")

# get the abosulte path where the data is stored following the structure depicted above

# you need to go where the data is 
#path_dataset = "../SIDTD/fin2/"
path_dataset = sys.argv[1] # base path 
img_path = sys.argv[2] #relative image path
annotation_path = sys.argv[3] #relative annotation path


# generating the data
#def __init__(self, absolute_path:str, imgs_path:str, annotation_path:str, fake_template:dict = None, delta_boundary:int=10, class_image:str = None):
gen = Template_Generator.Template_Generator(absolute_path=path_dataset, imgs_path=img_path, annotation_path = annotation_path)

gen.create_inpaint_and_rewrite(path_store=None)
gen.create_crop_and_replace(path_store=None)

#gen.store_generated_dataset(path_store=None) #[None for dedault]

