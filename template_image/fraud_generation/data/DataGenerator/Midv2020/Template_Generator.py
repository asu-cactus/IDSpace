import matplotlib.pyplot as plt

from data.DataGenerator.Midv  import Midv
from utils.util  import *

from typing import *
from unicodedata import name
from PIL import ImageFont, ImageDraw, Image

import os
import random
import tqdm


class Template_Generator(Midv):

    #__slots__ = ["_img_loader", "_classes", "_fake_template", "_transformations","_fake_img_loader","_annotations_path","_imgs_path","_delta_boundary","_static_path", "_flag"]

    def __init__(self, absolute_path:str, imgs_path:str, annotation_path:str, fake_template:dict = None, delta_boundary:int=10 ):


        """
            The initialitation of the Generator just create the structure with Images in memory that will serve as a template to create the 
            new information. 
        """
        
        if fake_template is None:
            self._fake_template = super().MetaData

        #assert isinstance(fake_template, dict), "The metadata template to save the information must be a dict"


        super().__init__(absolute_path)

        self.ALLOWED_EXTENSIONS = ('.jpg', '.jpeg', '.png')
        #path_template = super().get_template_path()
        #path_annotatons = super().get_img_annotations_path()
        self._delta_boundary = delta_boundary
        self._annotations_path = os.path.join(absolute_path, annotation_path)
        self._imgs_path = os.path.join(absolute_path, imgs_path)

        self._img_loader = [] 

        self.create_loader()



    #def create_and_map_classes_imgs(self):
    #    map_class = {
    #        
    #    }
    #    for image in os.listdir(self._imgs_path):
    #        class_image = image.split("_")[0]
    #        map_class[class_image] = []
    #    return map_class
    
    def create_and_map_classes_annotations(self):
        map_annotation = read_json(self._annotations_path)                
        return map_annotation    
            

    def create_loader(self) -> List[object]:
        map_annotations = self.create_and_map_classes_annotations()

        for im in tqdm.tqdm(os.listdir(self._imgs_path), desc="create_loader"):
            if not im.lower().endswith(self.ALLOWED_EXTENSIONS):
                continue
            #clas, original, id = im.split("_")
            name_img = im
            src_img = None #os.path.join(self._static_path_images, clas,original, id)
            img = read_img(os.path.join(self._imgs_path, im))

            class_template = map_annotations
            self._img_loader.append(super(Template_Generator, self).Img(img,class_template,name_img,src_img))

    def create_inpaint_and_rewrite(self, path_store: Optional[str] = None) -> List[Image.Image]:

        if path_store is None:
            path = self.absoulute_path+"/SIDTD_Generated"
        else:
            path = path_store

        print(f"Data beeing stored in the path: {path}")
        
        img_bucket = self._img_loader
        meta_dates = {}
        inpaint_path = f"{self.absoulute_path}/inpaint_and_rewrite/"
        for idx in tqdm.tqdm(range(len(img_bucket)), desc="inpaint"): # First, make a transformation for any kind of document

            img = random.choice(img_bucket)

            img_id = idx #int(img._relative_path.split("/")[-1].split(".")[0])
            fake_img, swap_info, meta_data =  super().Inpaint_and_Rewrite(img=img._img,img_id=img._name,info=img._meta)
            #print("field:", type(field))
            name_fake_generated =  img._name[:-4] + "_fake_" + str(idx)


            #Creating the dict with the metadata
            #fake_meta = vars(self._fake_template(src=img._relative_path, type_transformation="Inpaint_and_Rewrite",field=field,loader="Midv2020",name=name_fake_generated))

            generated_img = super().Img(img._img, img._meta, img._name)

            #generated_img.fake_meta = fake_meta
            generated_img.fake_name = name_fake_generated #fake_meta["name"]
            generated_img.fake_img = fake_img
            md = {}
            md['fraud'] = True
            md['src'] = swap_info
            md['des'] = meta_data
            meta_dates[name_fake_generated] = md
            meta_dates['save_quality'] = int(random.uniform(60, 100))
            store1(generated_img, inpaint_path, meta_dates['save_quality'])

            #self._fake_img_loader.append(generated_img)
        
        #print(f"General Inpainted for the class {key} Done")
        # equal generation
        write_json(meta_dates, f"{self.absoulute_path}", f"inpaint_and_rewrite")
    def create_crop_and_replace(self, path_store: Optional[str] = None) -> List[Image.Image]:

        if path_store is None:
            path = self.absoulute_path+"/SIDTD_Generated"
        else:
            path = path_store

        print(f"Data beeing stored in the path: {path}")
        
        img_bucket = self._img_loader
        meta_dates = {}
        crop_path = f"{self.absoulute_path}/crop_and_replace/"
        ran_choices = list(range(int(len(img_bucket)/2), len(img_bucket)))
        for counter, idx in tqdm.tqdm(enumerate(range(int(len(img_bucket)/2 + 0.5))), desc="crop"): # First, make a transformation for any kind of document

            delta1 = random.sample(range(self._delta_boundary),2)
            delta2 = random.sample(range(self._delta_boundary),2)

            img = img_bucket[idx]
            img_id2 = random.choice(ran_choices) #int(img2._relative_path.split("/")[-1].split(".")[0])
            img2 = img_bucket[img_id2]
            md1 = {}
            md2 = {}
            src = {}
            des = {}
            src['name'] = img._name
            des['name'] = img2._name
            fake_img1, fake_img2 , swap_info_1, swap_info_2 = super().Crop_and_Replace(img1=img._img, img2=img2._img, info=img._meta, additional_info=None, img_id1=img._name ,img_id2=img2._name,delta1=delta1,delta2=delta2)
            src['region'] = swap_info_1
            des['region'] = swap_info_2
            src['shift'] = [delta1, delta2]
            des['shift'] = [delta2, delta1]
            md1['fraud'] = True
            md1['src'] = src
            md1['des'] = des
            md2['fraud'] = True
            md2['src'] = des
            md2['des'] = src
            
            #img1 info
            name_fake_generated =  img._name[:-4] + "_fake_1_" + str(idx)

            #fake_meta = vars(self._fake_template(src=img._relative_path, second_src=img2._relative_path, shift=(delta1,delta2),type_transformation=name_transform,field=field, second_field=field2,loader="Midv2020",name=name_fake_generated))

            # craeting fake img1
            generated_img = super().Img(img._img, img._meta, img._name)

            generated_img.fake_meta = None #fake_meta
            generated_img.fake_name = name_fake_generated
            generated_img.fake_img = fake_img1
            generated_img.complement_img = fake_img2

            save_quality = int(random.uniform(60, 100))
            md1['save_quality'] = save_quality
            meta_dates[name_fake_generated] = md1

            store1(generated_img, crop_path, save_quality)


            #img2 info
            name_fake_generated =  img._name.split(".")[0] + "_fake_2_"  + str(idx)
            #fake_meta2 = vars(self._fake_template(second_src=img._relative_path, src=img2._relative_path, shift=(delta1,delta2),type_transformation=name_transform,second_field=field, field=field2,loader="Midv2020",name=name_fake_generated))
            

            # craeting fake img2
            generated_img2 = super().Img(img2._img, img2._meta, img2._name)

            generated_img2.fake_meta = None #fake_meta2
            generated_img2.fake_name = name_fake_generated
            generated_img2.fake_img = fake_img2
            generated_img2.complement_img = fake_img1

            save_quality = int(random.uniform(60, 100))
            md2['save_quality'] = save_quality
            meta_dates[name_fake_generated] = md2
            store1(generated_img2, crop_path, save_quality)

            #self._fake_img_loader.append(generated_img)
        
        #print(f"General Inpainted for the class {key} Done")
        # equal generation
        write_json(meta_dates, f"{self.absoulute_path}", f"crop_and_replace")



    def store_generated_dataset(self, path_store: Optional[str] = None):

        if path_store is None:
            path = self.absoulute_path+"/SIDTD_Generated"
        else:
            path = path_store

        print(f"Data beeing stored in the path: {path}")
        store(self._fake_img_loader, path_store=path)




if __name__ == "__main__":
    gen = Template_Generator("/home/cboned/Midv2020/dataset/SIDTD")
    
    gen.create(5)
    
    gen.store_generated_dataset()
