import os
import random
import numpy as np
from PIL import Image
from pycocotools.coco import COCO as COCOApi
from pycocotools import mask as maskUtils

#COCO IDs to exclude
#Person, Bicycle, Car, Motorcycle, Bus, Train, Truck, Traffic Light, Stop Sign
COCO_CITYSCAPES_EXCLUSION = [1, 2, 3, 4, 6, 7, 8, 10, 13]

class COCOLoader:
    def __init__(self, root: str, split: str = "train2017", exclusion_list=None, max_samples=2000):
        self.root = root
        self.split = split
        self.max_samples = max_samples
        exclusion_list = exclusion_list if exclusion_list is not None else COCO_CITYSCAPES_EXCLUSION
        
        ann_file = os.path.join(root, "annotations", f"instances_{split}.json")
        
        print("OOD Loader: Parsing COCO JSON...")
        coco = COCOApi(ann_file)
        
        all_cats = coco.getCatIds()
        valid_cats = list(set(all_cats) - set(exclusion_list))
        
        img_ids = []
        for cat_id in valid_cats:
            img_ids.extend(coco.getImgIds(catIds=[cat_id]))
        img_ids = list(set(img_ids))
        
        if len(img_ids) > self.max_samples:
            img_ids = random.sample(img_ids, self.max_samples)
            
        print(f"OOD Loader: Selected {len(img_ids)} samples.")

        self.data = []
        
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            file_name = img_info['file_name']
            
            ann_ids = coco.getAnnIds(imgIds=img_id, catIds=valid_cats, iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            
            if anns:
                self.data.append((file_name, anns))
                
        del coco

    def __len__(self):
        return len(self.data)

    def annToMask(self, ann, height, width):
        if 'segmentation' in ann:
            if type(ann['segmentation']) == list:
                rles = maskUtils.frPyObjects(ann['segmentation'], height, width)
                rle = maskUtils.merge(rles)
            elif type(ann['segmentation']['counts']) == list:
                rle = maskUtils.frPyObjects(ann['segmentation'], height, width)
            else:
                rle = ann['segmentation']
        else:
            return np.zeros((height, width), dtype=np.uint8)
            
        m = maskUtils.decode(rle)
        return m

    def get_random_sample(self):
        file_name, anns = random.choice(self.data)
        
        img_path = os.path.join(self.root, self.split, file_name)
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        
        ann = random.choice(anns)
        mask = self.annToMask(ann, h, w)
        
        return np.array(image), mask
