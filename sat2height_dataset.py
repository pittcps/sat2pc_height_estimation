import os
from PIL import Image
import skimage.color
import skimage.io
import skimage.transform
import numpy as np
import torch
import json
import math
from generalized_dataset import GeneralizedDataset
import random



class Sat2HeightDataset(GeneralizedDataset):
    def __init__(self, image_dir, ann_dir, label_dir, mode, bootstarp, transform=None):
        super().__init__()

        self.mode = mode
        self.image_dir = image_dir
        self.ann_dir = ann_dir
        self.image_info = []
        self.ids = []
        self.transform = transform
        
        f = open(label_dir)
        self.annotation_dict = json.load(f)
        f.close()

        images = [os.path.splitext(name)[0] for name in os.listdir(ann_dir)]

        for i, img_name in enumerate(images):
            if bootstarp:
                if self.annotation_dict[img_name]['z_mean'] < 450:
                    self.add_image("roofs", image_id=img_name, path=os.path.join(image_dir, img_name + '.png'))
                else:
                    self.add_image("roofs", image_id=img_name, path=os.path.join(image_dir, img_name + '.png'))
                    self.add_image("roofs", image_id=img_name, path=os.path.join(image_dir, img_name + '.png'))
                    self.add_image("roofs", image_id=img_name, path=os.path.join(image_dir, img_name + '.png'))
                    self.add_image("roofs", image_id=img_name, path=os.path.join(image_dir, img_name + '.png'))
            else:
                self.add_image("roofs", image_id=img_name, path=os.path.join(image_dir, img_name + '.png'))
        
    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": int(image_id),
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)
        self.ids.append(len(self.ids))

    def get_number_of_samples(self):
        return len(self.image_info)

    def get_image(self, img_id):
        image_id = int(img_id)
        image = skimage.io.imread(self.image_info[image_id]['path'])[..., :3]
        return image
    
    def get_annotation(self, image_id):
        info = self.image_info[image_id]
        im_path = info['path']
        file_name =  os.path.splitext(os.path.basename(im_path))[0]
        ann_path = os.path.join(self.ann_dir, file_name + '.json')

        f = open(ann_path)
        ann = json.load(f)
        f.close()
        return ann
    
    def get_mask(self, img_id):
        ann = self.get_annotation(img_id)
        masks = ann['masks']
        aggregated_mask = np.asarray(masks[0])
        for i in range(1, len(masks)):
            aggregated_mask = np.logical_or(aggregated_mask, np.asarray(masks[i]))

        return aggregated_mask.T


    def get_sample_id(self, img_name):
        idx = -1
        for i, info in enumerate(self.image_info):
            if info['id'] == img_name:
                idx = i 

        if idx == -1:
            print('Sample does not exists')
            assert(False)
 
        return idx

    def __getitem__(self, i):
        img_id = self.ids[i]
        image = self.get_image(img_id)
        mask = self.get_mask(img_id)
        info = self.image_info[img_id]
        label = [self.annotation_dict[str(info['id'])]['z_mean'], 
                 self.annotation_dict[str(info['id'])]['z_std'], 
                 self.annotation_dict[str(info['id'])]['x_std'], 
                 self.annotation_dict[str(info['id'])]['y_std']]

        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
            label = torch.tensor(label)
            mask = torch.tensor(mask)
        
        if self.mode == 'test':
            return image, mask, label, info['id']
        else: 
            return image, mask, label

    