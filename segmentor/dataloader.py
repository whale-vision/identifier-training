import json
import os
import numpy as np
import torch
from PIL import Image

ANNOTATION_FILE = 'G:\Whale Stuff\_data\WhalesV3\.exports\coco-1646094622.351781.json'
# ANNOTATION_FILE = './datasets/WhalesV3/.exports/coco-1646094622.351781.json'
TARGET_WIDTH = 224
TARGET_HEIGHT = 224

def readImage(imagePath: str):
        img = Image.open(imagePath)

        img.convert('RGB')
        img = img.resize((TARGET_WIDTH, TARGET_HEIGHT))
        img = np.array(img).astype(np.float32)
        img /= 255.0

        #reshape from (h, w, c) to (c, h, w)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)

        return img

class WhaleDataset():
    def __init__(self):
        self.classes = ["flank", "fluke"]

        # read annotation file
        with open(ANNOTATION_FILE, 'r') as f:
            annotations = json.load(f)

        self.images = annotations['images']
        self.catagories = annotations['categories']
        self.annotations = annotations['annotations']

    def getBoxes(self, index):
        image = self.images[index]

        annotations = list(filter(lambda x: x['image_id'] == image['id'], self.annotations))
        boxes = list(map(lambda x: x['bbox'], annotations))
        labels = list(map(lambda x: x['category_id'], annotations))

        width = image['width']
        height = image['height']

        normalised_boxes = list(map(
            lambda x: [
                x[0] * TARGET_WIDTH / width, 
                x[1] * TARGET_HEIGHT / height, 
                (x[0] + x[2]) * TARGET_WIDTH / width, 
                (x[1] + x[3]) * TARGET_HEIGHT / height
            ],
            boxes
        ))

        return normalised_boxes, labels

    def __getitem__(self, index):
        image_details = self.images[index]
        img_path = os.path.abspath('../../../db/lpxsf2' + image_details['path'])
        # img_path = os.path.abspath('.' + image_details['path'])
        img = readImage(img_path)

        boxes, labels = self.getBoxes(index)

        img = torch.tensor(img)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        image_id = torch.tensor([index])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area   
        target["iscrowd"] = iscrowd

        return img, target
    
    def __len__(self):
        return len(self.images)
    

print(len(WhaleDataset()))