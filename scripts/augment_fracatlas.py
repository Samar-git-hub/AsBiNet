import albumentations as A
import cv2
import numpy as np
import os 
import pandas as pd
import json
from tqdm import tqdm

root_dir = 'data/FracAtlas'
aug_output_dir = 'data/FracAtlas/Augmented'
orig_resized_dir = 'data/FracAtlas/processed/original'

multiplier = 50
target_size = 512

aug_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
    A.GaussianBlur(p=0.5),
    A.RandomGamma(p=0.5),

    # Resize preserving aspect ratio
    A.LongestMaxSize(max_size=target_size),
    A.PadIfNeeded(min_height=target_size, min_width=target_size, border_mode=cv2.BORDER_CONSTANT, value=0, max_value=0)
])

def load_coco_data(root_dir):

    # Loading annotations (for creating a mask)
    json_path = os.path.join(root_dir, 'Annotations', 'COCO JSON', 'COCO_fracture_masks.json')

    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Filename lookup
    img_id_map = { img['id']: img['file_name'] for img in coco_data['images']}

    # Coordinates lookup
    ann_map = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']

        if img_id not in ann_map:
            ann_map[img_id] = []
        
        ann_map[img_id].append(ann['segmentation'])

    return img_id_map, ann_map
        
    