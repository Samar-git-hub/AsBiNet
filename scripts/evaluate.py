import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm
from medpy.metric.binary import hd95
from thop import profile
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")
root_dir = 'data/FracAtlas'
test_csv_path = 'data/FracAtlas/processed/original/test.csv'
high_res_mask_dir = 'data/FracAtlas/masks/Fractured'
target_size = 512


