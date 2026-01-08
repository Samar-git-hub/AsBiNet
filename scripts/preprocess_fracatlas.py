import pandas as pd
from pycocotools.coco import COCO
import torch

# Test loading one of the split files
train_df = pd.read_csv(r'data\FracAtlas\Utilities\Fracture Split\train.csv')
print(f"Total training images according to CSV: {len(train_df)}")

# Test if GPU is available 
print(f"Is CUDA available: {torch.cuda.is_available()}")