import os
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import DATASETS

import mmseg_fracatlas_dataset

def main():

    cfg = Config.fromfile('mmseg_workspace/configs/deeplabv3_resnet50.py')
    
    cfg.work_dir = './mmseg_experiments/deeplabv3_resnet50'

    runner = Runner.from_cfg(cfg)

    print(f"Starting training with MMSegmentation")
    runner.train()

if __name__ == '__main__':
    main()